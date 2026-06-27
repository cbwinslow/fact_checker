"""Pipeline harness - orchestrates all agents in sequence.

Flow:
  ingest -> analyze_images (video frames) -> extract_claims
         -> retrieve_evidence -> draft_verdicts

The image analysis stage (Stage 2) is inserted between ingest and claim
extraction. When a video is ingested, key frames are extracted from the
media file and each frame is analysed by the vision LLM. Visible text
claims found in frames are merged into the claim extraction context.

Supports optional live DB status updates between stages so the
GET /jobs/{job_id} polling endpoint reflects real progress.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from .models import ImageAnalysis, ImageSourceType, JobStatus, PipelineResult, VideoJob
from .services.ingest import ingest
from .agents.claim_extractor import extract_claims
from .agents.evidence_agent import retrieve_evidence
from .agents.verdict_agent import draft_verdicts
from .agents.image_analyst import analyse_images

log = logging.getLogger(__name__)


async def _update_job_status(
    job_id,
    status: JobStatus,
    error: Optional[str] = None,
    session=None,
) -> None:
    """Write live job status to DB if a session is provided.

    Falls back to a no-op when session is None so harness can still
    be called from tests or CLI without a DB connection.
    """
    if session is None:
        return
    try:
        from sqlalchemy import update
        from .db import VideoJobRow
        values = {"status": status.value, "updated_at": datetime.utcnow()}
        if error is not None:
            values["error"] = error
        await session.execute(
            update(VideoJobRow)
            .where(VideoJobRow.id == job_id)
            .values(**values)
        )
        await session.flush()
        log.debug("[harness] Job %s status -> %s", job_id, status.value)
    except Exception as exc:  # pragma: no cover
        log.warning("[harness] Failed to write status for job %s: %s", job_id, exc)


async def _extract_frames_for_job(
    job_id,
    local_path: Optional[Path],
    artifact_dir: Path,
) -> List[Path]:
    """Extract video frames for image analysis.

    Returns an empty list (no-op) when local_path is None or not a video file.
    Only runs frame extraction for video files (not for audio-only or URL-only jobs).
    """
    if not local_path or not local_path.exists():
        return []

    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v"}
    if local_path.suffix.lower() not in VIDEO_EXTENSIONS:
        return []

    try:
        from .services.vision import extract_frames
        frames_dir = artifact_dir / str(job_id) / "frames"
        frames = extract_frames(
            video_path=local_path,
            output_dir=frames_dir,
            interval_sec=30.0,
            max_frames=20,
        )
        log.info("[harness] Extracted %d frames for job %s", len(frames), job_id)
        return frames
    except Exception as exc:
        # Frame extraction is best-effort - a failure here should not abort the pipeline
        log.warning(
            "[harness] Frame extraction failed for job %s (skipping vision stage): %s",
            job_id, exc,
        )
        return []


async def run_pipeline(
    url: Optional[str] = None,
    local_path: Optional[Path] = None,
    job_id=None,
    session=None,
    image_paths: Optional[List[str]] = None,
) -> PipelineResult:
    """Run the full fact-checking pipeline and return a PipelineResult.

    Args:
        url:         YouTube / web URL to fact-check.
        local_path:  Path to a local audio/video file.
        job_id:      Pre-generated UUID for the job (generated if None).
        session:     Optional AsyncSession. When provided, live status updates
                     are written to the DB between pipeline stages.
        image_paths: Optional list of image file paths to analyse directly.
                     Useful for image-only fact-checking jobs.
    """
    if job_id is None:
        job_id = uuid4()

    job = VideoJob(id=job_id, url=url, local_path=str(local_path) if local_path else None)

    segments: list = []
    images:   list = []
    claims:   list = []
    evidence: list = []
    verdicts: list = []

    try:
        # ----------------------------------------------------------------
        # Stage 1: Ingest (audio/video/captions)
        # ----------------------------------------------------------------
        log.info("[harness] Stage 1/5: Ingesting media...")
        job.status = JobStatus.INGESTING
        await _update_job_status(job_id, JobStatus.INGESTING, session=session)

        segments, ingest_source = await ingest(job_id=job_id, url=url, local_path=local_path)
        job.ingest_source = ingest_source
        log.info("[harness] Ingested %d segments via %s", len(segments), ingest_source)

        # ----------------------------------------------------------------
        # Stage 2: Image / frame analysis (vision)
        # ----------------------------------------------------------------
        log.info("[harness] Stage 2/5: Analysing images/frames...")
        job.status = JobStatus.ANALYZING
        await _update_job_status(job_id, JobStatus.ANALYZING, session=session)

        from .config import settings as _settings
        artifact_dir = _settings.artifact_dir

        # Collect image paths: user-provided + auto-extracted frames
        all_image_paths: List[Path] = []
        frame_timestamps: List[float] = []
        source_type = ImageSourceType.VIDEO_FRAME

        if image_paths:
            # Direct image paths supplied by the caller (e.g. image upload endpoint)
            all_image_paths = [Path(p) for p in image_paths]
            source_type = ImageSourceType.UPLOAD
        elif local_path:
            # Try to extract frames from a video file
            frames = await _extract_frames_for_job(job_id, local_path, artifact_dir)
            all_image_paths = frames
            # Estimate frame timestamps from the filename pattern frame_NNNN.jpg
            for f in frames:
                try:
                    n = int(f.stem.split("_")[-1])  # frame_0001 -> 1
                    frame_timestamps.append(float(n) * 30.0)  # interval_sec=30
                except Exception:
                    frame_timestamps.append(0.0)

        if all_image_paths:
            images = await analyse_images(
                job_id=job_id,
                image_paths=all_image_paths,
                source_type=source_type,
                frame_timestamps=frame_timestamps or None,
            )
            log.info("[harness] Analysed %d images", len(images))
        else:
            log.info("[harness] No images to analyse (video-frame extraction skipped or no images provided)")

        # ----------------------------------------------------------------
        # Stage 3: Claim extraction
        # Merge visible_claims from image analyses into context
        # ----------------------------------------------------------------
        log.info("[harness] Stage 3/5: Extracting claims...")
        job.status = JobStatus.EXTRACTING
        await _update_job_status(job_id, JobStatus.EXTRACTING, session=session)

        claims = await extract_claims(job_id=job_id, segments=segments)

        # Promote visible_claims from each ImageAnalysis as additional Claim objects
        from .models import Claim
        for img_analysis in images:
            for visible_text in img_analysis.visible_claims:
                if visible_text.strip():
                    claims.append(Claim(
                        job_id=job_id,
                        image_id=img_analysis.id,
                        text=visible_text.strip(),
                        is_checkable=True,
                        confidence=0.80,
                        context=f"Visible text in video frame at {img_analysis.frame_sec}s"
                                if img_analysis.frame_sec is not None
                                else "Visible text in image",
                    ))

        log.info("[harness] Extracted %d claims (%d from images)", len(claims),
                 sum(1 for c in claims if c.image_id is not None))

        # ----------------------------------------------------------------
        # Stage 4: Evidence retrieval
        # ----------------------------------------------------------------
        log.info("[harness] Stage 4/5: Retrieving evidence...")
        job.status = JobStatus.RETRIEVING
        await _update_job_status(job_id, JobStatus.RETRIEVING, session=session)

        evidence = await retrieve_evidence(claims=claims)
        log.info("[harness] Retrieved %d evidence items", len(evidence))

        # ----------------------------------------------------------------
        # Stage 5: Verdict drafting
        # ----------------------------------------------------------------
        log.info("[harness] Stage 5/5: Drafting verdicts...")
        job.status = JobStatus.VERDICTING
        await _update_job_status(job_id, JobStatus.VERDICTING, session=session)

        verdicts = await draft_verdicts(claims=claims, evidence=evidence)
        log.info("[harness] Drafted %d verdicts", len(verdicts))

        job.status = (
            JobStatus.REVIEW
            if any(v.requires_human_review for v in verdicts)
               or any(i.manipulation_risk.value in ("high", "medium") for i in images)
            else JobStatus.DONE
        )
        await _update_job_status(job_id, job.status, session=session)
        log.info("[harness] Pipeline complete. Job status: %s", job.status)

    except Exception as exc:
        log.error("[harness] Pipeline failed for job %s: %s", job_id, exc, exc_info=True)
        job.status = JobStatus.FAILED
        job.error  = str(exc)
        await _update_job_status(job_id, JobStatus.FAILED, error=str(exc), session=session)
        raise

    return PipelineResult(
        job=job,
        segments=segments,
        images=images,
        claims=claims,
        evidence=evidence,
        verdicts=verdicts,
    )
