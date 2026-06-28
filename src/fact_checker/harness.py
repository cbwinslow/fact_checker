"""harness.py - Pipeline orchestrator for the fact-checker.

Orchestrates all agents and services in sequence, passing a typed
:class:`~fact_checker.models.AnalysisContext` packet between stages so
every agent has full visibility into the job state.

Full pipeline (7 stages)::

  Stage 1  ingest         -- MediaRouter routes any input type to the correct
                             ingestor (video/audio/PDF/web/text/docx/image).
  Stage 2  analyse_images -- Vision LLM analyses frames / uploaded images;
                             visible text claims are merged into context.
  Stage 3  embed          -- Transcript segments are chunked and embedded
                             into a per-job VectorStore for semantic retrieval.
  Stage 4  extract_claims -- LLM extracts atomic checkable claims from
                             transcript segments + image visible_claims.
  Stage 5  deep_research  -- DeepResearchAgent performs multi-hop research:
                             semantic retrieval, Google FC, Serper web search,
                             full-page scraping, adversarial counter-query,
                             Wikipedia lookup, source credibility scoring.
  Stage 6  draft_verdicts -- VerdictAgent generates evidence-backed verdicts.

All stages update the shared AnalysisContext in-place so failures can be
partially recovered and intermediate results are always accessible.

DB status updates are written between stages when an AsyncSession is
provided, enabling real-time progress polling via the REST API.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from .models import (
    AnalysisContext,
    Claim,
    ImageSourceType,
    JobStatus,
    PipelineResult,
    VideoJob,
)
from .services.file_router import MediaRouter
from .agents.claim_extractor import extract_claims
from .agents.verdict_agent import draft_verdicts
from .agents.image_analyst import analyse_images
from .agents.deep_research_agent import deep_research

log = logging.getLogger(__name__)

# Module-level MediaRouter singleton (stateless, safe to reuse)
_router = MediaRouter()


# ---------------------------------------------------------------------------
# DB status helper
# ---------------------------------------------------------------------------

async def _update_job_status(
    job_id,
    status: JobStatus,
    error: Optional[str] = None,
    session=None,
) -> None:
    """Write live job status to the database if a session is provided.

    Falls back to a no-op when ``session`` is ``None`` so the harness
    can be called from tests or the CLI without a DB connection.

    Args:
        job_id: UUID of the job to update.
        status: New JobStatus value.
        error:  Optional error message string.
        session: Optional SQLAlchemy AsyncSession.
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
        log.debug("[harness] Job %s -> %s", job_id, status.value)
    except Exception as exc:  # pragma: no cover
        log.warning("[harness] Status update failed for job %s: %s", job_id, exc)


# ---------------------------------------------------------------------------
# Frame extraction helper
# ---------------------------------------------------------------------------

async def _extract_frames_for_job(
    job_id,
    local_path: Optional[Path],
    artifact_dir: Path,
) -> List[Path]:
    """Extract video key frames for the image analysis stage.

    Returns an empty list (no-op) when ``local_path`` is ``None``,
    does not exist, or is not a recognised video file extension.

    Args:
        job_id:       UUID of the pipeline job.
        local_path:   Path to the local video file (or None).
        artifact_dir: Root directory for storing extracted frames.

    Returns:
        Sorted list of frame file Paths, or an empty list.
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
        log.warning(
            "[harness] Frame extraction failed for job %s (skipping vision): %s",
            job_id, exc,
        )
        return []


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

async def run_pipeline(
    url: Optional[str] = None,
    local_path: Optional[Path] = None,
    job_id=None,
    session=None,
    image_paths: Optional[List[str]] = None,
    # User-provided context
    user_context: Optional[str] = None,
    user_question: Optional[str] = None,
    focus_areas: Optional[List[str]] = None,
) -> PipelineResult:
    """Run the full fact-checking pipeline and return a PipelineResult.

    Accepts any combination of input types; the MediaRouter automatically
    selects the correct ingest pathway.  All intermediate artefacts are
    accumulated in a typed AnalysisContext packet passed between stages.

    Args:
        url:         YouTube / web / direct media URL to fact-check.
        local_path:  Path to a local file (video, audio, PDF, text, docx,
                     image - any type handled by the MediaRouter).
        job_id:      Pre-generated UUID for the job (generated if None).
        session:     Optional AsyncSession.  When provided, live status
                     updates are written to the DB between pipeline stages.
        image_paths: Optional list of image file paths to analyse directly.
                     Useful for image-only fact-checking jobs.

    Returns:
        :class:`~fact_checker.models.PipelineResult` containing all stage
        outputs.

    Raises:
        Exception: Any unhandled exception from a pipeline stage.  The job
                   is marked FAILED in the DB before re-raising.
    """
    if job_id is None:
        job_id = uuid4()

    job = VideoJob(
        id=job_id,
        url=url,
        local_path=str(local_path) if local_path else None,
    )

    # Initialise the shared AnalysisContext packet
    ctx = AnalysisContext(
        job_id=job_id,
        user_context=user_context,
        user_question=user_question,
        focus_areas=focus_areas or [],
    )

    try:
        # ----------------------------------------------------------------
        # Stage 1: Ingest
        # MediaRouter detects input type and routes appropriately
        # ----------------------------------------------------------------
        log.info("[harness] Stage 1/6: Ingesting input...")
        job.status = JobStatus.INGESTING
        await _update_job_status(job_id, JobStatus.INGESTING, session=session)

        ctx.segments, ctx.ingest_source = await _router.route(
            job_id=job_id,
            url=url,
            local_path=local_path,
            image_paths=image_paths,
        )
        job.ingest_source = ctx.ingest_source
        log.info(
            "[harness] Ingested %d segments via %s",
            len(ctx.segments), ctx.ingest_source,
        )

        # ----------------------------------------------------------------
        # Stage 2: Image / frame analysis (vision)
        # ----------------------------------------------------------------
        log.info("[harness] Stage 2/6: Analysing images/frames...")
        job.status = JobStatus.ANALYZING
        await _update_job_status(job_id, JobStatus.ANALYZING, session=session)

        from .config import settings as _settings
        artifact_dir = _settings.artifact_dir

        all_image_paths: List[Path] = []
        frame_timestamps: List[float] = []
        source_type = ImageSourceType.VIDEO_FRAME

        if image_paths:
            all_image_paths = [Path(p) for p in image_paths]
            source_type = ImageSourceType.UPLOAD
        elif local_path:
            frames = await _extract_frames_for_job(job_id, local_path, artifact_dir)
            all_image_paths = frames
            for f in frames:
                try:
                    n = int(f.stem.split("_")[-1])
                    frame_timestamps.append(float(n) * 30.0)
                except Exception:
                    frame_timestamps.append(0.0)

        if all_image_paths:
            ctx.images = await analyse_images(
                job_id=job_id,
                image_paths=all_image_paths,
                source_type=source_type,
                frame_timestamps=frame_timestamps or None,
            )
            log.info("[harness] Analysed %d images", len(ctx.images))
        else:
            log.info("[harness] No images to analyse - skipping vision stage.")

        # ----------------------------------------------------------------
        # Stage 3: Embed transcript into VectorStore
        # ----------------------------------------------------------------
        log.info("[harness] Stage 3/6: Embedding context...")
        job.status = JobStatus.EMBEDDING
        await _update_job_status(job_id, JobStatus.EMBEDDING, session=session)

        if ctx.segments:
            try:
                from .services.embedder import embed_segments
                from .services.vector_store import VectorStore
                ctx.chunks = await embed_segments(job_id=job_id, segments=ctx.segments)
                ctx.vector_store = VectorStore(job_id=job_id)
                ctx.vector_store.add(ctx.chunks)
                log.info(
                    "[harness] Embedded %d chunks into VectorStore (%d indexed)",
                    len(ctx.chunks), ctx.vector_store.count,
                )
            except Exception as exc:
                log.warning(
                    "[harness] Embedding stage failed (continuing without vectors): %s", exc
                )
        else:
            log.info("[harness] No segments to embed - skipping embedding stage.")

        # ----------------------------------------------------------------
        # Stage 4: Claim extraction
        # Merge visible_claims from image analyses into the claim list
        # ----------------------------------------------------------------
        log.info("[harness] Stage 4/6: Extracting claims...")
        job.status = JobStatus.EXTRACTING
        await _update_job_status(job_id, JobStatus.EXTRACTING, session=session)

        ctx.claims = await extract_claims(
            job_id=job_id,
            segments=ctx.segments,
            user_context=ctx.user_context,
            user_question=ctx.user_question,
            focus_areas=ctx.focus_areas,
        )

        # Promote visible_claims from each ImageAnalysis as Claim objects
        for img_analysis in ctx.images:
            for visible_text in img_analysis.visible_claims:
                if visible_text.strip():
                    ctx.claims.append(Claim(
                        job_id=job_id,
                        image_id=img_analysis.id,
                        text=visible_text.strip(),
                        is_checkable=True,
                        confidence=0.80,
                        context=(
                            f"Visible text in video frame at {img_analysis.frame_sec}s"
                            if img_analysis.frame_sec is not None
                            else "Visible text in image"
                        ),
                    ))

        log.info(
            "[harness] Extracted %d claims (%d from images)",
            len(ctx.claims),
            sum(1 for c in ctx.claims if c.image_id is not None),
        )

        # ----------------------------------------------------------------
        # Stage 5: Deep research
        # DeepResearchAgent takes handoff from ingest team and performs
        # multi-hop iterative research with semantic context retrieval
        # ----------------------------------------------------------------
        log.info("[harness] Stage 5/6: Deep research...")
        job.status = JobStatus.RESEARCHING
        await _update_job_status(job_id, JobStatus.RESEARCHING, session=session)

        ctx.research_results = await deep_research(
            claims=ctx.claims,
            context=ctx,
        )
        # Flatten all research evidence into ctx.evidence
        for rr in ctx.research_results:
            ctx.evidence.extend(rr.evidence)
        log.info(
            "[harness] Deep research complete: %d results, %d evidence items",
            len(ctx.research_results), len(ctx.evidence),
        )

        # ----------------------------------------------------------------
        # Stage 6: Verdict drafting
        # ----------------------------------------------------------------
        log.info("[harness] Stage 6/6: Drafting verdicts...")
        job.status = JobStatus.VERDICTING
        await _update_job_status(job_id, JobStatus.VERDICTING, session=session)

        ctx.verdicts = await draft_verdicts(
            claims=ctx.claims,
            evidence=ctx.evidence,
        )
        log.info("[harness] Drafted %d verdicts", len(ctx.verdicts))

        job.status = (
            JobStatus.REVIEW
            if any(v.requires_human_review for v in ctx.verdicts)
            or any(
                i.manipulation_risk.value in ("high", "medium")
                for i in ctx.images
            )
            else JobStatus.DONE
        )
        await _update_job_status(job_id, job.status, session=session)
        log.info("[harness] Pipeline complete. Status: %s", job.status)

    except Exception as exc:
        log.error(
            "[harness] Pipeline FAILED for job %s: %s", job_id, exc, exc_info=True
        )
        job.status = JobStatus.FAILED
        job.error  = str(exc)
        await _update_job_status(
            job_id, JobStatus.FAILED, error=str(exc), session=session
        )
        raise

    return PipelineResult(
        job=job,
        segments=ctx.segments,
        images=ctx.images,
        chunks=ctx.chunks,
        claims=ctx.claims,
        research=ctx.research_results,
        evidence=ctx.evidence,
        verdicts=ctx.verdicts,
    )
