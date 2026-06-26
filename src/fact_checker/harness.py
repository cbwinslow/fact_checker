"""Pipeline harness - orchestrates all agents in sequence.

Flow: ingest -> extract_claims -> retrieve_evidence -> draft_verdicts

Supports optional live DB status updates between stages so the
GET /jobs/{job_id} polling endpoint reflects real progress.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .models import JobStatus, PipelineResult, VideoJob
from .services.ingest import ingest
from .agents.claim_extractor import extract_claims
from .agents.evidence_agent import retrieve_evidence
from .agents.verdict_agent import draft_verdicts

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


async def run_pipeline(
    url: Optional[str] = None,
    local_path: Optional[Path] = None,
    job_id=None,
    session=None,
) -> PipelineResult:
    """Run the full fact-checking pipeline and return a PipelineResult.

    Args:
        url:        YouTube / web URL to fact-check.
        local_path: Path to a local audio/video file.
        job_id:     Pre-generated UUID for the job (generated if None).
        session:    Optional AsyncSession. When provided, live status
                    updates are written to the DB between pipeline stages
                    so polling /jobs/{job_id} returns real-time progress.
    """
    if job_id is None:
        job_id = uuid4()

    job = VideoJob(id=job_id, url=url, local_path=str(local_path) if local_path else None)

    try:
        # Stage 1: Ingest
        log.info("[harness] Stage 1/4: Ingesting media...")
        job.status = JobStatus.INGESTING
        await _update_job_status(job_id, JobStatus.INGESTING, session=session)
        segments, ingest_source = await ingest(job_id=job_id, url=url, local_path=local_path)
        job.ingest_source = ingest_source
        log.info("[harness] Ingested %d segments via %s", len(segments), ingest_source)

        # Stage 2: Claim extraction
        log.info("[harness] Stage 2/4: Extracting claims...")
        job.status = JobStatus.EXTRACTING
        await _update_job_status(job_id, JobStatus.EXTRACTING, session=session)
        claims = await extract_claims(job_id=job_id, segments=segments)
        log.info("[harness] Extracted %d claims", len(claims))

        # Stage 3: Evidence retrieval
        log.info("[harness] Stage 3/4: Retrieving evidence...")
        job.status = JobStatus.RETRIEVING
        await _update_job_status(job_id, JobStatus.RETRIEVING, session=session)
        evidence = await retrieve_evidence(claims=claims)
        log.info("[harness] Retrieved %d evidence items", len(evidence))

        # Stage 4: Verdict drafting
        log.info("[harness] Stage 4/4: Drafting verdicts...")
        job.status = JobStatus.VERDICTING
        await _update_job_status(job_id, JobStatus.VERDICTING, session=session)
        verdicts = await draft_verdicts(claims=claims, evidence=evidence)
        log.info("[harness] Drafted %d verdicts", len(verdicts))

        job.status = JobStatus.REVIEW if any(v.requires_human_review for v in verdicts) else JobStatus.DONE
        await _update_job_status(job_id, job.status, session=session)
        log.info("[harness] Pipeline complete. Job status: %s", job.status)

    except Exception as exc:
        log.error("[harness] Pipeline failed for job %s: %s", job_id, exc, exc_info=True)
        job.status = JobStatus.FAILED
        job.error = str(exc)
        await _update_job_status(job_id, JobStatus.FAILED, error=str(exc), session=session)
        # Re-raise so api.py background task can log it
        raise

    return PipelineResult(
        job=job,
        segments=segments,
        claims=claims,
        evidence=evidence,
        verdicts=verdicts,
    )
