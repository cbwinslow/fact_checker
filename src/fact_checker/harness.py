"""Pipeline harness - orchestrates all agents in sequence.

Flow: ingest -> extract_claims -> retrieve_evidence -> draft_verdicts
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .models import JobStatus, PipelineResult, VideoJob
from .services.ingest import ingest
from .agents.claim_extractor import extract_claims
from .agents.evidence_agent import retrieve_evidence
from .agents.verdict_agent import draft_verdicts

log = logging.getLogger(__name__)


async def run_pipeline(
    url: Optional[str] = None,
    local_path: Optional[Path] = None,
    job_id=None,
) -> PipelineResult:
    """Run the full fact-checking pipeline and return a PipelineResult."""
    if job_id is None:
        job_id = uuid4()

    job = VideoJob(id=job_id, url=url, local_path=str(local_path) if local_path else None)

    try:
        # Stage 1: Ingest
        log.info("[harness] Stage 1/4: Ingesting media...")
        job.status = JobStatus.INGESTING
        segments, ingest_source = await ingest(job_id=job_id, url=url, local_path=local_path)
        job.ingest_source = ingest_source
        log.info("[harness] Ingested %d segments via %s", len(segments), ingest_source)

        # Stage 2: Claim extraction
        log.info("[harness] Stage 2/4: Extracting claims...")
        job.status = JobStatus.EXTRACTING
        claims = await extract_claims(job_id=job_id, segments=segments)
        log.info("[harness] Extracted %d claims", len(claims))

        # Stage 3: Evidence retrieval
        log.info("[harness] Stage 3/4: Retrieving evidence...")
        job.status = JobStatus.RETRIEVING
        evidence = await retrieve_evidence(claims=claims)
        log.info("[harness] Retrieved %d evidence items", len(evidence))

        # Stage 4: Verdict drafting
        log.info("[harness] Stage 4/4: Drafting verdicts...")
        job.status = JobStatus.VERDICTING
        verdicts = await draft_verdicts(claims=claims, evidence=evidence)
        log.info("[harness] Drafted %d verdicts", len(verdicts))

        job.status = JobStatus.REVIEW if any(v.requires_human_review for v in verdicts) else JobStatus.DONE
        log.info("[harness] Pipeline complete. Job status: %s", job.status)

        return PipelineResult(
            job=job,
            segments=segments,
            claims=claims,
            evidence=evidence,
            verdicts=verdicts,
        )

    except Exception as e:
        log.error("[harness] Pipeline failed: %s", e, exc_info=True)
        job.status = JobStatus.FAILED
        job.error = str(e)
        return PipelineResult(job=job)
