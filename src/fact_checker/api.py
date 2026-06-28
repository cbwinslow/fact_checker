"""FastAPI layer for fact_checker.

Endpoints:
    POST /submit              - Submit URL or file path for async fact-checking
    GET  /jobs/{job_id}       - Get job result by ID from database
    GET  /jobs                - List recent jobs (paginated)
    DELETE /jobs/{job_id}     - Delete a job and its data
    POST /jobs/{job_id}/retry - Re-queue a failed job
    GET  /health              - Health check (unauthenticated)
    GET  /metrics             - Pipeline metrics summary (authenticated)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import require_api_key
from .config import settings
from .db import AsyncSessionLocal, get_session, init_db, save_pipeline_result, get_job_row
from .harness import run_pipeline
from .models import JobStatus, Citation, EvidenceItem
from .services.search_providers import get_registry, SearchResult, enrich_search_results_with_quotes

log = logging.getLogger(__name__)

app = FastAPI(
    title="Fact Checker API",
    description="AI-powered multi-modal fact-checking pipeline",
    version="0.2.0",
)

# ---------------------------------------------------------------------------
# CORS - configurable via settings.cors_origins
# ---------------------------------------------------------------------------
_cors_origins = getattr(settings, "cors_origins", "*")
if isinstance(_cors_origins, str):
    _cors_origins = [o.strip() for o in _cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    await init_db()
    log.info("[api] Database initialised on startup.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class SubmitRequest(BaseModel):
    url: Optional[str] = None
    local_path: Optional[str] = None
    webhook_url: Optional[str] = None  # POST result here when done
    priority: int = 0                  # higher = processed first (future use)
    # User-provided context
    user_context: Optional[str] = None          # Background info, hypothesis
    user_question: Optional[str] = None         # Specific question to answer
    focus_areas: List[str] = Field(default_factory=list)  # Topics to prioritize


class SubmitResponse(BaseModel):
    job_id: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    url: Optional[str]
    ingest_source: Optional[str]
    error: Optional[str]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class JobListResponse(BaseModel):
    jobs: List[JobStatusResponse]
    total: int
    page: int
    page_size: int


class MetricsResponse(BaseModel):
    total_jobs: int
    done: int
    failed: int
    in_progress: int
    pending: int
    review: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _persist_failure(job_id: UUID, error_msg: str) -> None:
    """Open a *fresh* session to write FAILED status after a rollback."""
    try:
        from sqlalchemy import update
        from .db import VideoJobRow
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(VideoJobRow)
                .where(VideoJobRow.id == job_id)
                .values(
                    status=JobStatus.FAILED.value,
                    error=error_msg,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
    except Exception as exc:
        log.error("[api] Could not persist FAILED status for job %s: %s", job_id, exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Unauthenticated health check - used by load balancers and CI."""
    return {"status": "ok", "version": "0.2.0"}


@app.get("/metrics", response_model=MetricsResponse)
async def metrics(
    session: AsyncSession = Depends(get_session),
    _auth: None = Depends(require_api_key),
) -> MetricsResponse:
    """Return aggregate pipeline metrics."""
    from sqlalchemy import func, select
    from .db import VideoJobRow

    result = await session.execute(
        select(VideoJobRow.status, func.count(VideoJobRow.id))
        .group_by(VideoJobRow.status)
    )
    counts = dict(result.all())
    in_progress_statuses = {
        JobStatus.INGESTING, JobStatus.TRANSCRIBING, JobStatus.ANALYZING,
        JobStatus.EMBEDDING, JobStatus.EXTRACTING, JobStatus.RESEARCHING,
        JobStatus.RETRIEVING, JobStatus.VERDICTING,
    }
    return MetricsResponse(
        total_jobs=sum(counts.values()),
        done=counts.get(JobStatus.DONE.value, 0),
        failed=counts.get(JobStatus.FAILED.value, 0),
        in_progress=sum(counts.get(s.value, 0) for s in in_progress_statuses),
        pending=counts.get(JobStatus.PENDING.value, 0),
        review=counts.get(JobStatus.REVIEW.value, 0),
    )


@app.post("/submit", response_model=SubmitResponse, status_code=202)
async def submit(
    request: SubmitRequest,
    background_tasks: BackgroundTasks,
    _auth: None = Depends(require_api_key),
) -> SubmitResponse:
    """Submit a video URL or local path for async fact-checking."""
    if not request.url and not request.local_path:
        raise HTTPException(status_code=422, detail="Provide url or local_path")

    job_id = uuid4()
    local_path = Path(request.local_path) if request.local_path else None

    async def _run_and_persist() -> None:
        """Background task: run pipeline with live DB status updates."""
        async with AsyncSessionLocal() as session:
            # --- Pre-create the job row so status polling works immediately ---
            from .db import VideoJobRow
            session.add(VideoJobRow(
                id=job_id,
                url=request.url,
                local_path=str(local_path) if local_path else None,
                status=JobStatus.PENDING.value,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ))
            await session.flush()
            try:
                result = await run_pipeline(
                    url=request.url,
                    local_path=local_path,
                    job_id=job_id,
                    session=session,
                    user_context=request.user_context,
                    user_question=request.user_question,
                    focus_areas=request.focus_areas,
                )
                await save_pipeline_result(session, result)
                await session.commit()
                log.info(
                    "[api] Job %s completed with status %s",
                    job_id, result.job.status,
                )
                # Fire webhook if requested
                if request.webhook_url:
                    from .services.webhook_notifier import notify_webhook
                    await notify_webhook(
                        webhook_url=request.webhook_url,
                        job_id=job_id,
                        status=result.job.status.value,
                        verdict_count=len(result.verdicts),
                    )
            except Exception as exc:
                await session.rollback()
                log.error("[api] Job %s failed: %s", job_id, exc)
                # Write FAILED status via a FRESH session (previous one rolled back)
                await _persist_failure(job_id, str(exc))

    background_tasks.add_task(_run_and_persist)
    return SubmitResponse(
        job_id=str(job_id),
        message="Job submitted. Poll /jobs/{job_id} for status.",
    )


@app.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_session),
    _auth: None = Depends(require_api_key),
) -> JobListResponse:
    """List jobs with optional status filter, newest first."""
    from sqlalchemy import select, func
    from .db import VideoJobRow

    q = select(VideoJobRow)
    if status:
        q = q.where(VideoJobRow.status == status)
    q = q.order_by(VideoJobRow.created_at.desc())

    count_q = select(func.count()).select_from(q.subquery())
    total = (await session.execute(count_q)).scalar_one()

    q = q.offset((page - 1) * page_size).limit(page_size)
    rows = (await session.execute(q)).scalars().all()

    return JobListResponse(
        jobs=[
            JobStatusResponse(
                job_id=str(r.id),
                status=r.status,
                url=r.url,
                ingest_source=r.ingest_source,
                error=r.error,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in rows
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
    _auth: None = Depends(require_api_key),
) -> JobStatusResponse:
    """Fetch the current status of a fact-checking job."""
    row = await get_job_row(session, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(
        job_id=str(row.id),
        status=row.status,
        url=row.url,
        ingest_source=row.ingest_source,
        error=row.error,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@app.delete("/jobs/{job_id}", status_code=204)
async def delete_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
    _auth: None = Depends(require_api_key),
) -> None:
    """Delete a job and all its associated data (cascades to claims, evidence, verdicts)."""
    row = await get_job_row(session, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    await session.delete(row)
    await session.commit()
    log.info("[api] Deleted job %s", job_id)


@app.post("/jobs/{job_id}/retry", response_model=SubmitResponse, status_code=202)
async def retry_job(
    job_id: UUID,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    _auth: None = Depends(require_api_key),
) -> SubmitResponse:
    """Re-queue a FAILED job. Resets status to PENDING and reruns the pipeline."""
    from sqlalchemy import update
    from .db import VideoJobRow

    row = await get_job_row(session, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if row.status not in (JobStatus.FAILED.value, JobStatus.DONE.value):
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is in status '{row.status}' and cannot be retried",
        )

    # Reset status
    await session.execute(
        update(VideoJobRow)
        .where(VideoJobRow.id == job_id)
        .values(status=JobStatus.PENDING.value, error=None, updated_at=datetime.now(timezone.utc))
    )
    await session.commit()

    _url = row.url
    _local_path = Path(row.local_path) if row.local_path else None

    async def _rerun() -> None:
        async with AsyncSessionLocal() as s:
            try:
                result = await run_pipeline(
                    url=_url,
                    local_path=_local_path,
                    job_id=job_id,
                    session=s,
                )
                await save_pipeline_result(s, result)
                await s.commit()
            except Exception as exc:
                await s.rollback()
                await _persist_failure(job_id, str(exc))

    background_tasks.add_task(_rerun)
    return SubmitResponse(
        job_id=str(job_id),
        message="Job re-queued. Poll /jobs/{job_id} for status.",
    )


# ---------------------------------------------------------------------------
# New endpoints: Citations, Evidence, Streaming
# ---------------------------------------------------------------------------


@app.get("/jobs/{job_id}/citations")
async def get_citations(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
    _auth: None = Depends(require_api_key),
) -> List[dict]:
    """Get all citations for a completed job's verdicts."""
    from .db import VerdictRow, EvidenceRow
    from sqlalchemy import select
    
    # Get verdicts for this job
    verdict_result = await session.execute(
        select(VerdictRow).where(VerdictRow.job_id == job_id)
    )
    verdicts = verdict_result.scalars().all()
    
    if not verdicts:
        raise HTTPException(status_code=404, detail=f"No verdicts found for job {job_id}")
    
    # Get evidence for each verdict
    citations = []
    for verdict in verdicts:
        # Parse the JSON explanation for citations
        # In a real implementation, citations would be stored separately
        # For now, we return the evidence items linked to this verdict
        evidence_result = await session.execute(
            select(EvidenceRow).where(EvidenceRow.claim_id == verdict.claim_id)
        )
        evidence_items = evidence_result.scalars().all()
        
        for ev in evidence_items:
            citations.append({
                "evidence_id": str(ev.id),
                "claim_id": str(verdict.claim_id),
                "verdict": verdict.label,
                "source_url": ev.source_url,
                "title": ev.title,
                "snippet": ev.snippet,
                "relevance_score": ev.relevance_score,
            })
    
    return citations


@app.get("/jobs/{job_id}/evidence")
async def get_evidence(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
    _auth: None = Depends(require_api_key),
    claim_id: Optional[UUID] = Query(default=None, description="Filter by claim ID"),
) -> List[dict]:
    """Get all evidence items for a job, optionally filtered by claim."""
    from .db import EvidenceRow, ClaimRow
    from sqlalchemy import select
    
    # Build query
    query = select(EvidenceRow).join(ClaimRow).where(ClaimRow.job_id == job_id)
    if claim_id:
        query = query.where(EvidenceRow.claim_id == claim_id)
    
    result = await session.execute(query)
    evidence_items = result.scalars().all()
    
    return [
        {
            "evidence_id": str(ev.id),
            "claim_id": str(ev.claim_id),
            "source_url": ev.source_url,
            "title": ev.title,
            "snippet": ev.snippet,
            "relevance_score": ev.relevance_score,
            "is_factcheck_source": False,  # Would need to add this column
        }
        for ev in evidence_items
    ]


@app.get("/jobs/{job_id}/stream")
async def stream_job_progress(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
    _auth: None = Depends(require_api_key),
):
    """Stream real-time pipeline progress as Server-Sent Events (SSE)."""
    from sse_starlette.sse import EventSourceResponse
    from .db import VideoJobRow
    from sqlalchemy import select
    import asyncio
    
    async def event_generator():
        # Check if job exists
        result = await session.execute(
            select(VideoJobRow).where(VideoJobRow.id == job_id)
        )
        job = result.scalar_one_or_none()
        if not job:
            yield {"event": "error", "data": f"Job {job_id} not found"}
            return
        
        # Send initial status
        yield {"event": "status", "data": job.status}
        
        # Poll for updates
        last_status = job.status
        while job.status in ["pending", "ingesting", "transcribing", "analyzing", 
                            "embedding", "extracting", "researching", "retrieving", "verdicting"]:
            await asyncio.sleep(2)
            await session.refresh(job)
            if job.status != last_status:
                yield {"event": "status", "data": job.status}
                last_status = job.status
                if job.status in ["done", "failed", "review"]:
                    break
        
        # Final status
        yield {"event": "status", "data": job.status}
        if job.status == "done":
            yield {"event": "complete", "data": {"verdict_count": 0}}  # Would fetch actual count
    
    return EventSourceResponse(event_generator())


@app.post("/search")
async def search_web(
    query: str,
    max_results: int = 10,
    source_types: Optional[List[str]] = None,
    _auth: None = Depends(require_api_key),
) -> List[dict]:
    """Direct web search using free providers."""
    registry = get_registry()
    results = await registry.search_all(
        query,
        max_results_per_provider=max_results // 5,
        source_types=source_types,
    )
    
    return [
        {
            "url": r.url,
            "title": r.title,
            "snippet": r.snippet,
            "domain": r.domain,
            "score": r.score,
            "provider": r.provider,
            "source_type": r.source_type,
        }
        for r in results
    ]
