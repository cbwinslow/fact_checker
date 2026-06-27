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
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import require_api_key
from .config import settings
from .db import AsyncSessionLocal, get_session, init_db, save_pipeline_result, get_job_row
from .harness import run_pipeline
from .models import JobStatus

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
    from .db import VideoJobRow
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
