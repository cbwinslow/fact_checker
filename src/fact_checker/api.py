"""FastAPI layer for fact_checker.

Endpoints:
  POST /submit              - Submit URL or file path for async fact-checking
  GET  /jobs/{job_id}       - Get job result by ID from database
  GET  /health              - Health check (unauthenticated)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import require_api_key
from .db import AsyncSessionLocal, get_session, init_db, save_pipeline_result, get_job_row
from .harness import run_pipeline
from .models import JobStatus

log = logging.getLogger(__name__)

app = FastAPI(
    title="Fact Checker API",
    description="AI-powered video fact-checking pipeline",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


class SubmitResponse(BaseModel):
    job_id: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    url: Optional[str]
    ingest_source: Optional[str]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Unauthenticated health check - used by load balancers and CI."""
    return {"status": "ok", "version": "0.1.0"}


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
            from datetime import datetime
            session.add(VideoJobRow(
                id=job_id,
                url=request.url,
                local_path=str(local_path) if local_path else None,
                status=JobStatus.PENDING.value,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            ))
            await session.flush()

            try:
                # Pass session so harness can write intermediate status updates
                result = await run_pipeline(
                    url=request.url,
                    local_path=local_path,
                    job_id=job_id,
                    session=session,
                )
                await save_pipeline_result(session, result)
                await session.commit()
                log.info("[api] Job %s completed with status %s", job_id, result.job.status)
            except Exception as exc:
                await session.rollback()
                log.error("[api] Job %s failed: %s", job_id, exc)

    background_tasks.add_task(_run_and_persist)
    return SubmitResponse(
        job_id=str(job_id),
        message="Job submitted. Poll /jobs/{job_id} for status.",
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
    _auth: None = Depends(require_api_key),
) -> JobStatusResponse:
    """Fetch the current status of a fact-checking job from the database."""
    row = await get_job_row(session, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    # VideoJobRow uses .id and .error (not .job_id / .error_message)
    return JobStatusResponse(
        job_id=str(row.id),
        status=row.status,
        url=row.url,
        ingest_source=row.ingest_source,
        error=row.error,
    )
