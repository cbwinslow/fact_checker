"""FastAPI layer for fact_checker.

Endpoints:
  POST /submit          - Submit URL or file path for fact-checking
  GET  /jobs/{job_id}  - Get job result by ID (in-memory cache for now)
  GET  /health         - Health check
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .harness import run_pipeline
from .models import PipelineResult

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

# In-memory result cache (swap for DB in production)
_results: dict[str, PipelineResult] = {}


class SubmitRequest(BaseModel):
    url: Optional[str] = None
    local_path: Optional[str] = None


class SubmitResponse(BaseModel):
    job_id: str
    message: str


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/submit", response_model=SubmitResponse)
async def submit(
    request: SubmitRequest,
    background_tasks: BackgroundTasks,
):
    """Submit a video URL or local path for async fact-checking."""
    if not request.url and not request.local_path:
        raise HTTPException(status_code=400, detail="Provide url or local_path")

    local_path = Path(request.local_path) if request.local_path else None

    async def _run():
        result = await run_pipeline(url=request.url, local_path=local_path)
        _results[str(result.job.id)] = result

    background_tasks.add_task(_run)

    # Return a placeholder job ID immediately
    from uuid import uuid4
    job_id = str(uuid4())
    return SubmitResponse(job_id=job_id, message="Job submitted. Results available at /jobs/{job_id} once complete.")


@app.post("/submit/sync", response_model=PipelineResult)
async def submit_sync(request: SubmitRequest):
    """Submit synchronously and wait for full result (use for short videos)."""
    if not request.url and not request.local_path:
        raise HTTPException(status_code=400, detail="Provide url or local_path")
    local_path = Path(request.local_path) if request.local_path else None
    result = await run_pipeline(url=request.url, local_path=local_path)
    _results[str(result.job.id)] = result
    return result


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Retrieve a completed job result."""
    result = _results.get(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found or still running")
    return result


@app.get("/jobs")
async def list_jobs():
    """List all completed job IDs."""
    return {
        "jobs": [
            {"job_id": jid, "status": r.job.status.value, "url": r.job.url}
            for jid, r in _results.items()
        ]
    }
