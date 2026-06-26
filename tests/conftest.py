"""Shared pytest fixtures for fact_checker tests."""
from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from httpx import AsyncClient, ASGITransport

from fact_checker.api import app
from fact_checker.models import (
    Claim, EvidenceItem, JobStatus, PipelineResult,
    TranscriptSegment, Verdict, VerdictResult, VideoJob,
)


# ---------------------------------------------------------------------------
# Shared domain object fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def job_id():
    return uuid4()


@pytest.fixture
def sample_video_job(job_id):
    return VideoJob(id=job_id, url="https://www.youtube.com/watch?v=test123")


@pytest.fixture
def sample_segments(job_id):
    return [
        TranscriptSegment(
            job_id=job_id,
            start_sec=0.0,
            end_sec=10.0,
            text="The Earth is approximately 4.5 billion years old.",
        ),
        TranscriptSegment(
            job_id=job_id,
            start_sec=10.0,
            end_sec=20.0,
            text="Water covers about 71 percent of the Earth surface.",
        ),
    ]


@pytest.fixture
def sample_claims(job_id):
    claim1_id = uuid4()
    claim2_id = uuid4()
    return [
        Claim(
            id=claim1_id,
            job_id=job_id,
            text="The Earth is approximately 4.5 billion years old.",
            is_checkable=True,
            confidence=0.95,
        ),
        Claim(
            id=claim2_id,
            job_id=job_id,
            text="Water covers about 71 percent of the Earth surface.",
            is_checkable=True,
            confidence=0.90,
        ),
    ]


@pytest.fixture
def sample_evidence(sample_claims):
    return [
        EvidenceItem(
            claim_id=sample_claims[0].id,
            source_url="https://example.com/earth-age",
            title="Earth Age Study",
            snippet="The Earth is 4.54 billion years old.",
            relevance_score=0.95,
            is_factcheck_source=False,
        ),
    ]


@pytest.fixture
def sample_verdicts(sample_claims):
    return [
        VerdictResult(
            claim_id=sample_claims[0].id,
            verdict=Verdict.SUPPORTED,
            explanation="Multiple sources confirm Earth age is ~4.5 billion years.",
            confidence=0.92,
            requires_human_review=False,
        ),
        VerdictResult(
            claim_id=sample_claims[1].id,
            verdict=Verdict.SUPPORTED,
            explanation="USGS confirms water covers 71% of Earth surface.",
            confidence=0.97,
            requires_human_review=False,
        ),
    ]


@pytest.fixture
def sample_pipeline_result(sample_video_job, sample_segments, sample_claims, sample_evidence, sample_verdicts):
    sample_video_job.status = JobStatus.DONE
    return PipelineResult(
        job=sample_video_job,
        segments=sample_segments,
        claims=sample_claims,
        evidence=sample_evidence,
        verdicts=sample_verdicts,
    )


# ---------------------------------------------------------------------------
# Async HTTP client fixture (FastAPI test client)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def async_client():
    """Async HTTPX client for FastAPI endpoint tests."""
    # Override DB init to no-op for tests
    with patch("fact_checker.api.init_db", new_callable=AsyncMock):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client
