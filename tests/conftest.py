"""Shared pytest fixtures for fact_checker tests.

Uses anyio for async test support (pytest-anyio plugin).
All async tests should be decorated with @pytest.mark.asyncio.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from httpx import AsyncClient, ASGITransport

from fact_checker.models import (
    Claim,
    EvidenceItem,
    TranscriptSegment,
    VerdictResult,
    Verdict,
)


# ---------------------------------------------------------------------------
# anyio backend config - run all async tests with asyncio
# ---------------------------------------------------------------------------

@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


# ---------------------------------------------------------------------------
# Shared domain object fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def job_id():
    return uuid4()


@pytest.fixture
def sample_segments(job_id):
    """Two simple transcript segments for use in agent tests."""
    return [
        TranscriptSegment(
            job_id=job_id,
            start_sec=0.0,
            end_sec=10.0,
            text="The Earth is approximately 4.5 billion years old.",
            speaker=None,
        ),
        TranscriptSegment(
            job_id=job_id,
            start_sec=10.0,
            end_sec=20.0,
            text="Water covers about 71% of the Earth's surface.",
            speaker=None,
        ),
    ]


@pytest.fixture
def sample_claims(job_id):
    """Two Claim objects for use in verdict/evidence agent tests."""
    return [
        Claim(
            job_id=job_id,
            text="The Earth is approximately 4.5 billion years old.",
            is_checkable=True,
            confidence=0.95,
            context=None,
        ),
        Claim(
            job_id=job_id,
            text="Water covers about 71% of the Earth's surface.",
            is_checkable=True,
            confidence=0.90,
            context=None,
        ),
    ]


@pytest.fixture
def sample_evidence(sample_claims):
    """One EvidenceItem per sample claim."""
    return [
        EvidenceItem(
            claim_id=sample_claims[0].id,
            source_url="https://en.wikipedia.org/wiki/Earth",
            title="Earth - Wikipedia",
            snippet="Earth is estimated to be about 4.54 billion years old.",
            relevance_score=0.95,
            is_factcheck_source=False,
        ),
        EvidenceItem(
            claim_id=sample_claims[1].id,
            source_url="https://oceanservice.noaa.gov/facts/oceanpercent.html",
            title="How much of the ocean have we explored?",
            snippet="The ocean covers more than 70 percent of the Earth's surface.",
            relevance_score=0.90,
            is_factcheck_source=True,
        ),
    ]


@pytest.fixture
def app():
    """Import the FastAPI app with DB init patched out."""
    with patch("fact_checker.db.init_db", new_callable=AsyncMock):
        from fact_checker.api import app as _app
        yield _app


@pytest.fixture
async def client(app):
    """Async test client that works with anyio."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
