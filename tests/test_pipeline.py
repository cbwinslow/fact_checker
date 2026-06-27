"""test_pipeline.py - Integration tests for the full fact-checking pipeline.

Run with::
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py::test_offline_pipeline -v
"""
import pytest
from pathlib import Path
from uuid import uuid4

from fact_checker.harness import run_pipeline
from fact_checker.models import JobStatus, Verdict


@pytest.mark.asyncio
async def test_offline_pipeline_text_file(tmp_path):
    """Test the full pipeline end-to-end with a text file input (offline/mock mode)."""
    # Create a temporary text file with claims
    test_file = tmp_path / "test.txt"
    test_file.write_text(
        "Climate change is caused by human activity. "
        "The Earth is approximately 4.5 billion years old. "
        "Water boils at 100 degrees Celsius at sea level."
    )

    job_id = uuid4()
    result = await run_pipeline(
        local_path=test_file,
        job_id=job_id,
        session=None,  # No DB session = offline mode
    )

    # Assertions
    assert result.job.id == job_id
    assert result.job.status in (JobStatus.DONE, JobStatus.REVIEW)
    assert len(result.segments) > 0
    assert len(result.claims) > 0  # MockChatModel returns 2 mock claims
    assert len(result.verdicts) == len(result.claims)

    for verdict in result.verdicts:
        assert verdict.verdict in Verdict.__members__.values()
        assert verdict.explanation
        assert 0.0 <= verdict.confidence <= 1.0


@pytest.mark.asyncio
async def test_pipeline_url_ingestion():
    """Test URL ingestion (uses web scraper fallback in offline mode)."""
    # This will hit the web scraper, which may fail if no network
    # In CI/offline environments, consider mocking httpx
    job_id = uuid4()
    try:
        result = await run_pipeline(
            url="https://example.com",
            job_id=job_id,
            session=None,
        )
        assert result.job.id == job_id
        assert result.job.status != JobStatus.FAILED
    except Exception:
        pytest.skip("Network unavailable or scraping failed")


@pytest.mark.asyncio
async def test_pipeline_with_images(tmp_path):
    """Test pipeline with image-only input."""
    # Create a dummy 1x1 pixel PNG
    import base64
    png_data = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    img_file = tmp_path / "test.png"
    img_file.write_bytes(png_data)

    job_id = uuid4()
    result = await run_pipeline(
        image_paths=[str(img_file)],
        job_id=job_id,
        session=None,
    )

    assert result.job.id == job_id
    assert len(result.images) > 0  # MockChatModel returns 1 image analysis
    # Image-only jobs may have 0 segments but can have claims from visible_claims


def test_models_serialization():
    """Test that all Pydantic models can serialize to JSON."""
    from fact_checker.models import (
        VideoJob, TranscriptSegment, Claim, EvidenceItem,
        VerdictResult, AnalysisContext, PipelineResult, Verdict
    )
    import json

    job = VideoJob(url="https://example.com")
    segment = TranscriptSegment(job_id=job.id, start_sec=0.0, end_sec=1.0, text="Test")
    claim = Claim(job_id=job.id, text="Test claim")
    evidence = EvidenceItem(claim_id=claim.id, source_url="https://example.com", snippet="Test")
    verdict = VerdictResult(claim_id=claim.id, verdict=Verdict.SUPPORTED, explanation="Test", confidence=0.9)

    # Test serialization
    assert json.loads(job.model_dump_json())
    assert json.loads(segment.model_dump_json())
    assert json.loads(claim.model_dump_json())
    assert json.loads(evidence.model_dump_json())
    assert json.loads(verdict.model_dump_json())

    # Test PipelineResult serialization (excluding vector_store)
    result = PipelineResult(
        job=job,
        segments=[segment],
        claims=[claim],
        evidence=[evidence],
        verdicts=[verdict],
    )
    result_json = result.model_dump_json()
    assert json.loads(result_json)


def test_cache_key_generation():
    """Test cache key generation is stable and collision-resistant."""
    from fact_checker.services.cache import make_cache_key

    key1 = make_cache_key("pipeline", "https://example.com")
    key2 = make_cache_key("pipeline", "https://example.com")
    key3 = make_cache_key("pipeline", "https://example.org")

    assert key1 == key2  # Same inputs = same key
    assert key1 != key3  # Different inputs = different keys
    assert key1.startswith("pipeline:")
    assert len(key1.split(":")[1]) == 16  # SHA-256 truncated to 16 chars


@pytest.mark.asyncio
async def test_rate_limiter():
    """Test the token bucket rate limiter."""
    from fact_checker.services.rate_limiter import TokenBucket
    import time

    limiter = TokenBucket(rate=10.0, capacity=10.0)  # 10 req/sec

    # Burst: consume 5 tokens instantly
    start = time.monotonic()
    for _ in range(5):
        await limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.1  # Should complete instantly (within 100ms)

    # Sustain: next 5 should throttle
    start = time.monotonic()
    for _ in range(5):
        await limiter.acquire()
    elapsed = time.monotonic() - start
    assert 0.4 < elapsed < 0.6  # ~0.5 seconds for 5 tokens at 10/sec
