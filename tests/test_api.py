"""Tests for fact_checker FastAPI endpoints."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_JOB_ID = str(uuid.uuid4())


def _make_job_row(status: str = "pending") -> MagicMock:
    """Return a minimal mock VideoJobRow."""
    row = MagicMock()
    row.job_id = FAKE_JOB_ID
    row.status = status
    row.url = "https://youtube.com/watch?v=test"
    row.ingest_source = "youtube"
    row.error_message = None
    row.claims = []
    row.verdicts = []
    return row


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def app():
    """Import the FastAPI app with DB init patched out."""
    with patch("fact_checker.db.init_db", new_callable=AsyncMock):
        from fact_checker.api import app as _app
        return _app


@pytest.fixture()
async def client(app):
    """Async test client for the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    """Tests for GET /health."""

    @pytest.mark.anyio
    async def test_health_returns_200(self, client):
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_health_body(self, client):
        response = await client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


# ---------------------------------------------------------------------------
# Submit endpoint
# ---------------------------------------------------------------------------

class TestSubmit:
    """Tests for POST /submit."""

    @pytest.mark.anyio
    async def test_submit_url_returns_202(self, client):
        with (
            patch("fact_checker.api.get_session"),
            patch("fact_checker.api.save_pipeline_result", new_callable=AsyncMock),
            patch("fact_checker.api.run_pipeline", new_callable=AsyncMock),
        ):
            response = await client.post(
                "/submit", json={"url": "https://youtube.com/watch?v=abc123"}
            )
        assert response.status_code == 202

    @pytest.mark.anyio
    async def test_submit_returns_job_id(self, client):
        with (
            patch("fact_checker.api.get_session"),
            patch("fact_checker.api.save_pipeline_result", new_callable=AsyncMock),
            patch("fact_checker.api.run_pipeline", new_callable=AsyncMock),
        ):
            response = await client.post(
                "/submit", json={"url": "https://youtube.com/watch?v=abc123"}
            )
        data = response.json()
        assert "job_id" in data
        # job_id should be a valid UUID
        uuid.UUID(data["job_id"])

    @pytest.mark.anyio
    async def test_submit_requires_url_or_path(self, client):
        """Submitting empty body should return 422 validation error."""
        response = await client.post("/submit", json={})
        assert response.status_code == 422

    @pytest.mark.anyio
    async def test_submit_local_path(self, client):
        with (
            patch("fact_checker.api.get_session"),
            patch("fact_checker.api.save_pipeline_result", new_callable=AsyncMock),
            patch("fact_checker.api.run_pipeline", new_callable=AsyncMock),
        ):
            response = await client.post(
                "/submit", json={"local_path": "/tmp/video.mp4"}
            )
        assert response.status_code == 202


# ---------------------------------------------------------------------------
# Get job endpoint
# ---------------------------------------------------------------------------

class TestGetJob:
    """Tests for GET /jobs/{job_id}."""

    @pytest.mark.anyio
    async def test_get_job_found(self, client):
        mock_row = _make_job_row(status="completed")
        with patch(
            "fact_checker.api.get_job_row",
            new_callable=AsyncMock,
            return_value=mock_row,
        ):
            response = await client.get(f"/jobs/{FAKE_JOB_ID}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == FAKE_JOB_ID
        assert data["status"] == "completed"

    @pytest.mark.anyio
    async def test_get_job_not_found_returns_404(self, client):
        with patch(
            "fact_checker.api.get_job_row",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.get(f"/jobs/{FAKE_JOB_ID}")
        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_get_job_invalid_uuid_returns_422(self, client):
        response = await client.get("/jobs/not-a-valid-uuid")
        assert response.status_code == 422

    @pytest.mark.anyio
    async def test_get_job_pending_status(self, client):
        mock_row = _make_job_row(status="pending")
        with patch(
            "fact_checker.api.get_job_row",
            new_callable=AsyncMock,
            return_value=mock_row,
        ):
            response = await client.get(f"/jobs/{FAKE_JOB_ID}")
        assert response.json()["status"] == "pending"

    @pytest.mark.anyio
    async def test_get_job_ingest_source_present(self, client):
        mock_row = _make_job_row()
        with patch(
            "fact_checker.api.get_job_row",
            new_callable=AsyncMock,
            return_value=mock_row,
        ):
            response = await client.get(f"/jobs/{FAKE_JOB_ID}")
        assert "ingest_source" in response.json()
