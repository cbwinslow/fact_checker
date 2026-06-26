"""Shared pytest fixtures for fact_checker tests.

Uses anyio for async test support (pytest-anyio plugin).
All async tests should be decorated with @pytest.mark.anyio.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from httpx import AsyncClient, ASGITransport


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
