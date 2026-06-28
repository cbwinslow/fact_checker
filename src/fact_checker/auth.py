"""API key authentication dependency for fact_checker.

Usage
-----
When API_KEY is set in the environment, all endpoints that depend on
`require_api_key` will enforce bearer token authentication.

If API_KEY is not set (empty string / unset), authentication is disabled
so local development works without any setup.

Example .env
------------
    API_KEY=super-secret-key-here

Clients pass the key via the Authorization header::

    Authorization: Bearer <API_KEY>
"""
from __future__ import annotations

import os

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Read once at import time; reload the app to pick up changes.
_API_KEY: str = os.environ.get("API_KEY", "").strip()

# HTTPBearer scheme - returns 403 automatically if the Authorization header
# is entirely missing, so we use auto_error=False and handle it ourselves.
_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """FastAPI dependency that enforces API key authentication.

    - If ``API_KEY`` env var is empty/unset, the check is **bypassed** so
      local development does not need a key.
    - Otherwise the ``Authorization: Bearer <key>`` header must match.
    """
    if not _API_KEY:
        # Auth disabled - allow all requests through.
        return

    if credentials is None or credentials.credentials != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


__all__ = ["require_api_key"]
