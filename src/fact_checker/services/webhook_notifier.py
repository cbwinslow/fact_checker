"""services/webhook_notifier.py - Fire-and-forget webhook notifications.

POSTs a JSON summary payload to a caller-supplied URL when a pipeline job
completes, fails, or requires human review.  The caller registers a
``webhook_url`` on the /submit endpoint; this module handles delivery.

Features:
- Automatic retries with exponential back-off (up to 3 attempts).
- HMAC-SHA256 request signing via ``get_settings().webhook_secret``.
- Non-blocking: failures are logged but never re-raise.

Dependencies::
    pip install httpx
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional
from uuid import UUID

import httpx

from ..config import get_settings

log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds


async def notify_webhook(
    webhook_url: str,
    job_id: UUID,
    status: str,
    verdict_count: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """POST a JSON notification payload to ``webhook_url``.

    The request body is a JSON object::

        {
            "job_id": "<uuid>",
            "status": "done" | "failed" | "review",
            "verdict_count": <int>,
            "timestamp": <unix epoch float>,
            ...extra fields...
        }

    If ``get_settings().webhook_secret`` is set, an HMAC-SHA256 signature of the
    raw JSON body is attached in the ``X-Fact-Checker-Signature`` header so
    receivers can verify authenticity.

    Args:
        webhook_url: Target URL to POST to.
        job_id: UUID of the completed job.
        status: Final job status string.
        verdict_count: Number of verdicts produced.
        extra: Optional additional key/value pairs to merge into the payload.
    """
    payload: Dict[str, Any] = {
        "job_id": str(job_id),
        "status": status,
        "verdict_count": verdict_count,
        "timestamp": time.time(),
    }
    if extra:
        payload.update(extra)

    body = json.dumps(payload, default=str).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    # Sign the payload if a secret is configured
    secret = getattr(settings, "webhook_secret", "").strip()
    if secret:
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        headers["X-Fact-Checker-Signature"] = f"sha256={sig}"

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(webhook_url, content=body, headers=headers)
                resp.raise_for_status()
            log.info(
                "[webhook] Notified %s for job %s (status=%s, attempt=%d)",
                webhook_url, job_id, status, attempt,
            )
            return
        except Exception as exc:
            log.warning(
                "[webhook] Delivery attempt %d/%d failed for job %s: %s",
                attempt, _MAX_RETRIES, job_id, exc,
            )
            if attempt < _MAX_RETRIES:
                import asyncio
                await asyncio.sleep(_BACKOFF_BASE ** attempt)

    log.error(
        "[webhook] All %d delivery attempts failed for job %s -> %s",
        _MAX_RETRIES, job_id, webhook_url,
    )
