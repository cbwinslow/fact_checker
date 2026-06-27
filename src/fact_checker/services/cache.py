"""services/cache.py - Redis-backed result cache for the fact-checker pipeline.

Caches serialised PipelineResult objects so repeated requests for the same URL
or content hash return immediately without re-running the full pipeline.

Backend resolution order:
  1. Redis (via ``redis.asyncio``) if ``settings.redis_url`` is set.
  2. In-process LRU dict (``_MemoryCache``) for single-process deployments or
     when Redis is unavailable.  Limited to ``settings.cache_max_size`` entries.

Dependencies (optional)::
    pip install redis
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

from ..config import settings

log = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 60 * 60 * 24  # 24 h
_DEFAULT_MAX_SIZE = 512

# ---------------------------------------------------------------------------
# In-memory fallback cache (LRU via OrderedDict)
# ---------------------------------------------------------------------------

class _MemoryCache:
    """Simple thread-safe-ish LRU dict cache."""

    def __init__(self, max_size: int = _DEFAULT_MAX_SIZE) -> None:
        from collections import OrderedDict
        self._store: OrderedDict[str, str] = OrderedDict()
        self._max_size = max_size

    async def get(self, key: str) -> Optional[str]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    async def set(self, key: str, value: str, ttl: int = _DEFAULT_TTL_SECONDS) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)  # evict oldest

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def flush(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Redis cache
# ---------------------------------------------------------------------------

class _RedisCache:
    """Redis-backed async cache."""

    def __init__(self, redis_url: str) -> None:
        import redis.asyncio as aioredis
        self._client = aioredis.from_url(redis_url, decode_responses=True)

    async def get(self, key: str) -> Optional[str]:
        try:
            return await self._client.get(key)
        except Exception as exc:
            log.warning("[cache] Redis GET failed: %s", exc)
            return None

    async def set(self, key: str, value: str, ttl: int = _DEFAULT_TTL_SECONDS) -> None:
        try:
            await self._client.set(key, value, ex=ttl)
        except Exception as exc:
            log.warning("[cache] Redis SET failed: %s", exc)

    async def delete(self, key: str) -> None:
        try:
            await self._client.delete(key)
        except Exception as exc:
            log.warning("[cache] Redis DELETE failed: %s", exc)

    async def flush(self) -> None:
        try:
            await self._client.flushdb()
        except Exception as exc:
            log.warning("[cache] Redis FLUSH failed: %s", exc)


# ---------------------------------------------------------------------------
# Singleton backend
# ---------------------------------------------------------------------------

def _build_backend():
    redis_url = getattr(settings, "redis_url", "").strip()
    if redis_url:
        try:
            backend = _RedisCache(redis_url)
            log.info("[cache] Using Redis backend: %s", redis_url)
            return backend
        except ImportError:
            log.warning("[cache] redis package not installed - falling back to memory cache")
    max_size = getattr(settings, "cache_max_size", _DEFAULT_MAX_SIZE)
    log.info("[cache] Using in-memory LRU cache (max_size=%d)", max_size)
    return _MemoryCache(max_size=max_size)


_backend = None


def _get_backend():
    global _backend
    if _backend is None:
        _backend = _build_backend()
    return _backend


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_cache_key(prefix: str, *parts: Any) -> str:
    """Create a stable cache key from a prefix and arbitrary parts.

    The parts are joined and SHA-256-hashed to keep keys short and
    safe for any cache backend.

    Args:
        prefix: Human-readable namespace prefix (e.g. ``"pipeline_result"``)
        *parts: Key components (URLs, hashes, IDs, etc.)

    Returns:
        Cache key string of the form ``"<prefix>:<sha256[:16]>"``.
    """
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"


async def cache_get(key: str) -> Optional[Any]:
    """Retrieve a cached value by key.

    Returns the deserialised Python object, or ``None`` on miss.

    Args:
        key: Cache key as returned by :func:`make_cache_key`.

    Returns:
        Deserialised Python object or ``None``.
    """
    raw = await _get_backend().get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception as exc:
        log.warning("[cache] Deserialisation error for key %s: %s", key, exc)
        return None


async def cache_set(
    key: str,
    value: Any,
    ttl: int = _DEFAULT_TTL_SECONDS,
) -> None:
    """Store a value in the cache.

    The value is JSON-serialised before storage.

    Args:
        key: Cache key.
        value: Python object to cache (must be JSON-serialisable).
        ttl: Time-to-live in seconds (default 24 h).
    """
    try:
        raw = json.dumps(value, default=str)
    except Exception as exc:
        log.warning("[cache] Serialisation error for key %s: %s", key, exc)
        return
    await _get_backend().set(key, raw, ttl=ttl)


async def cache_delete(key: str) -> None:
    """Remove a single entry from the cache.

    Args:
        key: Cache key to remove.
    """
    await _get_backend().delete(key)


async def cache_flush() -> None:
    """Clear the entire cache.  Use with care in production."""
    await _get_backend().flush()
    log.warning("[cache] Cache flushed.")
