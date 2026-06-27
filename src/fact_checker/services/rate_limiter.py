"""services/rate_limiter.py - Token-bucket rate limiter for external API calls.

Prevents hammering of third-party APIs (Serper, Google Fact Check, OpenRouter,
Wikipedia) by enforcing per-service request budgets.

Two implementations are provided:
  ``TokenBucket``       - in-process async token bucket (single worker)
  ``RedisTokenBucket``  - Redis-backed token bucket (multi-process / multi-worker)

Usage example::

    limiter = get_limiter("serper")  # returns singleton for the service
    await limiter.acquire()          # blocks until a token is available
    # ... make API call ...

Rate limits are configured in ``settings`` via fields named
``ratelimit_<service>_rps`` (requests per second, float).  Defaults are
coded below in ``_DEFAULT_RPS``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

from ..config import settings

log = logging.getLogger(__name__)

# Default requests-per-second budgets for each external service
_DEFAULT_RPS: Dict[str, float] = {
    "serper": 2.0,          # Serper.dev web search
    "google_fc": 5.0,       # Google Fact Check Tools
    "openrouter": 3.0,      # OpenRouter LLM API
    "wikipedia": 10.0,      # Wikipedia REST API (generous public limit)
    "scraper": 2.0,         # General web scraping
    "default": 2.0,
}


class TokenBucket:
    """Async in-process token bucket rate limiter.

    Refills tokens continuously at ``rate`` tokens/second up to ``capacity``
    tokens (burst allowance = ``capacity``).  ``acquire()`` waits until a
    token is available without busy-looping.

    Args:
        rate: Sustained requests per second (refill rate).
        capacity: Maximum burst tokens (defaults to ``rate * 2``).
    """

    def __init__(self, rate: float, capacity: Optional[float] = None) -> None:
        self.rate = max(rate, 0.01)
        self.capacity = capacity if capacity is not None else rate * 2
        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until ``tokens`` are available, then consume them.

        Args:
            tokens: Number of tokens to consume (default 1).
        """
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait_for = (tokens - self._tokens) / self.rate
            await asyncio.sleep(wait_for)


class RedisTokenBucket:
    """Redis-backed distributed token bucket for multi-worker deployments.

    Uses a Lua script to atomically refill and consume tokens.
    Falls back to ``TokenBucket`` if Redis is unavailable.

    Args:
        service: Service name (used as Redis key prefix).
        rate: Sustained requests per second.
        capacity: Maximum burst tokens.
        redis_url: Redis connection URL.
    """

    _LUA_SCRIPT = """
    local key = KEYS[1]
    local rate = tonumber(ARGV[1])
    local capacity = tonumber(ARGV[2])
    local tokens_requested = tonumber(ARGV[3])
    local now = tonumber(ARGV[4])
    local data = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(data[1]) or capacity
    local last_refill = tonumber(data[2]) or now
    local elapsed = now - last_refill
    tokens = math.min(capacity, tokens + elapsed * rate)
    if tokens >= tokens_requested then
        tokens = tokens - tokens_requested
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, 3600)
        return 1
    else
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, 3600)
        return 0
    end
    """

    def __init__(
        self,
        service: str,
        rate: float,
        capacity: Optional[float] = None,
        redis_url: str = "",
    ) -> None:
        self.service = service
        self.rate = max(rate, 0.01)
        self.capacity = capacity if capacity is not None else rate * 2
        self._fallback = TokenBucket(rate, self.capacity)
        self._client = None
        if redis_url:
            try:
                import redis.asyncio as aioredis
                self._client = aioredis.from_url(redis_url)
            except ImportError:
                log.warning("[rate_limiter] redis not installed; using local bucket for %s", service)

    async def acquire(self, tokens: float = 1.0) -> None:
        """Acquire tokens, using Redis if available, local bucket otherwise."""
        if self._client is None:
            await self._fallback.acquire(tokens)
            return
        key = f"rate_limit:{self.service}"
        while True:
            try:
                result = await self._client.eval(
                    self._LUA_SCRIPT, 1, key,
                    self.rate, self.capacity, tokens, time.monotonic(),
                )
                if result:
                    return
                wait_for = tokens / self.rate
                await asyncio.sleep(wait_for)
            except Exception as exc:
                log.warning("[rate_limiter] Redis eval failed (%s); using local bucket", exc)
                await self._fallback.acquire(tokens)
                return


# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_limiters: Dict[str, TokenBucket] = {}


def get_limiter(service: str) -> TokenBucket:
    """Return the singleton rate limiter for a named external service.

    Limiter instances are created once and cached for the process lifetime.
    The RPS is resolved from ``settings.ratelimit_<service>_rps`` with a
    fallback to ``_DEFAULT_RPS``.

    Args:
        service: Service name (e.g. ``"serper"``, ``"wikipedia"``).

    Returns:
        A :class:`TokenBucket` (or :class:`RedisTokenBucket`) instance.
    """
    global _limiters
    if service not in _limiters:
        rps_key = f"ratelimit_{service}_rps"
        rps = float(getattr(settings, rps_key, _DEFAULT_RPS.get(service, _DEFAULT_RPS["default"])))
        redis_url = getattr(settings, "redis_url", "").strip()
        if redis_url:
            limiter = RedisTokenBucket(service, rps, redis_url=redis_url)
        else:
            limiter = TokenBucket(rps)
        _limiters[service] = limiter
        log.debug("[rate_limiter] Created limiter for '%s' @ %.1f rps", service, rps)
    return _limiters[service]
