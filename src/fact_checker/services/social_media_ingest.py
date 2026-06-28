"""services/social_media_ingest.py - Social media content ingestion.

Ingests posts, threads, and articles from social media platforms into
``TranscriptSegment`` objects that the rest of the pipeline can process.

Supported platforms:
  - Twitter / X  (via ``tweepy`` with OAuth2 bearer token)
  - Reddit        (via ``praw`` with read-only app credentials)
  - Nitter        (public scrape fallback when Tweepy is unavailable)
  - Generic URL   (falls back to ``web_scraper.scrape_article``)

Environment / Settings variables required::

    TWITTER_BEARER_TOKEN   - Twitter API v2 Bearer Token
    REDDIT_CLIENT_ID       - Reddit app client ID
    REDDIT_CLIENT_SECRET   - Reddit app client secret
    REDDIT_USER_AGENT      - Reddit app user agent string

Dependencies (optional)::
    pip install tweepy praw
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple
from uuid import UUID

from ..config import get_settings
from ..models import IngestSource, TranscriptSegment

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL pattern helpers
# ---------------------------------------------------------------------------

_TWITTER_RE = re.compile(
    r"https?://(?:www\.)?(?:twitter\.com|x\.com)/[^/]+/status/(\d+)"
)
_REDDIT_POST_RE = re.compile(
    r"https?://(?:www\.)?reddit\.com/r/[^/]+/comments/([a-z0-9]+)"
)
_REDDIT_PROFILE_RE = re.compile(
    r"https?://(?:www\.)?reddit\.com/u(?:ser)?/([^/]+)/?$"
)


def is_social_media_url(url: str) -> bool:
    """Return True if the URL points to a supported social media platform.

    Args:
        url: URL string to test.

    Returns:
        True if the URL matches a known social media pattern.
    """
    return bool(
        _TWITTER_RE.search(url)
        or _REDDIT_POST_RE.search(url)
        or _REDDIT_PROFILE_RE.search(url)
    )


async def ingest_social_url(
    job_id: UUID,
    url: str,
) -> Tuple[List[TranscriptSegment], IngestSource]:
    """Route a social media URL to the correct platform ingestor.

    Args:
        job_id: UUID of the owning pipeline job.
        url: Social media URL.

    Returns:
        Tuple of (segments, IngestSource).

    Raises:
        ValueError: If the URL does not match any supported platform.
    """
    if _TWITTER_RE.search(url):
        return await _ingest_tweet(job_id, url)
    if _REDDIT_POST_RE.search(url) or _REDDIT_PROFILE_RE.search(url):
        return await _ingest_reddit(job_id, url)
    raise ValueError(f"[social_media_ingest] Unsupported URL: {url}")


# ---------------------------------------------------------------------------
# Twitter / X
# ---------------------------------------------------------------------------

async def _ingest_tweet(
    job_id: UUID,
    url: str,
) -> Tuple[List[TranscriptSegment], IngestSource]:
    """Ingest a tweet/thread using Tweepy v4 (API v2), falling back to Nitter.

    Args:
        job_id: Pipeline job UUID.
        url: Tweet URL.

    Returns:
        Tuple of (segments, WEB_ARTICLE | SCREENSHOT IngestSource).
    """
    bearer = getattr(settings, "twitter_bearer_token", "").strip()
    if bearer:
        try:
            return await _tweepy_ingest(job_id, url, bearer)
        except Exception as exc:
            log.warning("[social] Tweepy failed (%s) - falling back to Nitter", exc)
    return await _nitter_ingest(job_id, url)


async def _tweepy_ingest(
    job_id: UUID,
    url: str,
    bearer_token: str,
) -> Tuple[List[TranscriptSegment], IngestSource]:
    """Use the Tweepy Twitter API v2 client to fetch tweet text."""
    import tweepy
    import asyncio

    match = _TWITTER_RE.search(url)
    if not match:
        raise ValueError(f"Could not extract tweet ID from {url}")
    tweet_id = int(match.group(1))

    client = tweepy.Client(bearer_token=bearer_token)
    loop = asyncio.get_event_loop()

    response = await loop.run_in_executor(
        None,
        lambda: client.get_tweet(
            tweet_id,
            tweet_fields=["text", "created_at", "author_id", "conversation_id"],
            expansions=["author_id"],
        ),
    )
    if not response.data:
        raise ValueError(f"Tweet {tweet_id} not found or not accessible")

    tweet = response.data
    author = ""
    if response.includes and response.includes.get("users"):
        author = response.includes["users"][0].username

    # Also fetch conversation thread (up to 10 replies)
    texts = [f"@{author}: {tweet.text}"]
    try:
        thread_resp = await loop.run_in_executor(
            None,
            lambda: client.search_recent_tweets(
                query=f"conversation_id:{tweet.conversation_id}",
                tweet_fields=["text", "author_id"],
                max_results=10,
            ),
        )
        if thread_resp.data:
            for t in thread_resp.data:
                if t.id != tweet_id:
                    texts.append(t.text)
    except Exception as exc:
        log.debug("[social] Thread fetch failed: %s", exc)

    full_text = "\n".join(texts)
    from .file_router import _text_to_segments
    segments = _text_to_segments(job_id, full_text, source_name=f"tweet:{tweet_id}")
    log.info("[social] Tweepy: %d segments from tweet %d", len(segments), tweet_id)
    return segments, IngestSource.WEB_ARTICLE


async def _nitter_ingest(
    job_id: UUID,
    url: str,
) -> Tuple[List[TranscriptSegment], IngestSource]:
    """Scrape tweet text via the Nitter public mirror (no API key required)."""
    import httpx
    from .web_scraper import html_to_text
    from .file_router import _text_to_segments

    # Convert twitter.com / x.com URL to nitter.net
    nitter_url = re.sub(r"https?://(?:www\.)?(?:twitter|x)\.com", "https://nitter.net", url)
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(nitter_url, headers={"User-Agent": "FactCheckerBot/1.0"})
            resp.raise_for_status()
            text = html_to_text(resp.text, url=nitter_url)
    except Exception as exc:
        log.warning("[social] Nitter scrape failed for %s: %s", url, exc)
        # Final fallback: use generic web scraper
        from .web_scraper import scrape_article
        return await scrape_article(job_id=job_id, url=url)

    segments = _text_to_segments(job_id, text, source_name=f"nitter:{url}")
    log.info("[social] Nitter: %d segments from %s", len(segments), url)
    return segments, IngestSource.WEB_ARTICLE


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------

async def _ingest_reddit(
    job_id: UUID,
    url: str,
) -> Tuple[List[TranscriptSegment], IngestSource]:
    """Ingest a Reddit post (with top comments) or user profile.

    Uses PRAW if credentials are configured; falls back to the JSON API
    (reddit.com/url.json) otherwise.

    Args:
        job_id: Pipeline job UUID.
        url: Reddit URL.

    Returns:
        Tuple of (segments, WEB_ARTICLE).
    """
    client_id = getattr(settings, "reddit_client_id", "").strip()
    client_secret = getattr(settings, "reddit_client_secret", "").strip()
    user_agent = getattr(settings, "reddit_user_agent", "FactCheckerBot/1.0").strip()

    if client_id and client_secret:
        try:
            return await _praw_ingest(job_id, url, client_id, client_secret, user_agent)
        except Exception as exc:
            log.warning("[social] PRAW failed (%s) - falling back to JSON API", exc)

    return await _reddit_json_ingest(job_id, url)


async def _praw_ingest(
    job_id: UUID,
    url: str,
    client_id: str,
    client_secret: str,
    user_agent: str,
) -> Tuple[List[TranscriptSegment], IngestSource]:
    """Use PRAW (read-only) to fetch post title, selftext, and top comments."""
    import praw
    import asyncio
    from .file_router import _text_to_segments

    loop = asyncio.get_event_loop()
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    submission = await loop.run_in_executor(
        None, lambda: reddit.submission(url=url)
    )
    parts = [f"Title: {submission.title}"]
    if submission.selftext:
        parts.append(submission.selftext)
    # Top 10 top-level comments
    await loop.run_in_executor(None, lambda: submission.comments.replace_more(limit=0))
    for comment in list(submission.comments)[:10]:
        if hasattr(comment, "body") and comment.body != "[deleted]":
            parts.append(comment.body)
    full_text = "\n\n".join(parts)
    segments = _text_to_segments(job_id, full_text, source_name=f"reddit:{url}")
    log.info("[social] PRAW: %d segments from %s", len(segments), url)
    return segments, IngestSource.WEB_ARTICLE


async def _reddit_json_ingest(
    job_id: UUID,
    url: str,
) -> Tuple[List[TranscriptSegment], IngestSource]:
    """Fallback: use Reddit's unauthenticated JSON API."""
    import httpx
    from .file_router import _text_to_segments

    json_url = url.rstrip("/") + ".json?limit=10"
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(
                json_url,
                headers={"User-Agent": "FactCheckerBot/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        log.warning("[social] Reddit JSON API failed: %s", exc)
        from .web_scraper import scrape_article
        return await scrape_article(job_id=job_id, url=url)

    parts = []
    try:
        post = data[0]["data"]["children"][0]["data"]
        parts.append(f"Title: {post.get('title', '')}")  
        if post.get("selftext"):
            parts.append(post["selftext"])
        comments = data[1]["data"]["children"]
        for child in comments[:10]:
            body = child.get("data", {}).get("body", "")
            if body and body != "[deleted]":
                parts.append(body)
    except (KeyError, IndexError, TypeError) as exc:
        log.warning("[social] Reddit JSON parse error: %s", exc)

    full_text = "\n\n".join(parts) or "(no content extracted)"
    segments = _text_to_segments(job_id, full_text, source_name=f"reddit_json:{url}")
    log.info("[social] Reddit JSON: %d segments from %s", len(segments), url)
    return segments, IngestSource.WEB_ARTICLE
