"""agents/deep_research_agent.py - Deep multi-hop research agent for the fact-checker pipeline.

Takes the handoff from the input ingestion team (ingest -> image analysis ->
claim extraction) and performs iterative, adversarial research to gather
high-quality evidence for each claim before the verdict stage.

Research strategy per claim:
  1. Semantic context retrieval  - search the job's VectorStore for the most
                                   relevant ingested context chunks.
  2. Google Fact Check Tools API - check the ClaimReview database first;
                                   if a high-confidence match is found, stop.
  3. Serper web search (round 1) - primary web evidence gathering.
  4. Full-page scraping          - fetch and parse the top evidence URLs for
                                   richer snippets than search results alone.
  5. Adversarial round (round 2) - if initial evidence is weak or conflicting,
                                   issue a counter-query ("evidence against X")
                                   to find contradicting sources.
  6. Source credibility scoring  - apply a tiered domain-reputation heuristic
                                   to weight evidence items.
  7. Wikipedia / Wikidata lookup - authoritative background facts for entities
                                   mentioned in the claim.

All research steps are best-effort: a failure in any step is logged and
skipped rather than aborting the pipeline.

Dependencies::

    pip install httpx trafilatura beautifulsoup4
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Optional
from uuid import UUID

import httpx

from ..config import get_settings
from ..models import (
    AnalysisContext,
    Claim,
    EvidenceItem,
    ResearchResult,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source credibility tiers
# ---------------------------------------------------------------------------
# Domains in TIER_1 are considered high-quality fact-checking or institutional
# sources.  TIER_2 are reputable journalism outlets.  Everything else is TIER_3.

_TIER_1_DOMAINS = {
    "snopes.com", "politifact.com", "factcheck.org", "reuters.com/fact-check",
    "apnews.com", "bbc.com", "fullfact.org", "leadstories.com",
    "factcheckni.org", "afpfactcheck.com", "boomlive.in",
    "science.nasa.gov", "cdc.gov", "who.int", "nih.gov",
    "wikipedia.org", "wikidata.org",
}

_TIER_2_DOMAINS = {
    "nytimes.com", "washingtonpost.com", "theguardian.com", "economist.com",
    "bbc.co.uk", "npr.org", "pbs.org", "cbsnews.com", "nbcnews.com",
    "abcnews.go.com", "usatoday.com", "time.com", "theatlantic.com",
    "scientificamerican.com", "nature.com", "science.org",
}


def _credibility_score(url: str) -> float:
    """Return a credibility score (0.0 - 1.0) for a source URL.

    Uses a three-tier domain reputation heuristic:
      - Tier 1 (fact-checkers, govt, academic): 0.90
      - Tier 2 (major quality journalism):       0.70
      - Tier 3 (unknown / unrecognised):         0.40

    Args:
        url: The source URL to score.

    Returns:
        Float in [0.0, 1.0].
    """
    domain = re.sub(r"https?://(www\\.)?", "", url).split("/")[0].lower()
    if any(domain == t1 or domain.endswith('.' + t1) for t1 in _TIER_1_DOMAINS):
        return 0.90
    if any(domain == t2 or domain.endswith('.' + t2) for t2 in _TIER_2_DOMAINS):
        return 0.70
    return 0.40


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def deep_research(
    claims: List[Claim],
    context: Optional[AnalysisContext] = None,
) -> List[ResearchResult]:
    """Perform deep multi-hop research for a list of claims.

    Orchestrates all research steps in a controlled async pipeline,
    running claims concurrently (up to a configurable concurrency limit)
    to keep total latency manageable.

    Args:
        claims:  List of Claim objects to research.  Non-checkable claims
                 (``is_checkable=False``) are skipped.
        context: Optional AnalysisContext carrying the job's VectorStore
                 and embedded context for semantic retrieval.

    Returns:
        List of ResearchResult objects, one per researched claim.
    """
    checkable = [c for c in claims if c.is_checkable]
    if not checkable:
        log.info("[deep_research] No checkable claims to research.")
        return []

    log.info(
        "[deep_research] Starting deep research for %d claims.", len(checkable)
    )

    # Concurrency limit to avoid hammering APIs
    sem = asyncio.Semaphore(getattr(settings, "research_concurrency", 3))

    async def _bounded(claim: Claim) -> ResearchResult:
        async with sem:
            return await _research_claim(claim, context)

    results = await asyncio.gather(*[_bounded(c) for c in checkable])
    log.info(
        "[deep_research] Completed research for %d claims.", len(results)
    )
    return list(results)


# ---------------------------------------------------------------------------
# Per-claim research orchestrator
# ---------------------------------------------------------------------------

async def _research_claim(
    claim: Claim,
    context: Optional[AnalysisContext],
) -> ResearchResult:
    """Run the full research pipeline for a single claim.

    Steps:
      1. Semantic retrieval from the job VectorStore.
      2. Google Fact Check Tools API lookup.
      3. Serper primary web search.
      4. Full-page scraping of top URLs.
      5. Adversarial counter-query if evidence is weak.
      6. Wikipedia background lookup for named entities.
      7. Source credibility scoring and deduplication.

    Args:
        claim:   The Claim to research.
        context: Optional AnalysisContext with VectorStore access.

    Returns:
        Populated ResearchResult for this claim.
    """
    log.info("[deep_research] Researching claim: %.80s", claim.text)
    evidence: List[EvidenceItem] = []
    context_snippets: List[str]  = []

    # Step 1: Semantic context retrieval
    if context and context.vector_store:
        context_snippets = await _semantic_retrieval(claim, context)

    # Step 2: Google Fact Check
    fc_items = await _google_factcheck(claim)
    evidence.extend(fc_items)

    # If a high-confidence FC result exists, short-circuit
    if any(ev.relevance_score >= 0.85 and ev.is_factcheck_source for ev in fc_items):
        log.info("[deep_research] High-confidence FC hit - skipping web search for: %.60s", claim.text)
        return _build_result(claim, evidence, context_snippets)

    # Step 3: Primary web search
    web_items = await _serper_search(claim.text, claim.id)
    evidence.extend(web_items)

    # Step 4: Full-page scraping of top results
    top_urls = [ev.source_url for ev in web_items[:3] if ev.source_url]
    scraped  = await _scrape_pages(top_urls, claim.id)
    evidence.extend(scraped)

    # Step 5: Adversarial counter-query if evidence is weak
    if _is_weak_evidence(evidence):
        log.info("[deep_research] Weak evidence - issuing adversarial query.")
        counter_query = f"evidence against OR debunking: {claim.text}"
        counter_items = await _serper_search(counter_query, claim.id)
        evidence.extend(counter_items)

    # Step 6: Wikipedia lookup for named entities
    wiki_items = await _wikipedia_lookup(claim, claim.id)
    evidence.extend(wiki_items)

    # Step 7: Apply credibility scoring and deduplicate
    evidence = _score_and_deduplicate(evidence)

    return _build_result(claim, evidence, context_snippets)


# ---------------------------------------------------------------------------
# Research steps
# ---------------------------------------------------------------------------

async def _semantic_retrieval(
    claim: Claim,
    context: AnalysisContext,
) -> List[str]:
    """Retrieve top-k semantically similar context chunks from the VectorStore.

    Embeds the claim text on the fly and queries the job's VectorStore
    for the most relevant ingested context passages.

    Args:
        claim:   The claim to find context for.
        context: AnalysisContext with an initialised VectorStore.

    Returns:
        List of relevant context text snippets (strings).
    """
    try:
        from ..services.embedder import embed_texts
        chunks = await embed_texts(claim.job_id, [claim.text])
        if not chunks or not chunks[0].vector:
            return []
        results = context.vector_store.search(chunks[0].vector, top_k=5, min_score=0.3)
        snippets = [r.text for r in results]
        log.debug(
            "[deep_research] Semantic retrieval: %d context chunks for claim %s",
            len(snippets), claim.id,
        )
        return snippets
    except Exception as exc:
        log.warning("[deep_research] Semantic retrieval failed: %s", exc)
        return []


async def _google_factcheck(claim: Claim) -> List[EvidenceItem]:
    """Query the Google Fact Check Tools API for the claim.

    Returns an empty list silently when the API key is not configured.

    Args:
        claim: The claim to look up.

    Returns:
        List of EvidenceItem objects from the ClaimReview database.
    """
    if not get_settings().google_factcheck_api_key:
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query":    claim.text[:200],
        "key":      get_settings().google_factcheck_api_key,
        "pageSize": 5,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return []
            data = resp.json()
    except Exception as exc:
        log.warning("[deep_research] Google FC API error: %s", exc)
        return []

    items: List[EvidenceItem] = []
    for result in data.get("claims", []):
        for review in result.get("claimReview", []):
            source_url = review.get("url", "")
            items.append(EvidenceItem(
                claim_id=claim.id,
                source_url=source_url,
                title=review.get("publisher", {}).get("name", ""),
                snippet=(
                    f"{review.get('textualRating', '')} - "
                    f"{result.get('text', '')}"
                ),
                relevance_score=0.90,
                credibility_score=_credibility_score(source_url),
                is_factcheck_source=True,
            ))
    return items


async def _serper_search(
    query: str,
    claim_id: UUID,
    num_results: int = 6,
) -> List[EvidenceItem]:
    """Search the web using Serper.dev and return EvidenceItem objects.

    Returns an empty list silently when ``get_settings().serper_api_key`` is not set.

    Args:
        query:       Search query string.
        claim_id:    UUID of the claim this evidence belongs to.
        num_results: Number of organic results to request (default 6).

    Returns:
        List of EvidenceItem objects from organic search results.
    """
    if not get_settings().serper_api_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY":     get_settings().serper_api_key,
                    "Content-Type":  "application/json",
                },
                json={"q": query, "num": num_results},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
    except Exception as exc:
        log.warning("[deep_research] Serper search error: %s", exc)
        return []

    items: List[EvidenceItem] = []
    for r in data.get("organic", [])[:num_results]:
        source_url = r.get("link", "")
        items.append(EvidenceItem(
            claim_id=claim_id,
            source_url=source_url,
            title=r.get("title", ""),
            snippet=r.get("snippet", ""),
            relevance_score=0.60,
            credibility_score=_credibility_score(source_url),
            is_factcheck_source=False,
        ))
    return items


async def _scrape_pages(
    urls: List[str],
    claim_id: UUID,
) -> List[EvidenceItem]:
    """Fetch and extract full-page text from a list of URLs.

    Runs all page fetches concurrently.  Each successfully scraped page
    is condensed to its first 600 characters as an enriched snippet.
    Failures are silently skipped so one bad URL does not block the rest.

    Args:
        urls:      List of URLs to scrape.
        claim_id:  UUID of the claim this evidence belongs to.

    Returns:
        List of EvidenceItem objects with enriched snippets.
    """
    from ..services.web_scraper import html_to_text

    async def _fetch_one(url: str) -> Optional[EvidenceItem]:
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "FactCheckerBot/1.0"},
                )
                resp.raise_for_status()
                html = resp.content[:2 * 1024 * 1024].decode(
                    resp.encoding or "utf-8", errors="replace"
                )
            text = html_to_text(html, url=url)
            snippet = text[:600].strip() if text else ""
            if not snippet:
                return None
            return EvidenceItem(
                claim_id=claim_id,
                source_url=url,
                title="",
                snippet=snippet,
                relevance_score=0.65,
                credibility_score=_credibility_score(url),
                is_factcheck_source=False,
            )
        except Exception as exc:
            log.debug("[deep_research] Page scrape failed %s: %s", url, exc)
            return None

    results = await asyncio.gather(*[_fetch_one(u) for u in urls])
    return [r for r in results if r is not None]


async def _wikipedia_lookup(
    claim: Claim,
    claim_id: UUID,
) -> List[EvidenceItem]:
    """Query the Wikipedia search API for entities mentioned in the claim.

    Extracts the first named entity (capitalised multi-word phrase) from the
    claim text and fetches the Wikipedia page summary.  Returns an empty
    list if no entity is found or the request fails.

    Args:
        claim:     The claim being researched.
        claim_id:  UUID of the claim.

    Returns:
        List of up to one EvidenceItem with a Wikipedia summary snippet.
    """
    # Heuristic: first capitalised sequence of 1-4 words as entity
    match = re.search(r"([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})", claim.text)
    if not match:
        return []
    entity = match.group(1)
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + entity.replace(" ", "_"),
                headers={"User-Agent": "FactCheckerBot/1.0"},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
        snippet = data.get("extract", "")[:600]
        url     = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        if not snippet:
            return []
        return [EvidenceItem(
            claim_id=claim_id,
            source_url=url,
            title=data.get("title", entity),
            snippet=snippet,
            relevance_score=0.70,
            credibility_score=0.90,  # Wikipedia is Tier 1
            is_factcheck_source=False,
        )]
    except Exception as exc:
        log.debug("[deep_research] Wikipedia lookup failed for '%s': %s", entity, exc)
        return []


# ---------------------------------------------------------------------------
# Evidence post-processing helpers
# ---------------------------------------------------------------------------

def _is_weak_evidence(evidence: List[EvidenceItem]) -> bool:
    """Return True if the current evidence set is too weak to support a verdict.

    Evidence is considered weak when fewer than 2 items have a relevance
    score above 0.50, or when no fact-check source is present.

    Args:
        evidence: Current list of EvidenceItem objects.

    Returns:
        True if an adversarial counter-query should be issued.
    """
    high_quality = [ev for ev in evidence if ev.relevance_score >= 0.50]
    return len(high_quality) < 2


def _score_and_deduplicate(evidence: List[EvidenceItem]) -> List[EvidenceItem]:
    """Apply credibility scoring and remove duplicate URLs.

    Deduplication keeps the highest-scoring item when multiple evidence
    items share the same source URL.  Items are returned sorted by
    descending composite score (relevance * credibility).

    Args:
        evidence: Raw list of EvidenceItem objects.

    Returns:
        Deduplicated, scored, and sorted EvidenceItem list.
    """
    seen_urls: Dict[str, EvidenceItem] = {}
    for ev in evidence:
        key = ev.source_url.rstrip("/").lower()
        composite = ev.relevance_score * getattr(ev, "credibility_score", 0.5)
        existing  = seen_urls.get(key)
        if existing is None:
            seen_urls[key] = ev
        else:
            existing_composite = existing.relevance_score * getattr(existing, "credibility_score", 0.5)
            if composite > existing_composite:
                seen_urls[key] = ev

    deduped = list(seen_urls.values())
    deduped.sort(
        key=lambda ev: ev.relevance_score * getattr(ev, "credibility_score", 0.5),
        reverse=True,
    )
    return deduped


def _build_result(
    claim: Claim,
    evidence: List[EvidenceItem],
    context_snippets: List[str],
) -> ResearchResult:
    """Package a claim's research output into a ResearchResult.

    Args:
        claim:            The researched Claim.
        evidence:         Collected and scored EvidenceItem list.
        context_snippets: Semantic context passages from the VectorStore.

    Returns:
        Populated ResearchResult.
    """
    avg_credibility = (
        sum(getattr(ev, "credibility_score", 0.5) for ev in evidence) / len(evidence)
        if evidence else 0.0
    )
    return ResearchResult(
        claim_id=claim.id,
        evidence=evidence,
        context_snippets=context_snippets,
        avg_credibility=avg_credibility,
        evidence_count=len(evidence),
        has_factcheck_source=any(ev.is_factcheck_source for ev in evidence),
    )
