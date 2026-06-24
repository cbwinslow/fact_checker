"""EvidenceRetrievalAgent - retrieves evidence for each claim.

Priority order:
1. Google Fact Check Tools API (ClaimReview database)
2. Serper.dev web search (general evidence)
3. Fallback: empty evidence with low confidence flag
"""
from __future__ import annotations
import logging
from pathlib import Path
from uuid import UUID
import httpx

from ..config import settings
from ..models import Claim, EvidenceItem

log = logging.getLogger(__name__)


async def _google_factcheck(claim_text: str, claim_id: UUID) -> list[EvidenceItem]:
    """Query Google Fact Check Tools API."""
    if not settings.google_factcheck_api_key:
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": claim_text, "key": settings.google_factcheck_api_key, "pageSize": 5}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            return []
        data = resp.json()
    items = []
    for result in data.get("claims", []):
        for review in result.get("claimReview", []):
            items.append(EvidenceItem(
                claim_id=claim_id,
                source_url=review.get("url", ""),
                title=review.get("publisher", {}).get("name", ""),
                snippet=f"{review.get('textualRating', '')} - {result.get('text', '')}",
                relevance_score=0.9,
                is_factcheck_source=True,
            ))
    return items


async def _serper_search(claim_text: str, claim_id: UUID) -> list[EvidenceItem]:
    """Use Serper.dev for general web evidence retrieval."""
    if not settings.serper_api_key:
        return []
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"},
            json={"q": claim_text, "num": 5},
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
    items = []
    for r in data.get("organic", [])[:5]:
        items.append(EvidenceItem(
            claim_id=claim_id,
            source_url=r.get("link", ""),
            title=r.get("title", ""),
            snippet=r.get("snippet", ""),
            relevance_score=0.6,
            is_factcheck_source=False,
        ))
    return items


async def retrieve_evidence(claims: list[Claim]) -> list[EvidenceItem]:
    """Retrieve evidence for all claims; returns combined EvidenceItem list."""
    all_evidence: list[EvidenceItem] = []
    for claim in claims:
        if not claim.is_checkable:
            continue
        log.info("[EvidenceAgent] Retrieving for: %s", claim.text[:80])
        # Layer 1: Google Fact Check
        fc_items = await _google_factcheck(claim.text, claim.id)
        if fc_items:
            all_evidence.extend(fc_items)
            continue
        # Layer 2: Serper web search
        web_items = await _serper_search(claim.text, claim.id)
        all_evidence.extend(web_items)
        if not web_items:
            log.warning("[EvidenceAgent] No evidence found for claim: %s", claim.text[:80])
    log.info("[EvidenceAgent] Retrieved %d evidence items for %d claims", len(all_evidence), len(claims))
    return all_evidence
