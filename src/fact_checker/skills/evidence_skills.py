"""evidence_skills.py - Reusable skill functions for evidence processing.

File: src/fact_checker/skills/evidence_skills.py

Provides stateless utility functions used by the EvidenceRetrievalAgent and
DeepResearchAgent to score source credibility, generate optimised search
queries, rank evidence snippets, and identify fact-check domains.

All functions are pure (no I/O, no LLM calls) and safe to unit-test
without external dependencies.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Domain credibility tiers
# Each set contains registered domains that are known to host high-quality
# fact-checking or primary-source content.
# ---------------------------------------------------------------------------

_TIER1_FACTCHECK_DOMAINS: frozenset[str] = frozenset({
    "snopes.com", "politifact.com", "factcheck.org", "reuters.com",
    "apnews.com", "bbc.com", "bbc.co.uk", "fullfact.org",
    "factcheckni.org", "afpfactcheck.com", "boomlive.in",
    "science.nasa.gov", "cdc.gov", "who.int", "nih.gov",
    "wikipedia.org", "wikidata.org", "leadstories.com",
})

_TIER2_DOMAINS: frozenset[str] = frozenset({
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "economist.com", "theatlantic.com", "ft.com",
    "npr.org", "pbs.org", "propublica.org",
    "nature.com", "science.org", "thelancet.com",
})

_UNRELIABLE_SIGNALS: tuple[str, ...] = (
    "infowars", "naturalnews", "breitbart", "theonion",
    "babylonbee", "clickhole", "worldnewsdailyreport",
)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def is_factcheck_domain(url: str) -> bool:
    """Return True if the URL belongs to a Tier-1 fact-checking domain.

    Used to give automatic high relevance weighting to results from
    established fact-checking organisations.

    Args:
        url: Absolute or relative URL string.

    Returns:
        True if the domain is in the Tier-1 fact-check set.
    """
    try:
        hostname = urlparse(url).hostname or ""
        # Strip www. prefix for matching
        hostname = re.sub(r"^www\.", "", hostname.lower())
        return hostname in _TIER1_FACTCHECK_DOMAINS
    except Exception:
        return False


def score_source_credibility(url: str) -> Tuple[int, float]:
    """Score a source URL for credibility, returning (tier, score).

    Tier 1 = established fact-checkers / primary sources  (score 0.85-1.0)
    Tier 2 = major reputable journalism / academic        (score 0.60-0.84)
    Tier 3 = minor outlets / sourced blogs                (score 0.30-0.59)
    Tier 4 = social media / anonymous / unreliable        (score 0.00-0.29)

    Args:
        url: Absolute URL string of the evidence source.

    Returns:
        Tuple of (tier: int, score: float).
    """
    try:
        hostname = urlparse(url).hostname or ""
        hostname = re.sub(r"^www\.", "", hostname.lower())
    except Exception:
        return 4, 0.1

    # Tier 4: known unreliable
    if any(sig in hostname for sig in _UNRELIABLE_SIGNALS):
        return 4, 0.05

    # Tier 1: fact-check / primary
    if hostname in _TIER1_FACTCHECK_DOMAINS:
        return 1, 0.92

    # .gov / .edu / .int domains are generally authoritative
    if hostname.endswith((".gov", ".edu", ".int", ".ac.uk")):
        return 1, 0.88

    # Tier 2: major journalism / academic publishers
    if hostname in _TIER2_DOMAINS:
        return 2, 0.75

    # Tier 3: everything else with a recognisable TLD
    if hostname.endswith((".com", ".org", ".net", ".co.uk")):
        return 3, 0.45

    return 4, 0.20


def generate_search_queries(claim_text: str, claim_type: str = "unknown") -> List[str]:
    """Generate 3-5 targeted search queries for a given claim.

    Query strategies vary by claim type:
    - ``statistical``: adds "statistics", "data", "study" suffixes.
    - ``historical``:  adds "history", "timeline", "when did" prefix variants.
    - ``attributional``: extracts the attributed entity and generates
                         quote-verification style queries.
    - Default: generates a direct query, a negation query ("[claim] false"),
               and a fact-check query ("[claim] fact check").

    Args:
        claim_text: The normalised claim string.
        claim_type: Semantic type from ``classify_claim_type``.

    Returns:
        List of 3-5 distinct search query strings, ordered by priority.
    """
    # Extract a compact claim summary (first 120 chars)
    summary = claim_text[:120].rstrip(",;")

    base_queries = [
        summary,
        f"{summary} fact check",
        f"is it true that {summary.lower()}",
    ]

    if claim_type == "statistical":
        # Find any numbers/percentages and build data-focused queries
        numbers = re.findall(r"\d+\.?\d*%?|\d+ million|\d+ billion", claim_text)
        num_str = numbers[0] if numbers else ""
        base_queries += [
            f"{summary} source data study",
            f"{num_str} statistics official data" if num_str else f"{summary} official data",
        ]

    elif claim_type == "historical":
        base_queries += [
            f"{summary} history timeline",
            f"{summary} when did this happen",
        ]

    elif claim_type == "attributional":
        # Try to extract who is being quoted
        match = re.search(
            r"([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})"
            r".{0,30}(?:said|stated|claimed|announced|tweeted|wrote)",
            claim_text,
        )
        if match:
            entity = match.group(1)
            base_queries += [
                f"{entity} quote verification",
                f"{entity} statement {summary[:60]}",
            ]
        else:
            base_queries.append(f"{summary} quote source verification")

    elif claim_type == "causal":
        base_queries += [
            f"{summary} research evidence",
            f"{summary} debunked",
        ]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in base_queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            unique.append(q)

    return unique[:5]


def rank_evidence_snippets(
    snippets: List[dict],
    claim_text: str,
    max_results: int = 8,
) -> List[dict]:
    """Rank retrieved evidence snippets by a composite relevance score.

    Scoring formula per snippet:
        score = (term_overlap * 0.4) + (credibility_score * 0.4)
                + (factcheck_bonus * 0.2)

    Where:
    - ``term_overlap``: Jaccard similarity between claim tokens and snippet tokens.
    - ``credibility_score``: from ``score_source_credibility``.
    - ``factcheck_bonus``: 1.0 if ``is_factcheck_domain``, else 0.0.

    Args:
        snippets: List of evidence dicts with at least ``source_url`` and
                  ``snippet`` keys.  ``relevance_score`` and
                  ``is_factcheck_source`` are updated in place.
        claim_text: The claim being researched.
        max_results: Maximum number of snippets to return after ranking.

    Returns:
        Top ``max_results`` snippets sorted by composite score descending.
    """
    claim_tokens = set(re.findall(r"\b[a-z]{3,}\b", claim_text.lower()))

    scored: list[tuple[float, dict]] = []
    for item in snippets:
        url = item.get("source_url", "")
        snippet_text = item.get("snippet", "") + " " + item.get("title", "")
        snippet_tokens = set(re.findall(r"\b[a-z]{3,}\b", snippet_text.lower()))

        # Jaccard similarity
        if claim_tokens | snippet_tokens:
            overlap = len(claim_tokens & snippet_tokens) / len(claim_tokens | snippet_tokens)
        else:
            overlap = 0.0

        _tier, cred_score = score_source_credibility(url)
        fc_bonus = 1.0 if is_factcheck_domain(url) else 0.0

        composite = (overlap * 0.4) + (cred_score * 0.4) + (fc_bonus * 0.2)

        # Persist scores into the dict for downstream consumers
        item["relevance_score"] = round(composite, 4)
        item["is_factcheck_source"] = bool(fc_bonus)
        scored.append((composite, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:max_results]]


# ---------------------------------------------------------------------------
# Source diversity & contradiction detection
# ---------------------------------------------------------------------------


def enforce_source_diversity(evidence: List[EvidenceItem]) -> Dict[str, object]:
    """Enforce source diversity requirements on evidence set.
    
    Returns a diversity report with pass/fail status and details.
    """
    from datetime import datetime
    domains = set(ev.domain for ev in evidence if ev.domain)
    types = set(ev.source_type for ev in evidence if ev.source_type)
    has_factcheck = any(ev.is_factcheck_source for ev in evidence)
    
    # Temporal freshness
    now = datetime.now()
    fresh_count = 0
    for ev in evidence:
        if ev.published_date:
            age_days = (now - ev.published_date).days
            if age_days < 365:
                fresh_count += 1
    
    meets_diversity = (
        len(domains) >= 3 and
        len(types) >= 2 and
        (has_factcheck or len(evidence) >= 5)
    )
    
    return {
        "unique_domains": len(domains),
        "domains": list(domains),
        "source_types": list(types),
        "has_factcheck_source": has_factcheck,
        "fresh_sources": fresh_count,
        "total_sources": len(evidence),
        "meets_diversity": meets_diversity,
    }


def detect_contradictions(evidence: List[EvidenceItem]) -> List[Dict[str, object]]:
    """Detect contradictory evidence using stance classification.
    
    Returns list of contradiction pairs with details.
    """
    # This is a heuristic - in production, would use LLM stance classification
    # For now, we detect based on high-relevance items from different domains
    # that might contradict (simplified)
    
    high_relevance = [ev for ev in evidence if ev.relevance_score >= 0.7]
    contradictions = []
    
    # Simple heuristic: if we have fact-check sources with different ratings
    factcheck_items = [ev for ev in high_relevance if ev.is_factcheck_source]
    if len(factcheck_items) >= 2:
        # Check if snippets suggest different conclusions
        for i, ev1 in enumerate(factcheck_items):
            for ev2 in factcheck_items[i+1:]:
                # Very simplified - would need NLP in production
                if "true" in ev1.snippet.lower() and "false" in ev2.snippet.lower():
                    contradictions.append({
                        "evidence_1_id": str(ev1.id),
                        "evidence_2_id": str(ev2.id),
                        "type": "factcheck_disagreement",
                        "description": "Fact-check sources disagree on claim veracity",
                    })
                elif "false" in ev1.snippet.lower() and "true" in ev2.snippet.lower():
                    contradictions.append({
                        "evidence_1_id": str(ev1.id),
                        "evidence_2_id": str(ev2.id),
                        "type": "factcheck_disagreement",
                        "description": "Fact-check sources disagree on claim veracity",
                    })
    
    return contradictions


def check_diversity_and_contradictions(evidence: List[EvidenceItem]) -> Dict[str, object]:
    """Combined check for diversity and contradictions."""
    diversity_report = enforce_source_diversity(evidence)
    contradictions = detect_contradictions(evidence)
    
    return {
        "diversity": diversity_report,
        "contradictions": contradictions,
        "has_contradictions": len(contradictions) > 0,
        "meets_all_criteria": diversity_report["meets_diversity"] and len(contradictions) == 0,
    }
