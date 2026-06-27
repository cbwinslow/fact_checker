"""research_skills.py - Reusable skill functions for multi-hop research planning.

File: src/fact_checker/skills/research_skills.py

Provides stateless utility functions used by the DeepResearchAgent to plan
multi-hop research strategies, generate adversarial counter-queries, analyse
evidence gaps, and synthesise research briefs into a compact handoff summary
for the VerdictDraftAgent.

All functions are pure (no I/O, no LLM calls) and safe to unit-test
without external dependencies.
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Topics that require extra research caution and mandatory human review
_SENSITIVE_TOPICS: tuple[str, ...] = (
    "vaccine", "covid", "cancer", "medication", "drug", "suicide",
    "election", "vote", "ballot", "president", "congress",
    "climate change", "global warming",
    "abortion", "gun control", "immigration",
)

# Evidence strength thresholds
_STRONG_EVIDENCE_THRESHOLD = 0.75
_WEAK_EVIDENCE_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def plan_research_queries(
    claim_text: str,
    claim_type: str = "unknown",
    existing_queries: Optional[List[str]] = None,
) -> List[str]:
    """Plan a multi-hop sequence of research queries for a claim.

    Generates a prioritised query plan that goes beyond simple fact-check
    searches to include source-verification, context-widening, and
    adversarial angle queries.

    Query plan structure:
    1. Primary direct query (exact claim language).
    2. Fact-check database query ("site:politifact.com OR site:snopes.com").
    3. Wikipedia entity lookup for key named entities.
    4. Statistical source query (for statistical claims).
    5. Counter-narrative query (see ``generate_counter_queries``).

    Args:
        claim_text:       Normalised claim string.
        claim_type:       Semantic type from ``classify_claim_type``.
        existing_queries: Queries already used; new queries will not duplicate.

    Returns:
        Ordered list of research query strings (max 6).
    """
    existing = set(existing_queries or [])
    queries: list[str] = []

    summary = claim_text[:100].rstrip(",;.")

    # 1. Direct primary query
    q1 = summary
    if q1 not in existing:
        queries.append(q1)
        existing.add(q1)

    # 2. Fact-check site-search (for web search engines that support site:)
    q2 = f"{summary} site:politifact.com OR site:snopes.com OR site:factcheck.org"
    if q2 not in existing:
        queries.append(q2)
        existing.add(q2)

    # 3. Wikipedia lookup for named entities in the claim
    entities = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}\b", claim_text)
    if entities:
        wiki_query = f"Wikipedia {entities[0]}"
        if wiki_query not in existing:
            queries.append(wiki_query)
            existing.add(wiki_query)

    # 4. Type-specific queries
    if claim_type == "statistical":
        q_stat = f"{summary} official statistics government data"
        if q_stat not in existing:
            queries.append(q_stat)
            existing.add(q_stat)

    elif claim_type == "historical":
        q_hist = f"{summary} primary source history"
        if q_hist not in existing:
            queries.append(q_hist)
            existing.add(q_hist)

    elif claim_type == "causal":
        q_causal = f"{summary} peer reviewed study evidence"
        if q_causal not in existing:
            queries.append(q_causal)
            existing.add(q_causal)

    # 5. Counter-narrative query
    counter = generate_counter_queries(claim_text)
    if counter and counter[0] not in existing:
        queries.append(counter[0])
        existing.add(counter[0])

    return queries[:6]


def generate_counter_queries(claim_text: str) -> List[str]:
    """Generate adversarial counter-queries to surface contradicting evidence.

    Counter-queries are designed to actively seek evidence that REFUTES or
    COMPLICATES the claim.  This combats confirmation bias in the research
    pipeline by ensuring the agent looks for evidence on both sides.

    Strategy:
    1. "[claim] false" - direct refutation search.
    2. "[claim] debunked" - fact-check debunking search.
    3. "[claim] misleading context" - context-gap search.
    4. "against [core subject]" - opposition search.

    Args:
        claim_text: Normalised claim string.

    Returns:
        List of 2-4 counter-query strings.
    """
    summary = claim_text[:100].rstrip(",;.")
    queries = [
        f"{summary} false",
        f"{summary} debunked misleading",
        f"{summary} context missing",
    ]

    # Extract the main subject for an opposition search
    # Take first noun phrase (capitalised sequence or first 4 words)
    entities = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}\b", claim_text)
    if entities:
        queries.append(f"criticism of {entities[0]}")
    else:
        words = claim_text.split()[:4]
        queries.append(f"against {' '.join(words)}")

    return queries


def analyse_evidence_gaps(evidence_items: List[dict], claim_text: str) -> Dict[str, object]:
    """Identify weaknesses and gaps in the collected evidence set.

    Evaluates the evidence list and produces a structured gap analysis
    report that tells the DeepResearchAgent whether additional research
    rounds are warranted.

    Gap signals detected:
    - No evidence at all.
    - All evidence is low credibility (Tier 3 or 4 only).
    - No fact-check sources present.
    - Evidence is contradictory (mix of supporting/refuting).
    - Evidence is stale (all items older than 2 years, if dates available).
    - Total evidence count below minimum threshold.

    Args:
        evidence_items: List of evidence dicts with at least ``source_url``,
                        ``snippet``, and ``relevance_score`` keys.
        claim_text:     The claim being researched.

    Returns:
        Dict with keys:
        - ``has_gaps`` (bool): True if significant gaps were detected.
        - ``gap_reasons`` (list[str]): Human-readable gap descriptions.
        - ``needs_more_research`` (bool): True if another research round is advised.
        - ``gap_severity`` (str): "critical" | "moderate" | "minor" | "none".
    """
    gap_reasons: list[str] = []

    if not evidence_items:
        return {
            "has_gaps": True,
            "gap_reasons": ["No evidence was retrieved."],
            "needs_more_research": True,
            "gap_severity": "critical",
        }

    # Check for fact-check sources
    from .evidence_skills import is_factcheck_domain, score_source_credibility
    has_factcheck = any(is_factcheck_domain(e.get("source_url", "")) for e in evidence_items)
    if not has_factcheck:
        gap_reasons.append("No established fact-checking source found.")

    # Check credibility spread
    tiers = [score_source_credibility(e.get("source_url", ""))[0] for e in evidence_items]
    if all(t >= 3 for t in tiers):
        gap_reasons.append("All evidence sources are Tier 3 or lower credibility.")

    # Check for contradictions (simple keyword heuristic)
    supporting = sum(1 for e in evidence_items if e.get("relevance_score", 0) > 0.5)
    low_relevance = sum(1 for e in evidence_items if e.get("relevance_score", 0) < 0.25)
    if low_relevance > supporting:
        gap_reasons.append("Most retrieved evidence has low relevance to the claim.")

    # Check minimum evidence count
    if len(evidence_items) < 3:
        gap_reasons.append(f"Only {len(evidence_items)} evidence item(s) found; ideally 3 or more.")

    has_gaps = len(gap_reasons) > 0
    needs_more = len(gap_reasons) >= 2 or not has_factcheck

    if len(gap_reasons) == 0:
        severity = "none"
    elif len(gap_reasons) == 1:
        severity = "minor"
    elif len(gap_reasons) == 2:
        severity = "moderate"
    else:
        severity = "critical"

    return {
        "has_gaps": has_gaps,
        "gap_reasons": gap_reasons,
        "needs_more_research": needs_more,
        "gap_severity": severity,
    }


def summarise_research_brief(
    claim_text: str,
    evidence_items: List[dict],
    gap_analysis: dict,
) -> str:
    """Produce a concise plain-English research handoff summary.

    Generates a short (3-5 sentence) summary of the research findings
    suitable for inclusion in the handoff payload to the VerdictDraftAgent.
    This is NOT an LLM call — it is a deterministic template-based summary
    used as a fallback or augmentation.

    Args:
        claim_text:     The claim being researched.
        evidence_items: Collected evidence dicts.
        gap_analysis:   Output of ``analyse_evidence_gaps``.

    Returns:
        Plain-English research summary string.
    """
    n_evidence = len(evidence_items)
    from .evidence_skills import is_factcheck_domain
    n_factcheck = sum(1 for e in evidence_items if is_factcheck_domain(e.get("source_url", "")))
    high_rel = [e for e in evidence_items if e.get("relevance_score", 0) >= 0.5]

    severity = gap_analysis.get("gap_severity", "none")
    gap_reasons = gap_analysis.get("gap_reasons", [])

    lines = [
        f"Research retrieved {n_evidence} evidence item(s) for the claim: '{claim_text[:80]}...'",
        f"{n_factcheck} item(s) came from established fact-checking organisations.",
        f"{len(high_rel)} item(s) had high relevance scores (>= 0.50).",
    ]

    if severity in ("moderate", "critical"):
        lines.append(f"Evidence gaps detected ({severity}): {'; '.join(gap_reasons[:2])}.")
        lines.append("Additional research rounds are recommended before rendering a verdict.")
    else:
        lines.append("Evidence coverage appears sufficient for an initial verdict.")

    return " ".join(lines)
