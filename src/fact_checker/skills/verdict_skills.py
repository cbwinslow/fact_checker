"""verdict_skills.py - Reusable skill functions for verdict processing.

File: src/fact_checker/skills/verdict_skills.py

Provides stateless utility functions used by the VerdictDraftAgent and
the harness pipeline to aggregate multiple verdicts, calibrate confidence
scores, route low-confidence or sensitive verdicts for human review, and
format final verdict reports for API and UI consumers.

All functions are pure (no I/O, no LLM calls) and safe to unit-test
without external dependencies.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Literal

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

VerdictLabel = Literal[
    "supported",
    "refuted",
    "misleading",
    "insufficient_evidence",
    "unverifiable",
]

# Human-review triggers
_SENSITIVE_KEYWORDS: tuple[str, ...] = (
    "vaccine", "covid", "election", "abortion", "gun", "suicide",
    "cancer", "medication", "immigration", "climate", "president",
)
_MIN_CONFIDENCE_FOR_AUTO = 0.60


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def aggregate_verdicts(verdicts: List[dict]) -> dict:
    """Aggregate a list of per-claim verdict dicts into a job-level summary.

    Computes:
    - Verdict distribution (count per label).
    - Mean and minimum confidence across all verdicts.
    - Overall job verdict: the most frequent label, with ties broken by
      severity (refuted > misleading > insufficient_evidence > supported
      > unverifiable).
    - Whether any verdict requires human review.

    Args:
        verdicts: List of verdict dicts, each with at least ``verdict``
                  (str), ``confidence`` (float), and
                  ``requires_human_review`` (bool) keys.

    Returns:
        Aggregated summary dict with keys:
        ``total_claims``, ``verdict_distribution``, ``mean_confidence``,
        ``min_confidence``, ``overall_verdict``, ``requires_human_review``,
        ``human_review_count``.
    """
    if not verdicts:
        return {
            "total_claims": 0,
            "verdict_distribution": {},
            "mean_confidence": 0.0,
            "min_confidence": 0.0,
            "overall_verdict": "insufficient_evidence",
            "requires_human_review": False,
            "human_review_count": 0,
        }

    labels = [v.get("verdict", "unverifiable") for v in verdicts]
    confidences = [float(v.get("confidence", 0.0)) for v in verdicts]
    review_flags = [bool(v.get("requires_human_review", False)) for v in verdicts]

    distribution = dict(Counter(labels))
    mean_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)

    # Severity-weighted tie-breaking order
    severity_order = [
        "refuted", "misleading", "insufficient_evidence", "supported", "unverifiable"
    ]
    most_common = Counter(labels).most_common()
    top_count = most_common[0][1]
    candidates = [label for label, count in most_common if count == top_count]
    # Pick the most severe among tied candidates
    overall = next(
        (s for s in severity_order if s in candidates),
        candidates[0],
    )

    human_review_count = sum(review_flags)

    return {
        "total_claims": len(verdicts),
        "verdict_distribution": distribution,
        "mean_confidence": round(mean_conf, 4),
        "min_confidence": round(min_conf, 4),
        "overall_verdict": overall,
        "requires_human_review": any(review_flags),
        "human_review_count": human_review_count,
    }


def calibrate_confidence(
    raw_confidence: float,
    evidence_count: int,
    has_factcheck_source: bool,
    contradictions_found: bool,
) -> float:
    """Calibrate a raw LLM-reported confidence score using evidence signals.

    The LLM's self-reported confidence is adjusted based on objective
    evidence quality signals:
    - Boost for fact-check sources (+0.05).
    - Penalty for contradictions in evidence (-0.10).
    - Penalty for thin evidence (< 2 items: -0.10, < 4 items: -0.05).
    - Floor at 0.05, ceiling at 0.95 (never fully certain or uncertain).

    Args:
        raw_confidence:      LLM-reported confidence (0.0-1.0).
        evidence_count:      Number of evidence items used in the verdict.
        has_factcheck_source: True if at least one Tier-1 fact-check source
                             was found.
        contradictions_found: True if the research found contradicting sources.

    Returns:
        Calibrated confidence float between 0.05 and 0.95.
    """
    conf = float(raw_confidence)

    # Boost for authoritative source
    if has_factcheck_source:
        conf += 0.05

    # Penalty for contradictions
    if contradictions_found:
        conf -= 0.10

    # Penalty for thin evidence
    if evidence_count < 2:
        conf -= 0.10
    elif evidence_count < 4:
        conf -= 0.05

    # Clamp
    return round(max(0.05, min(0.95, conf)), 4)


def route_for_human_review(
    verdict: str,
    confidence: float,
    claim_text: str,
    contradictions_found: bool = False,
) -> tuple[bool, str]:
    """Determine whether a verdict should be escalated for human review.

    Returns a (should_review, reason) tuple.  Escalation is triggered by:
    - Confidence below the auto-publish threshold (0.60).
    - Unresolved contradictions in the evidence.
    - The claim contains a sensitive topic keyword.
    - Verdict is ``misleading`` (nuanced — always warrants human check).

    Args:
        verdict:              Verdict label string.
        confidence:           Calibrated confidence score.
        claim_text:           The original claim text (for keyword scanning).
        contradictions_found: Whether contradictions were found in research.

    Returns:
        Tuple of (bool: should escalate, str: reason or empty string).
    """
    reasons: list[str] = []

    if confidence < _MIN_CONFIDENCE_FOR_AUTO:
        reasons.append(f"Low confidence ({confidence:.2f} < {_MIN_CONFIDENCE_FOR_AUTO}).")

    if contradictions_found:
        reasons.append("Unresolved contradictions found in evidence.")

    if verdict == "misleading":
        reasons.append("Verdict is 'misleading' — requires editorial judgment.")

    # Sensitive topic scan
    claim_lower = claim_text.lower()
    matched_topics = [kw for kw in _SENSITIVE_KEYWORDS if kw in claim_lower]
    if matched_topics:
        reasons.append(f"Sensitive topic(s) detected: {', '.join(matched_topics[:3])}.")

    if reasons:
        return True, " ".join(reasons)
    return False, ""


def format_verdict_report(verdict_result: dict, claim_text: str) -> str:
    """Format a verdict result dict into a human-readable plain-text report.

    Produces a short structured report string suitable for display in the
    TUI, email notifications, or API text fields.  This is NOT markdown —
    it is plain text with simple ASCII structure.

    Args:
        verdict_result: Verdict dict with keys ``verdict``, ``confidence``,
                        ``explanation``, ``requires_human_review``.
        claim_text:     The original claim string.

    Returns:
        Multi-line plain-text report string.
    """
    verdict = verdict_result.get("verdict", "unverifiable").upper()
    confidence = float(verdict_result.get("confidence", 0.0))
    explanation = verdict_result.get("explanation", "No explanation provided.")
    needs_review = bool(verdict_result.get("requires_human_review", False))

    # Confidence bar (10 chars)
    filled = round(confidence * 10)
    bar = "[" + "#" * filled + "-" * (10 - filled) + "]"

    lines = [
        "=" * 60,
        f"VERDICT:     {verdict}",
        f"CONFIDENCE:  {bar} {confidence:.0%}",
        f"CLAIM:       {claim_text[:120]}",
        "-" * 60,
        f"EXPLANATION: {explanation}",
    ]

    if needs_review:
        lines += ["-" * 60, "*** FLAGGED FOR HUMAN REVIEW ***"]

    lines.append("=" * 60)
    return "\n".join(lines)
