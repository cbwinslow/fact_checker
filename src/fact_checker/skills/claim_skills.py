"""claim_skills.py - Reusable skill functions for claim processing.

File: src/fact_checker/skills/claim_skills.py

Provides stateless utility functions used by the ClaimExtractionAgent and
the harness pipeline to normalise, deduplicate, prioritise, and classify
extracted claims before they are passed to the evidence-retrieval stage.

All functions are pure (no I/O, no LLM calls) so they can be unit-tested
without any external dependencies.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ClaimType = Literal[
    "statistical",
    "historical",
    "causal",
    "attributional",
    "definitional",
    "predictive",
    "unverifiable",
    "unknown",
]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def normalise_claim_text(text: str) -> str:
    """Normalise raw claim text for consistent downstream processing.

    Steps applied:
    1. Unicode NFC normalisation (collapse ligatures, diacritics).
    2. Strip leading/trailing whitespace.
    3. Collapse internal runs of whitespace to a single space.
    4. Strip common LLM artefacts: surrounding quotes, asterisks, dashes.
    5. Ensure the claim ends with a full stop.

    Args:
        text: Raw claim string extracted by the LLM.

    Returns:
        Cleaned, normalised claim string.
    """
    # NFC normalisation
    text = unicodedata.normalize("NFC", text)
    # Strip surrounding whitespace
    text = text.strip()
    # Collapse internal whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip wrapping quotes (single, double, curly)
    text = re.sub(r'^["\u201c\u2018\']+|["\u201d\u2019\']+$', "", text).strip()
    # Strip leading dashes or asterisks (markdown list artefacts)
    text = re.sub(r"^[-*\u2022]\s*", "", text).strip()
    # Ensure terminal full stop
    if text and text[-1] not in ".")"!?":
        text += "."
    return text


def deduplicate_claims(claims: List[dict], similarity_threshold: float = 0.82) -> List[dict]:
    """Remove near-duplicate claims using fuzzy string similarity.

    Two claims are considered duplicates if their normalised texts exceed
    `similarity_threshold` under SequenceMatcher ratio scoring.  When a
    duplicate pair is found the claim with the higher confidence score is
    retained.

    Args:
        claims: List of claim dicts, each with at least ``text`` and
                ``confidence`` keys.
        similarity_threshold: Ratio threshold (0.0-1.0) above which two
                              claims are treated as duplicates.  Default 0.82.

    Returns:
        Deduplicated list of claim dicts, preserving original order for
        the highest-confidence representative of each cluster.
    """
    if not claims:
        return []

    normalised = [normalise_claim_text(c.get("text", "")) for c in claims]
    kept_indices: list[int] = []

    for i, norm_i in enumerate(normalised):
        is_duplicate = False
        for j in kept_indices:
            ratio = SequenceMatcher(None, norm_i, normalised[j]).ratio()
            if ratio >= similarity_threshold:
                # Keep the one with higher confidence
                if claims[i].get("confidence", 0.0) > claims[j].get("confidence", 0.0):
                    kept_indices[kept_indices.index(j)] = i
                is_duplicate = True
                break
        if not is_duplicate:
            kept_indices.append(i)

    return [claims[i] for i in kept_indices]


def score_claim_priority(claim: dict) -> float:
    """Compute a priority score (0.0-1.0) for processing order.

    Higher-priority claims are processed first in the evidence-retrieval
    queue.  Priority is derived from:
    - LLM confidence in claim checkability (weight 0.5)
    - Claim specificity proxy: presence of numbers, names, dates (weight 0.3)
    - Whether the claim is flagged checkable (weight 0.2)

    Args:
        claim: Claim dict with ``text``, ``confidence``, and
               ``is_checkable`` keys.

    Returns:
        Float priority score between 0.0 and 1.0.
    """
    text: str = claim.get("text", "")
    confidence: float = float(claim.get("confidence", 0.5))
    is_checkable: bool = bool(claim.get("is_checkable", True))

    # Specificity proxy: count numeric tokens, capitalised words, and years
    numeric_tokens = len(re.findall(r"\b\d+\.?\d*\b", text))
    proper_nouns = len(re.findall(r"\b[A-Z][a-z]+\b", text))
    years = len(re.findall(r"\b(19|20)\d{2}\b", text))
    specificity = min(1.0, (numeric_tokens * 0.15 + proper_nouns * 0.08 + years * 0.2))

    checkable_bonus = 0.2 if is_checkable else 0.0

    priority = (confidence * 0.5) + (specificity * 0.3) + checkable_bonus
    return round(min(1.0, priority), 4)


def classify_claim_type(text: str) -> ClaimType:
    """Heuristically classify a claim into one of seven semantic types.

    This is a fast, regex-based classifier used to route claims to the
    appropriate research strategy.  It is intentionally lightweight — the
    DeepResearchAgent performs a more nuanced LLM-based classification.

    Classification rules (first match wins):
    - ``statistical``: contains numbers, percentages, or ranked superlatives.
    - ``historical``: mentions a past year or past-tense event.
    - ``causal``:     contains causal language ("caused", "led to", "resulted").
    - ``attributional``: attributes a quote or action to a named entity.
    - ``definitional``: uses "is", "are", "means", "defined as".
    - ``predictive``:   forward-looking language ("will", "projected", "forecast").
    - ``unverifiable``: purely subjective or opinion language.
    - ``unknown``:      none of the above patterns match.

    Args:
        text: Claim string to classify.

    Returns:
        One of the ``ClaimType`` literal values.
    """
    t = text.lower()

    if re.search(r"\d+%|\d+ percent|\d+ million|\d+ billion|\blargest\b|\bmost\b|\bhighest\b|\blowest\b", t):
        return "statistical"
    if re.search(r"\b(19|20)\d{2}\b|\b(was|were|had|became|signed|passed|launched|won|lost)\b", t):
        return "historical"
    if re.search(r"\b(caused?|led to|resulted? in|due to|because of|triggered?)\b", t):
        return "causal"
    if re.search(r"\b(said|stated|claimed|announced|tweeted|wrote|accused|denied)\b", t):
        return "attributional"
    if re.search(r"\b(is|are|means|defined as|refers? to|known as)\b", t):
        return "definitional"
    if re.search(r"\b(will|would|projected?|forecast|expected to|predicted?|estimated? to)\b", t):
        return "predictive"
    if re.search(r"\b(believe|think|feel|opinion|arguably|seems?|appears? to)\b", t):
        return "unverifiable"
    return "unknown"
