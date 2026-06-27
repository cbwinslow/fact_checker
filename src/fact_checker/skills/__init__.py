"""skills - reusable, stateless helper functions consumed by fact-checker agents.

File: src/fact_checker/skills/__init__.py

Each sub-module in this package exposes a focused set of pure (or lightly
stateful) utility functions that any agent may import.  Keeping skill logic
here prevents code duplication across agents and makes individual capabilities
easy to unit-test in isolation.

Package layout
--------------
claim_skills    -- claim normalisation, deduplication, priority scoring
evidence_skills -- source credibility scoring, query generation, snippet ranking
image_skills    -- frame selection heuristics, OCR post-processing, object tagging
research_skills -- multi-hop query planning, counter-query generation, gap analysis
verdict_skills  -- verdict aggregation, confidence calibration, human-review routing
ingest_skills   -- media-type detection, chunking strategies, segment merging
"""

from .claim_skills import (
    normalise_claim_text,
    deduplicate_claims,
    score_claim_priority,
    classify_claim_type,
)
from .evidence_skills import (
    score_source_credibility,
    generate_search_queries,
    rank_evidence_snippets,
    is_factcheck_domain,
)
from .image_skills import (
    select_keyframes,
    postprocess_ocr_text,
    tag_objects_from_analysis,
    correlate_frames_to_transcript,
)
from .research_skills import (
    plan_research_queries,
    generate_counter_queries,
    analyse_evidence_gaps,
    summarise_research_brief,
)
from .verdict_skills import (
    aggregate_verdicts,
    calibrate_confidence,
    route_for_human_review,
    format_verdict_report,
)
from .ingest_skills import (
    detect_media_type,
    chunk_text_segments,
    merge_short_segments,
    estimate_processing_cost,
)

__all__ = [
    # claim
    "normalise_claim_text",
    "deduplicate_claims",
    "score_claim_priority",
    "classify_claim_type",
    # evidence
    "score_source_credibility",
    "generate_search_queries",
    "rank_evidence_snippets",
    "is_factcheck_domain",
    # image
    "select_keyframes",
    "postprocess_ocr_text",
    "tag_objects_from_analysis",
    "correlate_frames_to_transcript",
    # research
    "plan_research_queries",
    "generate_counter_queries",
    "analyse_evidence_gaps",
    "summarise_research_brief",
    # verdict
    "aggregate_verdicts",
    "calibrate_confidence",
    "route_for_human_review",
    "format_verdict_report",
    # ingest
    "detect_media_type",
    "chunk_text_segments",
    "merge_short_segments",
    "estimate_processing_cost",
]
