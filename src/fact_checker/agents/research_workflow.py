"""agents/research_workflow.py - LangGraph-based multi-step deep research workflow.

Implements a ReAct (Reasoning + Acting) pattern for iterative, evidence-based research:
1. Decompose claim into sub-questions
2. For each sub-question: search → fetch → extract quotes
3. Cluster evidence by stance (supporting/refuting/neutral)
4. Check diversity gates (domains, types, freshness)
5. Synthesize final evidence with citations

Uses LangGraph for stateful, interruptible workflow execution.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict
from uuid import UUID

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..models import Claim, EvidenceItem, ResearchResult
from ..services.search_providers import get_registry, SearchResult, Quote, enrich_search_results_with_quotes
from ..config import get_settings
from ..skills.evidence_skills import score_source_credibility, is_factcheck_domain
from ..skills.research_skills import (
    plan_research_queries,
    generate_counter_queries,
    analyse_evidence_gaps,
    summarise_research_brief,
)
from ..skills.claim_skills import classify_claim_type

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------


class ResearchState(TypedDict):
    """State for the research workflow."""
    # Input
    claim: Claim
    claim_text: str
    claim_type: str
    
    # Research planning
    sub_questions: List[str]
    search_queries: List[str]
    
    # Search & retrieval
    search_results: List[SearchResult]
    enriched_results: List[SearchResult]
    
    # Quote extraction
    quotes: List[Quote]
    
    # Evidence processing
    evidence_items: List[EvidenceItem]
    evidence_by_stance: Dict[str, List[EvidenceItem]]  # supporting/refuting/neutral
    
    # Quality gates
    diversity_report: Optional[Dict]
    gap_analysis: Optional[Dict]
    needs_more_research: bool
    research_iteration: int
    
    # Final output
    final_evidence: List[EvidenceItem]
    citations: List[Dict]
    research_brief: str


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------


async def decompose_claim(state: ResearchState) -> ResearchState:
    """Decompose the claim into researchable sub-questions."""
    claim = state["claim"]
    claim_text = state["claim_text"]
    
    log.info(f"[research_workflow] Decomposing claim: {claim_text[:80]}")
    
    # Use existing research skills to plan queries
    queries = plan_research_queries(claim_text, state["claim_type"])
    
    # Also add counter-queries for adversarial research
    counter_queries = generate_counter_queries(claim_text)
    queries.extend(counter_queries[:2])  # Add top 2 counter-queries
    
    # De-duplicate while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)
    
    # Limit total queries
    unique_queries = unique_queries[:8]
    
    log.info(f"[research_workflow] Generated {len(unique_queries)} research queries")
    
    return {
        **state,
        "search_queries": unique_queries,
        "research_iteration": state.get("research_iteration", 0) + 1,
    }


async def search_all_providers(state: ResearchState) -> ResearchState:
    """Execute searches across all free providers in parallel."""
    queries = state["search_queries"]
    registry = get_registry()
    
    log.info(f"[research_workflow] Searching across {len(queries)} queries")
    
    all_results: List[SearchResult] = []
    
    # Search each query across all providers
    for query in queries:
        results = await registry.search_all(
            query,
            max_results_per_provider=3,
            source_types=["factcheck", "academic", "government", "news", "wiki"],
        )
        all_results.extend(results)
    
    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        key = r.url.rstrip("/").lower()
        if key not in seen_urls:
            seen_urls.add(key)
            unique_results.append(r)
    
    # Sort by score
    unique_results.sort(key=lambda x: x.score, reverse=True)
    
    log.info(f"[research_workflow] Found {len(unique_results)} unique search results")
    
    return {**state, "search_results": unique_results}


async def enrich_and_extract_quotes(state: ResearchState) -> ResearchState:
    """Fetch full content for top results and extract relevant quotes."""
    registry = get_registry()
    claim_text = state["claim_text"]
    results = state["search_results"]
    
    # Limit to top 15 results for enrichment (cost control)
    top_results = results[:15]
    
    log.info(f"[research_workflow] Enriching top {len(top_results)} results")
    
    # Enrich with full content and quotes
    enriched = await enrich_search_results_with_quotes(top_results, claim_text, registry)
    
    # Extract all quotes
    all_quotes: List[Quote] = []
    for result in enriched:
        if "quotes" in result.raw_data:
            for q_data in result.raw_data["quotes"]:
                all_quotes.append(Quote(
                    text=q_data["text"],
                    context_before=q_data["context_before"],
                    context_after=q_data["context_after"],
                    offset=q_data["offset"],
                    relevance_score=q_data["relevance_score"],
                ))
    
    log.info(f"[research_workflow] Extracted {len(all_quotes)} quotes")
    
    return {**state, "enriched_results": enriched, "quotes": all_quotes}


async def build_evidence_items(state: ResearchState) -> ResearchState:
    """Convert enriched search results to EvidenceItem objects."""
    claim = state["claim"]
    enriched = state["enriched_results"]
    quotes_by_result: Dict[str, List[Quote]] = {}
    
    # Build quote lookup
    for result in enriched:
        if "quotes" in result.raw_data:
            quotes_by_result[result.url] = [
                Quote(
                    text=q["text"],
                    context_before=q["context_before"],
                    context_after=q["context_after"],
                    offset=q["offset"],
                    relevance_score=q["relevance_score"],
                )
                for q in result.raw_data["quotes"]
            ]
    
    evidence_items: List[EvidenceItem] = []
    
    for result in enriched:
        # Find best quote for this result
        best_quote = None
        if result.url in quotes_by_result and quotes_by_result[result.url]:
            best_quote = max(quotes_by_result[result.url], key=lambda q: q.relevance_score)
        
        # Get credibility score
        _, cred_score = score_source_credibility(result.url)
        
        evidence = EvidenceItem(
            claim_id=claim.id,
            source_url=result.url,
            title=result.title,
            snippet=result.snippet,
            relevance_score=result.score,
            credibility_score=cred_score,
            is_factcheck_source=is_factcheck_domain(result.url),
            domain=result.domain,
            published_date=result.published_date,
            author=result.author,
            source_type=result.source_type,
            quote_text=best_quote.text if best_quote else None,
            quote_context=best_quote.full_context if best_quote else None,
            quote_offset=best_quote.offset if best_quote else None,
        )
        evidence_items.append(evidence)
    
    log.info(f"[research_workflow] Built {len(evidence_items)} evidence items")
    
    return {**state, "evidence_items": evidence_items}


async def classify_stance(state: ResearchState) -> ResearchState:
    """Classify each evidence item's stance toward the claim using LLM.
    
    For now, use a heuristic based on relevance and source type.
    In production, this would call an LLM to classify stance.
    """
    from ..config import build_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage
    import json
    
    claim_text = state["claim_text"]
    evidence_items = state["evidence_items"]
    
    if not evidence_items:
        return {**state, "evidence_by_stance": {"supporting": [], "refuting": [], "neutral": []}}
    
    # Use a fast model for stance classification
    llm = build_chat_model(task="fast", temperature=0.0, max_tokens=1024)
    
    stance_prompt = """You are a fact-checking assistant. Classify the stance of each evidence snippet toward the claim.

STANCE OPTIONS:
- "supporting": Evidence directly supports or confirms the claim
- "refuting": Evidence directly contradicts or refutes the claim  
- "neutral": Evidence is relevant but neither clearly supports nor refutes

CLAIM: {claim}

EVIDENCE SNIPPETS:
{evidence_list}

Return ONLY a JSON array of stance classifications in the same order:
["supporting", "refuting", "neutral", ...]"""
    
    evidence_list = "\n".join(
        f"[{i}] {ev.snippet[:200]}... (source: {ev.domain})"
        for i, ev in enumerate(evidence_items)
    )
    
    messages = [
        SystemMessage(content="You are a precise stance classifier for fact-checking."),
        HumanMessage(content=stance_prompt.format(claim=claim_text, evidence_list=evidence_list)),
    ]
    
    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        stances = json.loads(content)
        
        # Group by stance
        by_stance = {"supporting": [], "refuting": [], "neutral": []}
        for ev, stance in zip(evidence_items, stances):
            if stance in by_stance:
                by_stance[stance].append(ev)
            else:
                by_stance["neutral"].append(ev)
        
        log.info(f"[research_workflow] Stance distribution: "
                 f"supporting={len(by_stance['supporting'])}, "
                 f"refuting={len(by_stance['refuting'])}, "
                 f"neutral={len(by_stance['neutral'])}")
        
        return {**state, "evidence_by_stance": by_stance}
    except Exception as exc:
        log.warning(f"[research_workflow] Stance classification failed: {exc}")
        # Fallback: heuristic based on relevance
        by_stance = {"supporting": [], "refuting": [], "neutral": []}
        for ev in evidence_items:
            if ev.relevance_score >= 0.7:
                by_stance["supporting"].append(ev)
            elif ev.relevance_score >= 0.4:
                by_stance["neutral"].append(ev)
            else:
                by_stance["refuting"].append(ev)
        return {**state, "evidence_by_stance": by_stance}


async def check_diversity_gates(state: ResearchState) -> ResearchState:
    """Enforce source diversity requirements."""
    evidence_items = state["evidence_items"]
    
    # Count unique domains
    domains = set(ev.domain for ev in evidence_items)
    
    # Count source types
    types = set(ev.source_type for ev in evidence_items)
    
    # Check for fact-check sources
    has_factcheck = any(ev.is_factcheck_source for ev in evidence_items)
    
    # Temporal freshness (for time-sensitive claims)
    now = datetime.now()
    fresh_count = 0
    for ev in evidence_items:
        if ev.published_date:
            age_days = (now - ev.published_date).days
            if age_days < 365:  # Less than 1 year
                fresh_count += 1
    
    diversity_report = {
        "unique_domains": len(domains),
        "domains": list(domains),
        "source_types": list(types),
        "has_factcheck_source": has_factcheck,
        "fresh_sources": fresh_count,
        "total_sources": len(evidence_items),
        "meets_diversity": (
            len(domains) >= 3 and
            len(types) >= 2 and
            (has_factcheck or len(evidence_items) >= 5)
        ),
    }
    
    log.info(f"[research_workflow] Diversity report: {diversity_report}")
    
    return {**state, "diversity_report": diversity_report}


async def analyse_gaps(state: ResearchState) -> ResearchState:
    """Analyze evidence gaps and determine if more research is needed."""
    from ..skills.evidence_skills import check_diversity_and_contradictions
    
    claim_text = state["claim_text"]
    evidence_items = state["evidence_items"]
    diversity_report = state["diversity_report"]
    evidence_by_stance = state["evidence_by_stance"]
    
    # Use existing gap analysis
    gap_analysis = analyse_evidence_gaps(
        [{"source_url": ev.source_url, "snippet": ev.snippet, "relevance_score": ev.relevance_score} 
         for ev in evidence_items],
        claim_text,
    )
    
    # Check diversity and contradictions
    div_contra = check_diversity_and_contradictions(evidence_items)
    
    # Additional checks
    has_supporting = len(evidence_by_stance.get("supporting", [])) > 0
    has_refuting = len(evidence_by_stance.get("refuting", [])) > 0
    
    # Need more research if:
    needs_more = (
        gap_analysis["needs_more_research"] or
        not diversity_report["meets_diversity"] or
        div_contra["has_contradictions"] or
        (has_supporting and has_refuting and len(evidence_items) < 8)  # Contradiction needs more evidence
    )
    
    log.info(f"[research_workflow] Gap analysis: needs_more={needs_more}, "
             f"gaps={gap_analysis['gap_reasons']}, "
             f"contradictions={len(div_contra['contradictions'])}")
    
    return {
        **state,
        "gap_analysis": gap_analysis,
        "contradictions": div_contra["contradictions"],
        "needs_more_research": needs_more,
    }


async def maybe_iterate(state: ResearchState) -> str:
    """Decide whether to iterate or finalize."""
    iteration = state.get("research_iteration", 1)
    needs_more = state.get("needs_more_research", False)
    max_iterations = getattr(settings, "research_max_iterations", 2)
    
    if needs_more and iteration < max_iterations:
        log.info(f"[research_workflow] Iteration {iteration}: More research needed, continuing...")
        return "iterate"
    else:
        log.info(f"[research_workflow] Iteration {iteration}: Finalizing research")
        return "finalize"


async def finalize_research(state: ResearchState) -> ResearchState:
    """Build final ResearchResult with citations."""
    claim = state["claim"]
    evidence_items = state["evidence_items"]
    evidence_by_stance = state["evidence_by_stance"]
    gap_analysis = state["gap_analysis"]
    diversity_report = state["diversity_report"]
    
    # Build citations from evidence
    citations = []
    for i, ev in enumerate(evidence_items):
        if ev.quote_text:
            citations.append({
                "evidence_id": str(ev.id),
                "quote": ev.quote_text,
                "claim_fragment": claim.text[:100],  # Would be more precise in production
                "index": i + 1,
            })
    
    # Create research brief
    research_brief = summarise_research_brief(
        claim_text=claim.text,
        evidence_items=[{
            "source_url": ev.source_url,
            "snippet": ev.snippet,
            "relevance_score": ev.relevance_score,
        } for ev in evidence_items],
        gap_analysis=gap_analysis,
    )
    
    # Sort evidence by composite score
    sorted_evidence = sorted(
        evidence_items,
        key=lambda ev: ev.relevance_score * getattr(ev, "credibility_score", 0.5),
        reverse=True,
    )
    
    # Take top evidence
    final_evidence = sorted_evidence[:20]
    
    log.info(f"[research_workflow] Finalized with {len(final_evidence)} evidence items, "
             f"{len(citations)} citations")
    
    return {
        **state,
        "final_evidence": final_evidence,
        "citations": citations,
        "research_brief": research_brief,
    }


# ---------------------------------------------------------------------------
# Build LangGraph Workflow
# ---------------------------------------------------------------------------


def build_research_workflow() -> StateGraph:
    """Construct the research workflow graph."""
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("decompose", decompose_claim)
    workflow.add_node("search", search_all_providers)
    workflow.add_node("enrich", enrich_and_extract_quotes)
    workflow.add_node("build_evidence", build_evidence_items)
    workflow.add_node("classify_stance", classify_stance)
    workflow.add_node("diversity_check", check_diversity_gates)
    workflow.add_node("gap_analysis", analyse_gaps)
    workflow.add_node("finalize", finalize_research)
    
    # Define edges
    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "search")
    workflow.add_edge("search", "enrich")
    workflow.add_edge("enrich", "build_evidence")
    workflow.add_edge("build_evidence", "classify_stance")
    workflow.add_edge("classify_stance", "diversity_check")
    workflow.add_edge("diversity_check", "gap_analysis")
    
    # Conditional edge for iteration
    workflow.add_conditional_edges(
        "gap_analysis",
        maybe_iterate,
        {
            "iterate": "decompose",  # Loop back with new queries
            "finalize": "finalize",
        },
    )
    
    workflow.add_edge("finalize", END)
    
    return workflow


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Compile workflow with memory checkpointing
_research_workflow = build_research_workflow().compile(checkpointer=MemorySaver())


async def deep_research_langgraph(
    claims: List[Claim],
    context: Optional[Any] = None,
) -> List[ResearchResult]:
    """Run the LangGraph-based deep research workflow for multiple claims.
    
    Args:
        claims: List of Claim objects to research.
        context: Optional AnalysisContext (for VectorStore access, not yet used).
    
    Returns:
        List of ResearchResult objects.
    """
    checkable = [c for c in claims if c.is_checkable]
    if not checkable:
        log.info("[research_workflow] No checkable claims to research.")
        return []
    
    log.info(f"[research_workflow] Starting deep research for {len(checkable)} claims")
    
    results = []
    for claim in checkable:
        claim_type = classify_claim_type(claim.text)
        
        initial_state: ResearchState = {
            "claim": claim,
            "claim_text": claim.text,
            "claim_type": claim_type,
            "sub_questions": [],
            "search_queries": [],
            "search_results": [],
            "enriched_results": [],
            "quotes": [],
            "evidence_items": [],
            "evidence_by_stance": {"supporting": [], "refuting": [], "neutral": []},
            "diversity_report": None,
            "gap_analysis": None,
            "needs_more_research": False,
            "research_iteration": 0,
            "final_evidence": [],
            "citations": [],
            "research_brief": "",
        }
        
        # Run workflow with unique thread ID
        config = {"configurable": {"thread_id": f"research_{claim.id}"}}
        final_state = await _research_workflow.ainvoke(initial_state, config=config)
        
        # Build ResearchResult
        avg_credibility = (
            sum(getattr(ev, "credibility_score", 0.5) for ev in final_state["final_evidence"]) 
            / len(final_state["final_evidence"])
            if final_state["final_evidence"] else 0.0
        )
        
        research_result = ResearchResult(
            claim_id=claim.id,
            evidence=final_state["final_evidence"],
            context_snippets=[],  # Would include VectorStore snippets if context provided
            avg_credibility=avg_credibility,
            evidence_count=len(final_state["final_evidence"]),
            has_factcheck_source=any(ev.is_factcheck_source for ev in final_state["final_evidence"]),
        )
        results.append(research_result)
    
    log.info(f"[research_workflow] Completed research for {len(results)} claims")
    return results