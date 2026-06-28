# Implementation Plan: Free Deep-Research Fact-Checking Pipeline

## Overview
Transform the current pipeline into a production-ready, free deep-research system with:
- **Inline citations** in verdict explanations
- **Exact quote extraction** from sources
- **Free search providers** (no API keys required)
- **Multi-step ReAct research agent** with LangGraph
- **Source diversity & contradiction detection**
- **Enhanced TUI** with research visualization
- **Enhanced FastAPI** with streaming & citation endpoints

---

## Phase 1: Core Data Model Extensions (Day 1)

### Files to Modify
1. `src/fact_checker/models.py` - Add Citation model, extend EvidenceItem, extend VerdictResult
2. `src/fact_checker/skills/verdict_skills.py` - Update format_verdict_report for citations

### Changes
```python
# models.py - New Citation model
class Citation(BaseModel):
    evidence_id: UUID
    quote: str                    # Exact quote used
    claim_fragment: str           # Which part of claim this supports
    char_start: Optional[int] = None  # Position in explanation
    char_end: Optional[int] = None

# models.py - Extended EvidenceItem
class EvidenceItem(BaseModel):
    # ... existing fields ...
    quote_text: Optional[str] = None
    quote_context: Optional[str] = None  # ±200 chars around quote
    quote_offset: Optional[int] = None
    domain: str = ""
    published_date: Optional[datetime] = None
    author: Optional[str] = None
    source_type: Literal["factcheck", "news", "academic", "government", "wiki", "other"] = "other"

# models.py - Extended VerdictResult
class VerdictResult(BaseModel):
    # ... existing fields ...
    citations: List[Citation] = Field(default_factory=list)
    # explanation now contains [1], [2] markers referencing citations[]
```

---

## Phase 2: Free Search Providers (Day 1-2)

### New Files
1. `src/fact_checker/services/search_providers.py` - Abstract base + implementations

### Providers to Implement
| Provider | Free Tier | Method | Rate Limit |
|----------|-----------|--------|------------|
| **DuckDuckGo** | Unlimited | HTML scrape | Respectful delays |
| **Brave Search** | 2000/mo | REST API | 1 req/sec |
| **Semantic Scholar** | Unlimited | REST API | 100 req/5min |
| **Arxiv** | Unlimited | REST API | 3 sec delay |
| **PubMed/NCBI** | Unlimited | E-utilities | 3 req/sec |
| **Wikipedia** | Unlimited | REST API | ✅ Already done |
| **Crossref** | Unlimited | REST API | Polite |
| **GovInfo** | Unlimited | REST API | US gov docs |

### SearchProvider Interface
```python
class SearchProvider(ABC):
    name: str
    source_type: Literal["news", "academic", "government", "wiki", "general"]
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        pass
    
    @abstractmethod
    async def fetch_full(self, url: str) -> Optional[FullContent]:
        pass

class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    domain: str
    published_date: Optional[datetime]
    score: float  # Provider's relevance score

class FullContent(BaseModel):
    url: str
    title: str
    text: str
    quotes: List[Quote]  # Exact passages with offsets
    metadata: Dict
```

---

## Phase 3: Quote Extraction & Enhanced Scraping (Day 2)

### Files to Modify
1. `src/fact_checker/services/web_scraper.py` - Add quote extraction
2. `src/fact_checker/services/search_providers.py` - Integrate with providers

### Quote Extraction Logic
```python
def extract_quotes(text: str, claim: Claim, max_quotes: int = 3, 
                   context_chars: int = 200) -> List[Quote]:
    """Find claim-relevant passages in source text."""
    # 1. Split into sentences
    # 2. Score each sentence for relevance to claim (keyword overlap, entities)
    # 3. Return top-k with context windows
    # 4. Include character offsets for citation mapping
```

---

## Phase 4: ReAct Multi-Step DeepResearchAgent with LangGraph (Day 2-3)

### New Files
1. `src/fact_checker/agents/deep_research_agent_v2.py` - LangGraph-based agent

### LangGraph Workflow
```
┌─────────────┐
│  Decompose  │ → Sub-questions for claim
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│   Search    │ ──→ │   Scrape    │ (parallel per sub-q)
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────┐
│      Extract Quotes         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Cluster by Stance         │ (Supporting/Refuting/Neutral)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Check Diversity Gates     │ (domains, types, freshness)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Synthesize with Citations │
└─────────────────────────────┘
```

### State Schema
```python
class ResearchState(TypedDict):
    claim: Claim
    sub_questions: List[str]
    search_results: List[SearchResult]
    scraped_content: List[FullContent]
    quotes: List[Quote]
    evidence_clusters: Dict[str, List[Quote]]  # stance -> quotes
    diversity_report: DiversityReport
    final_evidence: List[EvidenceItem]
    citations: List[Citation]
```

---

## Phase 5: Verdict Agent with Citation Parsing (Day 3)

### Files to Modify
1. `src/fact_checker/prompts/verdict_draft.md` - Require [doc_id] citations
2. `src/fact_checker/agents/verdict_agent.py` - Parse citations, build Citation objects

### New Prompt Requirements
```
OUTPUT FORMAT:
{
  "verdict": "...",
  "confidence": 0.85,
  "explanation": "The claim is supported by multiple sources [1][2]. However, source [3] notes important context...",
  "requires_human_review": false,
  "citations": [
    {"evidence_id": "...", "quote": "exact text", "claim_fragment": "claim part"},
    ...
  ]
}
```

---

## Phase 6: Source Diversity & Contradiction Detection (Day 3-4)

### New File
1. `src/fact_checker/skills/evidence_skills.py` - Add diversity/contradiction functions

### Diversity Gates
```python
def enforce_source_diversity(evidence: List[EvidenceItem], 
                              min_domains: int = 3,
                              min_types: int = 2,
                              max_same_domain: int = 2) -> DiversityReport:
    """Ensure evidence meets diversity thresholds."""
    
def detect_contradictions(evidence: List[EvidenceItem]) -> List[Contradiction]:
    """Use LLM to classify stance of each evidence item, find conflicts."""
    
def temporal_freshness_score(evidence: List[EvidenceItem], 
                              half_life_days: int = 365) -> float:
    """Weight recent sources higher for time-sensitive claims."""
```

---

## Phase 7: TUI Enhancement (Day 4)

### Files to Modify
1. `src/fact_checker/tui.py` - Add research visualization, citation browser

### New TUI Screens
| Screen | Features |
|--------|----------|
| **Job List** | Same + status badges |
| **Job Detail** | Pipeline stage progress bar |
| **Claim View** | Claim + verdict + expandable citations |
| **Evidence Browser** | Filterable by stance, domain, type |
| **Quote Viewer** | Exact quote with source context |
| **Research Graph** | Sub-questions → sources → quotes flow |
| **Diversity Report** | Domain/type/temporal breakdown |

---

## Phase 8: FastAPI Enhancement (Day 4-5)

### Files to Modify
1. `src/fact_checker/api.py` - Add citation endpoints, streaming

### New Endpoints
| Endpoint | Purpose |
|----------|---------|
| `GET /jobs/{job_id}/claims/{claim_id}/evidence` | Paginated evidence with quotes |
| `GET /jobs/{job_id}/claims/{claim_id}/citations` | Structured citations for UI |
| `GET /jobs/{job_id}/research-graph` | Research flow visualization data |
| `GET /jobs/{job_id}/diversity-report` | Source diversity metrics |
| `WS /jobs/{job_id}/stream` | Real-time pipeline progress |
| `POST /search` | Direct search with free providers |

### Streaming Response
```python
@app.post("/submit", response_class=StreamingResponse)
async def submit_stream(request: SubmitRequest):
    """Stream pipeline progress as SSE."""
    async def event_generator():
        async for update in run_pipeline_streaming(...):
            yield f"data: {json.dumps(update)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## Phase 9: Configuration & Dependencies (Day 5)

### Files to Modify
1. `pyproject.toml` - Add new dependencies
2. `src/fact_checker/config.py` - Add search provider config

### New Dependencies
```toml
[project.optional-dependencies]
deep-research = [
    "langgraph>=0.2.0",           # Multi-step agents
    "duckduckgo-search>=5.0",     # Free DDG search
    "brave-search>=0.1",          # Brave API (optional)
    "semantic-scholar>=0.3",      # Semantic Scholar
    "arxiv>=2.0",                 # Arxiv API
    "biopython>=1.84",            # PubMed/NCBI
    "httpx>=0.27",                # Already in deps
    "lxml>=5.0",                  # HTML parsing
]
```

---

## Testing Strategy

### Unit Tests (New)
| Test | Target |
|------|--------|
| `test_citation_parsing.py` | VerdictAgent citation extraction |
| `test_quote_extraction.py` | Web scraper quote finding |
| `test_search_providers.py` | Each provider's search/fetch |
| `test_diversity_gates.py` | Source diversity enforcement |
| `test_contradiction_detection.py` | Stance classification |
| `test_research_graph.py` | LangGraph workflow |

### Integration Tests
| Test | Target |
|------|--------|
| `test_pipeline_citations.py` | Full pipeline with citations |
| `test_free_search.py` | End-to-end with free providers |
| `test_tui_citation_browser.py` | TUI evidence navigation |

---

## Rollout Order

1. **Models & Skills** (no external deps) → Run tests
2. **Search Providers** (free, no keys) → Run tests  
3. **Quote Extraction** → Run tests
4. **DeepResearchAgent v2** (LangGraph) → Run tests
5. **Verdict Agent + Prompts** → Run tests
6. **Diversity/Contradiction** → Run tests
7. **TUI Enhancements** → Manual test
8. **FastAPI Enhancements** → Run tests
9. **Full Integration Test** → All pass

---

## File Inventory Summary

### New Files (8)
- `src/fact_checker/services/search_providers.py`
- `src/fact_checker/agents/deep_research_agent_v2.py`
- `src/fact_checker/agents/research_workflow.py` (LangGraph)
- `tests/test_citation_parsing.py`
- `tests/test_quote_extraction.py`
- `tests/test_search_providers.py`
- `tests/test_diversity_gates.py`
- `tests/test_research_graph.py`

### Modified Files (12)
- `src/fact_checker/models.py`
- `src/fact_checker/skills/verdict_skills.py`
- `src/fact_checker/skills/evidence_skills.py`
- `src/fact_checker/services/web_scraper.py`
- `src/fact_checker/prompts/verdict_draft.md`
- `src/fact_checker/agents/verdict_agent.py`
- `src/fact_checker/harness.py` (wire new agent)
- `src/fact_checker/tui.py`
- `src/fact_checker/api.py`
- `pyproject.toml`
- `src/fact_checker/config.py`
- `src/fact_checker/config.py` (legacy settings.py)

---

## Success Criteria

- [ ] All 28 existing tests still pass
- [ ] 15+ new tests pass
- [ ] Pipeline runs end-to-end with **zero paid API keys**
- [ ] Verdict explanations contain **inline citations [1][2]**
- [ ] Each citation links to **exact quote** from source
- [ ] Source diversity ≥3 domains, ≥2 types enforced
- [ ] Contradictions detected and flagged for review
- [ ] TUI shows research flow and evidence browser
- [ ] FastAPI streams progress and serves citation data
- [ ] Latency < 5 min for 10-min video (with free providers)