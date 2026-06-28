# AGENTS.md — Agent Architecture & Pipeline Specification

> **Purpose**: Single source of truth for the fact_checker agent pipeline. Describes every agent, its inputs/outputs, model assignments, and how they compose into the end-to-end harness.

---

## 1. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FACT_CHECKER PIPELINE (6 Stages)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Stage 1: INGEST ──► Stage 2: ANALYZE_IMAGES ──► Stage 3: EMBED          │
│       │                    │                       │                       │
│       ▼                    ▼                       ▼                       │
│  MediaRouter          ImageAnalystAgent        Embedder + VectorStore     │
│  (file_router.py)     (image_analyst.py)       (embedder.py, vector_store)│
│       │                    │                       │                       │
│       └────────────────────┴───────────────────────┘                       │
│                                 │                                           │
│                                 ▼                                           │
│  Stage 4: EXTRACT_CLAIMS ──► Stage 5: DEEP_RESEARCH ──► Stage 6: VERDICT  │
│       │                    │                       │                       │
│       ▼                    ▼                       ▼                       │
│  ClaimExtractorAgent  DeepResearchAgent       VerdictDraftAgent           │
│  (claim_extractor.py) (deep_research_agent.py) (verdict_agent.py)         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Shared Context**: All stages read/write a single `AnalysisContext` packet (defined in `models.py`) that carries:
- `segments` — TranscriptSegment[]
- `images` — ImageAnalysis[]
- `chunks` — EmbeddedChunk[] + live `vector_store`
- `claims` — Claim[]
- `research_results` — ResearchResult[]
- `evidence` — EvidenceItem[]
- `verdicts` — VerdictResult[]

---

## 2. Agent Registry

| Agent | File | Task Slot | Model (Default) | Purpose |
|-------|------|-----------|-----------------|---------|
| **MediaRouter** | `services/file_router.py` | N/A (router) | — | Detects input type, routes to correct ingestor |
| **IngestService** | `services/ingest.py` | `multimodal` | `meta-llama/llama-4-maverick:free` | 3-layer transcript: yt-dlp → yt-transcript-api → Whisper |
| **AudioIngest** | `services/audio_ingest.py` | `multimodal` | `meta-llama/llama-4-maverick:free` | Audio-only Whisper ASR |
| **PDFIngest** | `services/pdf_ingest.py` | `fast` | `openai/gpt-oss-20b:free` | PDF text extraction |
| **WebScraper** | `services/web_scraper.py` | `fast` | `openai/gpt-oss-20b:free` | Article scraping + html_to_text |
| **ImageAnalystAgent** | `agents/image_analyst.py` | `multimodal` | `meta-llama/llama-4-maverick:free` | Vision analysis: objects, OCR, visible claims, manipulation risk |
| **Embedder** | `services/embedder.py` | `extraction` | `openai/gpt-oss-120b:free` | Text → vector embeddings for semantic retrieval |
| **ClaimExtractorAgent** | `agents/claim_extractor.py` | `extraction` | `openai/gpt-oss-120b:free` | Extract atomic checkable claims from transcript + image visible_claims |
| **DeepResearchAgent** | `agents/deep_research_agent.py` | `verification` | `nvidia/nemotron-3-ultra-253b:free` | Multi-hop research: semantic retrieval, Google FC, Serper, scraping, adversarial, Wikipedia |
| **EvidenceAgent** | `agents/evidence_agent.py` | `verification` | `nvidia/nemotron-3-ultra-253b:free` | Simpler evidence retrieval (Google FC + Serper only) |
| **VerdictDraftAgent** | `agents/verdict_agent.py` | `verification` | `nvidia/nemotron-3-ultra-253b:free` | Evidence-grounded verdict per claim |
| **Orchestrator** | `harness.py` | `orchestration` | `nvidia/nemotron-3-super-49b-v1:free` | Pipeline coordination, status updates, final summary |

---

## 3. Stage Specifications

### Stage 1: INGEST (`MediaRouter.route()`)
**Input**: `job_id`, `url?`, `local_path?`, `image_paths?`
**Output**: `(List[TranscriptSegment], IngestSource)`
**Context updates**: `ctx.segments`, `ctx.ingest_source`

Routes based on input:
- YouTube URL → `services.ingest.ingest()` (3-layer)
- Direct video/audio URL → download + Whisper
- Local video → `services.ingest.ingest(local_path=...)`
- Local audio → `services.audio_ingest.ingest_audio_file()`
- PDF (local/URL) → `services.pdf_ingest.ingest_pdf()`
- Web article → `services.web_scraper.scrape_article()`
- Text/MD/DOCX/HTML → `_text_to_segments()`
- Image-only → returns empty segments, `IngestSource.IMAGE`

### Stage 2: ANALYZE_IMAGES (`ImageAnalystAgent.analyse_images()`)
**Input**: `job_id`, `image_paths[]`, `source_type`, `frame_timestamps?`
**Output**: `List[ImageAnalysis]`
**Context updates**: `ctx.images`

For each image:
1. Extract EXIF metadata (`vision.read_image_metadata`)
2. Encode as base64 data URL (`vision.image_to_data_url`)
3. Call vision LLM with structured prompt (`prompts/image_analysis.md`)
4. Parse JSON → `ImageAnalysis` with:
   - `description`, `objects[]`, `text_in_image`, `visible_claims[]`
   - `manipulation_risk`, `manipulation_reason`, `context_notes`

### Stage 3: EMBED (`embedder.embed_segments()` + `VectorStore`)
**Input**: `job_id`, `ctx.segments`
**Output**: `List[EmbeddedChunk]`, live `VectorStore`
**Context updates**: `ctx.chunks`, `ctx.vector_store`

- Chunks transcript segments (overlap-aware)
- Embeds via `extraction` task model
- Stores in per-job ChromaDB collection

### Stage 4: EXTRACT_CLAIMS (`ClaimExtractorAgent.extract_claims()`)
**Input**: `job_id`, `ctx.segments`
**Output**: `List[Claim]`
**Context updates**: `ctx.claims` (transcript claims + promoted `visible_claims` from images)

- Builds full transcript text with timestamps
- Calls `extraction` model with `prompts/claim_extraction.md`
- Parses JSON array → `Claim` objects
- Promotes `ImageAnalysis.visible_claims` as Claims with `image_id` link

### Stage 5: DEEP_RESEARCH (`DeepResearchAgent.deep_research()`)
**Input**: `claims[]`, `ctx` (with vector_store)
**Output**: `List[ResearchResult]`
**Context updates**: `ctx.research_results`, `ctx.evidence` (flattened)

Per claim (concurrent, semaphore-limited):
1. **Semantic retrieval** — embed claim, query job VectorStore
2. **Google Fact Check API** — if key configured
3. **Serper web search** — primary evidence
4. **Full-page scrape** — top 3 URLs → enriched snippets
5. **Adversarial counter-query** — if evidence weak (`_is_weak_evidence`)
6. **Wikipedia lookup** — first named entity
7. **Score & deduplicate** — credibility tiers + composite relevance

### Stage 6: VERDICT (`VerdictDraftAgent.draft_verdicts()`)
**Input**: `claims[]`, `ctx.evidence` (grouped by claim_id)
**Output**: `List[VerdictResult]`
**Context updates**: `ctx.verdicts`

Per claim:
- Formats evidence as cited snippets
- Calls `verification` model with `prompts/verdict_draft.md`
- Parses JSON → `VerdictResult` with calibrated confidence
- Flags `requires_human_review` if confidence < 0.6 or sensitive topic

---

## 4. Model Assignments (Config-Driven)

All model selection is centralized in `config.py::MODEL_REGISTRY` and overrideable via `.env`:

| Task Slot | Default Model | Purpose |
|-----------|---------------|---------|
| `extraction` | `openai/gpt-oss-120b:free` | Structured JSON claim extraction |
| `verification` | `nvidia/nemotron-3-ultra-253b:free` | Deep reasoning, 1M context |
| `orchestration` | `nvidia/nemotron-3-super-49b-v1:free` | Pipeline synthesis |
| `multimodal` | `meta-llama/llama-4-maverick:free` | Vision (image_url content parts) |
| `tooluse` | `cohere/north-mini-code:free` | Structured DB writes, JSON schema |
| `fast` | `openai/gpt-oss-20b:free` | Quick subtasks, routing |

**Offline Mode**: If `OPENROUTER_API_KEY` unset → `MockChatModel` returns deterministic stub JSON for full pipeline exercise.

---

## 5. Data Flow Invariants

1. **AnalysisContext is the single source of truth** — never pass raw lists between stages.
2. **UUIDs are the only cross-stage keys** — `job_id`, `claim_id`, `image_id`, `segment_id`.
3. **Stages are idempotent where possible** — re-running a stage with same context produces same output.
4. **Errors are captured in context** — failed stages mark `job.status = FAILED` but preserve partial results.
5. **DB status updates between stages** — enables real-time polling via `/jobs/{job_id}`.

---

## 6. Adding a New Agent

1. Create `src/fact_checker/agents/<name>.py` with async entry point.
2. Add prompt to `src/fact_checker/prompts/<name>.md`.
3. Register in `config.py::MODEL_REGISTRY` if new model slot needed.
4. Add skill utilities to `src/fact_checker/skills/<domain>_skills.py` if reusable.
5. Wire into `harness.py::run_pipeline()` at correct stage.
6. Update `AnalysisContext` model if new artifact type introduced.
7. Add tests in `tests/test_agents.py`.

---

## 7. Key Integration Points

| Component | Entry Point | Called By |
|-----------|-------------|-----------|
| CLI | `cli.py::app` | `fact-checker submit/file/serve/tui` |
| API | `api.py::app` | `uvicorn fact_checker.api:app` |
| MCP | `mcp/fact_checker_mcp_server.py` | `python -m mcp.fact_checker_mcp_server` |
| Harness | `harness.py::run_pipeline()` | CLI, API, MCP, tests |
| DB | `db.py::init_db()`, `save_pipeline_result()` | API, harness (with session) |

---

## 8. Configuration Hierarchy

```
settings.py (BaseSettings)
    ├─ .env file (highest priority)
    ├─ Environment variables
    ├─ config.py::MODEL_REGISTRY defaults
    └─ Hardcoded fallbacks
```

Per-task model overrides: `MODEL_EXTRACTION`, `MODEL_VERIFICATION`, `MODEL_ORCHESTRATION`, `MODEL_MULTIMODAL`, `MODEL_TOOLUSE`, `MODEL_FAST`.

---

## 9. Testing Strategy

| Test File | Scope |
|-----------|-------|
| `tests/conftest.py` | Shared fixtures (mock segments, claims, evidence, FastAPI client) |
| `tests/test_agents.py` | Unit tests for each agent (mocked LLMs) |
| `tests/test_api.py` | API endpoint integration tests |
| `tests/test_pipeline.py` | End-to-end harness tests (mocked externals) |

Run: `pytest tests/ -v`