# FEATURES.md — Feature Catalog & Implementation Status

> **Purpose**: Living feature tracker mapping requirements to code locations, test coverage, and completion status.

---

## Feature Status Legend
| Symbol | Meaning |
|--------|---------|
| ✅ | Implemented, tested, working |
| 🟡 | Partially implemented / needs work |
| ❌ | Not yet implemented |
| 🔧 | Needs refactoring / tech debt |
| 📋 | Planned / documented only |

---

## 1. Input Ingestion

| Feature | Status | Implementation | Tests | Notes |
|---------|--------|----------------|-------|-------|
| YouTube URL ingest (yt-dlp captions) | ✅ | `services/ingest.py::_fetch_ytdlp_captions()` | `test_agents.py` | Layer 1, VTT parsing |
| YouTube URL ingest (youtube-transcript-api) | ✅ | `services/ingest.py::_fetch_yt_transcript_api()` | `test_agents.py` | Layer 2 fallback |
| YouTube URL ingest (Whisper ASR) | ✅ | `services/ingest.py::_transcribe_whisper()` | `test_agents.py` | Layer 3, audio download + transcribe |
| Direct video/audio URL download + ASR | ✅ | `file_router.py::_ingest_video_url()`, `_ingest_audio_url()` | — | Via MediaRouter |
| Local video file → transcript | ✅ | `file_router.py::_ingest_video()` → `ingest.ingest()` | — | |
| Local audio file → transcript | ✅ | `file_router.py::_ingest_audio()` → `audio_ingest.ingest_audio_file()` | — | |
| PDF ingest (local + URL) | ✅ | `services/pdf_ingest.py::ingest_pdf()`, `file_router.py::_ingest_pdf()` | — | Uses `pymupdf`/`marker-pdf` |
| Web article scraping | ✅ | `services/web_scraper.py::scrape_article()`, `html_to_text()` | `test_agents.py` | Trafilatura + BS4 |
| Text/MD/CSV file ingest | ✅ | `file_router.py::_ingest_text()` | — | Paragraph segmentation |
| DOCX file ingest | ✅ | `file_router.py::_ingest_docx()` | — | python-docx, fallback to raw text |
| Local HTML file ingest | ✅ | `file_router.py::_ingest_local_html()` | — | |
| Image-only job (vision only) | ✅ | `file_router.py::route()` returns `IngestSource.IMAGE` | — | Empty transcript, frames analysed |
| Media type auto-detection | ✅ | `skills/ingest_skills.py::detect_media_type()` | — | Extension + URL pattern + MIME |

---

## 2. Vision & Frame Analysis

| Feature | Status | Implementation | Tests | Notes |
|---------|--------|----------------|-------|-------|
| Keyframe extraction (ffmpeg) | ✅ | `services/vision.py::extract_frames()` | — | Interval 30s, max 20 frames |
| Video duration probe (ffprobe) | ✅ | `services/vision.py::_probe_duration()` | — | |
| EXIF metadata extraction | ✅ | `services/vision.py::read_image_metadata()` | — | Pillow + piexif |
| Image → base64 data URL | ✅ | `services/vision.py::image_to_data_url()` | — | Max side 1024px |
| Vision LLM analysis (objects, OCR, claims) | ✅ | `agents/image_analyst.py::analyse_images()` | `test_agents.py` | Multimodal model slot |
| Manipulation risk assessment | ✅ | `prompts/image_analysis.md` → LLM | — | low/medium/high + rationale |
| Visible claims promotion to Claim objects | ✅ | `harness.py` Stage 4 (lines 287-302) | — | `image_id` linkage |
| Frame-transcript timestamp correlation | 🟡 | `skills/image_skills.py::correlate_frames_to_transcript()` | — | Implemented but not wired in harness |

---

## 3. Claim Extraction

| Feature | Status | Implementation | Tests | Notes |
|---------|--------|----------------|-------|-------|
| Atomic claim extraction from transcript | ✅ | `agents/claim_extractor.py::extract_claims()` | `test_agents.py` | `extraction` model slot |
| Prompt-driven (external markdown) | ✅ | `prompts/claim_extraction.md` | — | |
| JSON output parsing + markdown fence stripping | ✅ | `claim_extractor.py` lines 58-63 | — | |
| Claim normalisation (NFC, quotes, punctuation) | ✅ | `skills/claim_skills.py::normalise_claim_text()` | — | |
| Deduplication (fuzzy similarity ≥ 0.82) | ✅ | `skills/claim_skills.py::deduplicate_claims()` | — | SequenceMatcher |
| Priority scoring (confidence + specificity + checkable) | ✅ | `skills/claim_skills.py::score_claim_priority()` | — | |
| Claim type classification (7 types) | ✅ | `skills/claim_skills.py::classify_claim_type()` | — | Regex heuristic |
| Checkable flag + confidence per claim | ✅ | `models.py::Claim` fields | — | |

---

## 4. Embedding & Vector Store

| Feature | Status | Implementation | Tests | Notes |
|---------|--------|----------------|-------|-------|
| Transcript chunking (overlap-aware) | ✅ | `services/embedder.py::embed_segments()` | — | 1500 char chunks, 150 overlap |
| Embedding via `extraction` model | ✅ | `services/embedder.py::embed_texts()` | — | OpenRouter embeddings |
| Per-job ChromaDB VectorStore | ✅ | `services/vector_store.py::VectorStore` | — | Persistent per job_id |
| Semantic retrieval (top-k, min_score) | ✅ | `VectorStore.search()` | — | Used by DeepResearchAgent |

---

## 5. Deep Research & Evidence Retrieval

| Feature | Status | Implementation | Tests | Notes |
|---------|--------|----------------|-------|-------|
| Google Fact Check Tools API | ✅ | `agents/evidence_agent.py::_google_factcheck()`, `deep_research_agent.py::_google_factcheck()` | — | Requires `GOOGLE_FACTCHECK_API_KEY` |
| Serper.dev web search | ✅ | `agents/evidence_agent.py::_serper_search()`, `deep_research_agent.py::_serper_search()` | — | Requires `SERPER_API_KEY` |
| Full-page scraping (top 3 results) | ✅ | `deep_research_agent.py::_scrape_pages()` | — | `services/web_scraper.py::html_to_text()` |
| Adversarial counter-queries | ✅ | `deep_research_agent.py::_is_weak_evidence()` + counter-query | — | Triggered when < 2 high-relevance items |
| Wikipedia entity lookup | ✅ | `deep_research_agent.py::_wikipedia_lookup()` | — | First capitalised phrase |
| Source credibility tiering (4 tiers) | ✅ | `skills/evidence_skills.py::score_source_credibility()` | — | Tier 1: fact-checkers, .gov, .edu |
| Fact-check domain detection | ✅ | `skills/evidence_skills.py::is_factcheck_domain()` | — | 20+ known domains |
| Evidence ranking (composite score) | ✅ | `skills/evidence_skills.py::rank_evidence_snippets()` | — | Overlap × credibility × FC bonus |
| Deduplication by URL | ✅ | `deep_research_agent.py::_score_and_deduplicate()` | — | Keeps highest composite |
| Research query planning (multi-hop) | ✅ | `skills/research_skills.py::plan_research_queries()` | — | 6 query types |
| Counter-query generation | ✅ | `skills/research_skills.py::generate_counter_queries()` | — | "false", "debunked", "context missing" |
| Evidence gap analysis | ✅ | `skills/research_skills.py::analyse_evidence_gaps()` | — | Severity: critical/moderate/minor/none |
| Research brief summarisation | ✅ | `skills/research_skills.py::summarise_research_brief()` | — | Template-based, no LLM |

---

## 6. Verdict Generation

| Feature | Status | Implementation | Tests | Notes |
|---------|--------|----------------|-------|-------|
| Verdict per claim (5 labels) | ✅ | `agents/verdict_agent.py::draft_verdicts()` | `test_agents.py` | `verification` model slot |
| Prompt-driven (external markdown) | ✅ | `prompts/verdict_draft.md` | — | |
| Evidence formatting with citations | ✅ | `verdict_agent.py` lines 73-76 | — | Markdown links in prompt |
| Confidence calibration | ✅ | `skills/verdict_skills.py::calibrate_confidence()` | — | FC boost, contradiction penalty, thin evidence penalty |
| Human review routing | ✅ | `skills/verdict_skills.py::route_for_human_review()` | — | Confidence < 0.6, contradictions, sensitive topics, misleading |
| Sensitive topic detection | ✅ | `verdict_skills.py` `_SENSITIVE_KEYWORDS` | — | 12 keywords |
| Job-level aggregation | ✅ | `skills/verdict_skills.py::aggregate_verdicts()` | — | Distribution, mean/min confidence, overall verdict |
| Human-readable report formatting | ✅ | `skills/verdict_skills.py::format_verdict_report()` | — | ASCII confidence bar |

---

## 7. Pipeline Orchestration

| Feature | Status | Implementation | Tests | Notes |
|---------|--------|----------------|-------|-------|
| 6-stage sequential pipeline | ✅ | `harness.py::run_pipeline()` | `test_pipeline.py` | Ingest → Vision → Embed → Extract → Research → Verdict |
| Shared AnalysisContext packet | ✅ | `models.py::AnalysisContext` | — | All artifacts flow through |
| Live DB status updates per stage | ✅ | `harness.py::_update_job_status()` | `test_api.py` | `pending` → `ingesting` → `analyzing` → `embedding` → `extracting` → `researching` → `verdicting` → `done`/`review`/`failed` |
| Error handling + partial preservation | ✅ | `harness.py` try/except + FAILED status | — | |
| Cost estimation pre-flight | ✅ | `skills/ingest_skills.py::estimate_processing_cost()` | — | Token tiers: low/medium/high |
| Concurrent claim research (semaphore) | ✅ | `deep_research_agent.py` semaphore | — | Default 3 concurrent |
| Webhook notification on completion | ✅ | `services/webhook_notifier.py::notify_webhook()` | — | Called from API background task |

---

## 8. API Layer

| Feature | Status | Implementation | Tests | Notes |
|---------|--------|----------------|-------|-------|
| `POST /submit` (async job) | ✅ | `api.py::submit()` | `test_api.py` | Returns `job_id`, 202 |
| `GET /jobs/{job_id}` (status + results) | ✅ | `api.py::get_job()` | `test_api.py` | |
| `GET /jobs` (paginated, filtered) | ✅ | `api.py::list_jobs()` | `test_api.py` | |
| `DELETE /jobs/{job_id}` (cascade) | ✅ | `api.py::delete_job()` | `test_api.py` | |
| `POST /jobs/{job_id}/retry` | ✅ | `api.py::retry_job()` | `test_api.py` | Resets FAILED/DONE → PENDING |
| `GET /health` (no auth) | ✅ | `api.py::health()` | `test_api.py` | |
| `GET /metrics` (auth) | ✅ | `api.py::metrics()` | `test_api.py` | Aggregated counts by status |
| API key auth (optional) | ✅ | `auth.py::require_api_key` | `test_api.py` | Disabled if `API_KEY` unset |
| CORS configurable | ✅ | `api.py` startup | — | `CORS_ORIGINS` env |

---

## 9. CLI & MCP

| Feature | Status | Implementation | Tests | Notes |
|---------|--------|----------------|-------|-------|
| `fact-checker submit <url>` | ✅ | `cli.py::submit()` | — | Pretty Rich table output |
| `fact-checker file <path>` | ✅ | `cli.py::file()` | — | |
| `fact-checker serve` | ✅ | `cli.py::serve()` | — | Uvicorn with reload |
| `fact-checker tui` | 🟡 | `cli.py::tui()` → `tui.py` | — | Textual dashboard, basic |
| `fact-checker jobs` | 🟡 | Not yet implemented | — | |
| MCP server | ✅ | `mcp/fact_checker_mcp_server.py` | — | 8 tools exposed |
| MCP: `submit_job` | ✅ | `mcp/fact_checker_mcp_server.py` | — | |
| MCP: `get_job_status` | ✅ | `mcp/fact_checker_mcp_server.py` | — | |
| MCP: `get_claims` | ✅ | `mcp/fact_checker_mcp_server.py` | — | |
| MCP: `get_verdicts` | ✅ | `mcp/fact_checker_mcp_server.py` | — | |
| MCP: `search_evidence` | ✅ | `mcp/fact_checker_mcp_server.py` | — | |
| MCP: `extract_claims_text` | ✅ | `mcp/fact_checker_mcp_server.py` | — | Direct text → claims |
| MCP: `detect_media_type` | ✅ | `mcp/fact_checker_mcp_server.py` | — | Uses `ingest_skills.detect_media_type` |
| MCP: `estimate_cost` | ✅ | `mcp/fact_checker_mcp_server.py` | — | Uses `ingest_skills.estimate_processing_cost` |

---

## 10. Database & Persistence

| Feature | Status | Implementation | Notes |
|---------|--------|----------------|-------|
| PostgreSQL async (asyncpg) | ✅ | `db.py` SQLAlchemy 2.0 async | |
| Alembic migrations | ✅ | `alembic/versions/0001_initial_schema.py` | 5 tables |
| Cascade deletes (job → segments/claims/evidence/verdicts) | ✅ | FK `ondelete="CASCADE"` | |
| Job row pre-creation for polling | ✅ | `api.py::submit()` background task | |
| Pipeline result persistence | ✅ | `db.py::save_pipeline_result()` | |
| VectorStore per-job (ChromaDB) | ✅ | `services/vector_store.py` | Ephemeral, in `artifact_dir` |

---

## 11. Configuration & Models

| Feature | Status | Implementation | Notes |
|---------|--------|----------------|-------|
| Pydantic Settings from `.env` | ✅ | `config.py::Settings`, `settings.py::Settings` | Two config files (legacy + new) |
| Per-task model registry | ✅ | `config.py::MODEL_REGISTRY` | 6 slots |
| Per-task env overrides | ✅ | `MODEL_EXTRACTION`, `MODEL_VERIFICATION`, etc. | |
| MockChatModel for offline dev | ✅ | `config.py::MockChatModel` | Deterministic stub JSON |
| Free-tier model defaults | ✅ | All defaults are `:free` on OpenRouter | |

---

## 12. Testing

| Test Suite | Status | Coverage |
|------------|--------|----------|
| `tests/conftest.py` fixtures | ✅ | Shared mocks, AsyncClient |
| `tests/test_agents.py` | ✅ | Agent unit tests (mocked LLMs) |
| `tests/test_api.py` | ✅ | API integration tests |
| `tests/test_pipeline.py` | ✅ | Harness E2E tests |
| `test_smoke.py` | ✅ | Trivial sanity check |

---

## 13. Documentation

| Doc | Status | Location |
|-----|--------|----------|
| README | ✅ | `README.md` |
| Agent Architecture | ✅ | `AGENTS.md` |
| Skills contracts | ✅ | `skills/*/SKILL.md` (6 files) |
| MCP contracts | ✅ | `mcp/SKILL.md`, `mcp/servers.md` |
| Software Requirements Spec | ✅ | `SRS.md` |
| Feature Catalog (this file) | ✅ | `FEATURES.md` |
| Prompts | ✅ | `src/fact_checker/prompts/*.md` (5 files) |

---

## 14. Known Gaps / Tech Debt

| Area | Issue | Severity |
|------|-------|----------|
| Dual config files | `settings.py` (legacy) + `config.py` (new) both exist | 🟡 Medium |
| Root-level duplicate modules | `claim_extraction.py`, `harness.py`, `models.py`, `openrouter.py`, `media.py` mirror `src/fact_checker/` | 🟡 Medium |
| Frame-transcript correlation | `correlate_frames_to_transcript()` implemented but not called in harness | 🟡 Medium |
| TUI dashboard | Minimal implementation, not feature-complete | 🟡 Low |
| `fact-checker jobs` CLI command | Not implemented | 🟡 Low |
| No claim clustering across jobs | Each job independent | 📋 Planned |
| No export formats (PDF, ClaimReview) | Only JSON/API output | 📋 Planned |
| No scheduled monitoring jobs | Cron/APScheduler not integrated | 📋 Planned |
| Single-threaded frame analysis | `analyse_images()` processes sequentially | 🔧 Could parallelize |
| No GPU detection for Whisper | Hardcoded `device=cpu` default | 🔧 Could auto-detect |

---

## 15. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-01-01 | Initial skeleton, basic pipeline |
| 0.2.0 | 2025-06-27 | Full pipeline: ingest → vision → embed → extract → research → verdict; API, CLI, MCP, skills, tests |
| 0.3.0 | Planned | Multi-language, claim graph, review UI, exports, monitoring |