# SRS.md — Software Requirements Specification

> **Project**: fact_checker — AI-powered multi-modal fact-checking pipeline  
> **Version**: 0.2.0  
> **Status**: Active development  

---

## 1. Introduction

### 1.1 Purpose
fact_checker is an automated pipeline that ingests video, audio, images, documents, and web content; extracts verifiable factual claims; performs deep multi-source research; and produces evidence-grounded verdicts with confidence scores and human-review flags.

### 1.2 Scope
- **In scope**: End-to-end fact-checking of public claims from YouTube, social media, news videos, podcasts, PDFs, web articles, and uploaded media files.
- **Out of scope**: Real-time streaming analysis, private/internal document verification, legal evidence processing.

### 1.3 Definitions
| Term | Definition |
|------|------------|
| **Claim** | An atomic, verifiable factual assertion extracted from source content. |
| **Verdict** | Evidence-backed classification: `supported`, `refuted`, `misleading`, `insufficient_evidence`, `unverifiable`. |
| **EvidenceItem** | A single retrieved source (URL + snippet + credibility score) relevant to a claim. |
| **Job** | One pipeline execution with a unique UUID, tracking status and all artifacts. |
| **AnalysisContext** | Typed packet carrying all intermediate artifacts between pipeline stages. |

---

## 2. Functional Requirements

### 2.1 Input Ingestion (FR-INGEST)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-INGEST-01 | Accept YouTube URLs (watch, shorts, live, youtu.be) | Must |
| FR-INGEST-02 | Accept direct video/audio file URLs (mp4, mkv, mp3, m4a, etc.) | Must |
| FR-INGEST-03 | Accept local file uploads: video, audio, PDF, images, text, DOCX, HTML | Must |
| FR-INGEST-04 | Accept web article URLs for scraping | Must |
| FR-INGEST-05 | Accept image-only jobs (no transcript) for vision analysis | Must |
| FR-INGEST-06 | 3-layer transcript pipeline: yt-dlp captions → youtube-transcript-api → Whisper ASR | Must |
| FR-INGEST-07 | Extract EXIF metadata from images (camera, GPS, datetime, software) | Should |
| FR-INGEST-08 | Normalise audio to 16 kHz mono WAV before Whisper | Must |
| FR-INGEST-09 | Return `IngestSource` enum identifying which layer succeeded | Must |

### 2.2 Vision & Frame Analysis (FR-VISION)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-VISION-01 | Extract keyframes from video at configurable interval (default 30s, max 20) | Must |
| FR-VISION-02 | Analyse each frame with vision-capable LLM (objects, OCR, visible claims) | Must |
| FR-VISION-03 | Detect manipulation risk: `low`, `medium`, `high` with rationale | Should |
| FR-VISION-04 | Correlate frame timestamps with transcript segments (±5s window) | Should |
| FR-VISION-05 | Promote `visible_claims` from frames as checkable `Claim` objects | Must |

### 2.3 Claim Extraction (FR-CLAIM)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-CLAIM-01 | Extract atomic, checkable claims from transcript segments | Must |
| FR-CLAIM-02 | Preserve numbers, names, dates exactly as stated | Must |
| FR-CLAIM-03 | Assign `is_checkable`, `confidence`, `context` per claim | Must |
| FR-CLAIM-04 | Deduplicate near-identical claims (similarity ≥ 0.82) | Should |
| FR-CLAIM-05 | Classify claim type: statistical, historical, causal, attributional, definitional, predictive, unverifiable | Should |
| FR-CLAIM-06 | Prioritise claims by specificity + confidence for research queue | Should |

### 2.4 Evidence Retrieval (FR-EVIDENCE)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-EVIDENCE-01 | Query Google Fact Check Tools API (ClaimReview database) | Should |
| FR-EVIDENCE-02 | Query Serper.dev for general web evidence | Should |
| FR-EVIDENCE-03 | Scrape full page content for top-N result URLs | Should |
| FR-EVIDENCE-04 | Score source credibility: Tier 1 (fact-checkers, .gov, .edu) → Tier 4 (social media) | Must |
| FR-EVIDENCE-05 | Generate adversarial counter-queries when evidence is weak | Should |
| FR-EVIDENCE-06 | Wikipedia lookup for named entities in claims | Should |
| FR-EVIDENCE-07 | Semantic retrieval from job VectorStore for context | Should |
| FR-EVIDENCE-08 | Deduplicate evidence by URL, keep highest composite score | Must |

### 2.5 Verdict Generation (FR-VERDICT)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-VERDICT-01 | Produce verdict per claim: `supported`, `refuted`, `misleading`, `insufficient_evidence`, `unverifiable` | Must |
| FR-VERDICT-02 | Calibrate LLM confidence using evidence signals (fact-check source boost, contradiction penalty, thin evidence penalty) | Must |
| FR-VERDICT-03 | Flag `requires_human_review` if confidence < 0.6, contradictions, sensitive topic, or `misleading` verdict | Must |
| FR-VERDICT-04 | Aggregate job-level summary: distribution, mean/min confidence, overall verdict | Should |
| FR-VERDICT-05 | Format human-readable plain-text report per verdict | Should |

### 2.6 Pipeline Orchestration (FR-ORCH)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-ORCH-01 | Execute all 6 stages sequentially with shared `AnalysisContext` | Must |
| FR-ORCH-02 | Write live job status to DB between stages (`pending` → `ingesting` → `analyzing` → `embedding` → `extracting` → `researching` → `verdicting` → `done`/`review`/`failed`) | Must |
| FR-ORCH-03 | Support async background execution with webhook notification on completion | Should |
| FR-ORCH-04 | Allow retry of failed jobs (reset status → rerun) | Should |
| FR-ORCH-05 | Estimate processing cost (tokens, time) before full run | Should |

### 2.7 API & Interfaces (FR-API)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-API-01 | `POST /submit` — submit URL/file for async fact-check, return `job_id` | Must |
| FR-API-02 | `GET /jobs/{job_id}` — poll status and results | Must |
| FR-API-03 | `GET /jobs` — paginated job list with status filter | Must |
| FR-API-04 | `DELETE /jobs/{job_id}` — delete job and all cascaded data | Should |
| FR-API-05 | `POST /jobs/{job_id}/retry` — re-queue failed/completed job | Should |
| FR-API-06 | `GET /health` — unauthenticated health check | Must |
| FR-API-07 | `GET /metrics` — authenticated pipeline metrics | Should |
| FR-API-08 | API key authentication (optional, disabled if `API_KEY` unset) | Should |
| FR-API-09 | CORS configurable via `CORS_ORIGINS` | Should |

### 2.8 CLI & MCP (FR-CLI)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-CLI-01 | `fact-checker submit <url>` — run pipeline, pretty-print results | Must |
| FR-CLI-02 | `fact-checker file <path>` — run on local file | Must |
| FR-CLI-03 | `fact-checker serve` — start FastAPI server with reload | Must |
| FR-CLI-04 | `fact-checker tui` — launch Textual dashboard | Should |
| FR-CLI-05 | `fact-checker jobs` — list recent jobs | Should |
| FR-MCP-01 | MCP server with tools: `submit_job`, `get_job_status`, `get_claims`, `get_verdicts`, `search_evidence`, `extract_claims_text`, `detect_media_type`, `estimate_cost` | Should |

---

## 3. Non-Functional Requirements

### 3.1 Performance
| ID | Requirement | Target |
|----|-------------|--------|
| NFR-PERF-01 | End-to-end latency for 10-min video | < 5 minutes |
| NFR-PERF-02 | Concurrent job processing | ≥ 3 parallel jobs |
| NFR-PERF-03 | VectorStore query latency | < 200ms |
| NFR-PERF-04 | Memory usage per job | < 2 GB |

### 3.2 Reliability
| ID | Requirement | Target |
|----|-------------|--------|
| NFR-REL-01 | Pipeline stage failures captured, partial results preserved | 100% |
| NFR-REL-02 | DB transaction atomicity per job | ACID |
| NFR-REL-03 | Graceful degradation when external APIs unavailable | Mock/stub responses |

### 3.3 Security
| ID | Requirement | Target |
|----|-------------|--------|
| NFR-SEC-01 | API keys never logged | Must |
| NFR-SEC-02 | Input validation on all endpoints | Must |
| NFR-SEC-03 | Temporary files cleaned up after processing | Must |

### 3.4 Maintainability
| ID | Requirement | Target |
|----|-------------|--------|
| NFR-MAIN-01 | All agents unit-testable with mocked LLMs | 100% |
| NFR-MAIN-02 | Skills (pure functions) have ≥ 90% test coverage | Should |
| NFR-MAIN-03 | Configuration via `.env` only, no code changes for model swaps | Must |

---

## 4. Data Model

### 4.1 Core Entities (from `models.py`)

| Entity | Key Fields |
|--------|------------|
| `VideoJob` | `id`, `url`, `local_path`, `status`, `ingest_source`, `created_at`, `error` |
| `TranscriptSegment` | `id`, `job_id`, `start_sec`, `end_sec`, `text`, `speaker` |
| `Claim` | `id`, `job_id`, `segment_id?`, `image_id?`, `text`, `is_checkable`, `confidence`, `context` |
| `EvidenceItem` | `id`, `claim_id`, `source_url`, `title`, `snippet`, `relevance_score`, `credibility_score`, `is_factcheck_source` |
| `VerdictResult` | `id`, `claim_id`, `verdict`, `explanation`, `confidence`, `evidence_ids[]`, `requires_human_review` |
| `ImageAnalysis` | `id`, `job_id`, `source_type`, `source_path`, `frame_sec`, `metadata`, `description`, `objects[]`, `text_in_image`, `visible_claims[]`, `manipulation_risk`, `manipulation_reason` |
| `EmbeddedChunk` | `id`, `job_id`, `text`, `vector[]`, `chunk_index`, `source_hash` |
| `ResearchResult` | `claim_id`, `evidence[]`, `context_snippets[]`, `avg_credibility`, `evidence_count`, `has_factcheck_source` |
| `AnalysisContext` | Aggregates all above + live `vector_store` |

### 4.2 Database Schema (Alembic 0001)

| Table | Columns |
|-------|---------|
| `video_jobs` | `job_id` (PK, UUID), `url`, `local_path`, `ingest_source`, `status`, `error_message`, `created_at`, `updated_at` |
| `transcript_segments` | `id` (BIGSERIAL PK), `job_id` (FK), `start_time`, `end_time`, `text`, `speaker` |
| `claims` | `claim_id` (PK, UUID), `job_id` (FK), `text`, `speaker`, `timestamp_start`, `timestamp_end`, `created_at` |
| `evidence_items` | `id` (BIGSERIAL PK), `claim_id` (FK), `source_url`, `snippet`, `relevance_score` |
| `verdicts` | `verdict_id` (PK, UUID), `claim_id` (FK), `job_id` (FK), `label`, `confidence`, `explanation`, `created_at` |

---

## 5. External Interfaces

### 5.1 LLM Provider (OpenRouter)
- **Endpoint**: `https://openrouter.ai/api/v1` (configurable)
- **Auth**: Bearer token via `OPENROUTER_API_KEY`
- **Models**: Per-task registry with free-tier defaults

### 5.2 Evidence APIs
| API | Purpose | Key Env Var |
|-----|---------|-------------|
| Google Fact Check Tools | ClaimReview database | `GOOGLE_FACTCHECK_API_KEY` |
| Serper.dev | General web search | `SERPER_API_KEY` |

### 5.3 Media Processing
- **ffmpeg/ffprobe**: Frame extraction, audio normalisation, duration probe
- **yt-dlp**: Video/audio download, caption extraction
- **faster-whisper**: Local ASR (CPU/GPU)

### 5.4 Database
- **PostgreSQL 16+** with `pgcrypto` for UUID generation
- **Async driver**: `asyncpg` via SQLAlchemy 2.0 async

---

## 6. Configuration

All settings via `pydantic-settings` from `.env` / environment:

```bash
# Required
OPENROUTER_API_KEY=sk-or-...
DATABASE_URL=postgresql+asyncpg://user:pass@host:port/db

# Optional model overrides
MODEL_EXTRACTION=openai/gpt-oss-120b:free
MODEL_VERIFICATION=nvidia/nemotron-3-ultra-253b:free
MODEL_ORCHESTRATION=nvidia/nemotron-3-super-49b-v1:free
MODEL_MULTIMODAL=meta-llama/llama-4-maverick:free
MODEL_TOOLUSE=cohere/north-mini-code:free
MODEL_FAST=openai/gpt-oss-20b:free

# Evidence APIs
GOOGLE_FACTCHECK_API_KEY=...
SERPER_API_KEY=...

# Whisper
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# Server
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
API_KEY=  # optional, enables auth

# Storage
ARTIFACT_DIR=./artifacts
MEDIA_CACHE_DIR=./media_cache
```

---

## 7. Deployment

### 7.1 Docker Compose (Production)
```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: fact_checker
      POSTGRES_PASSWORD: password
      POSTGRES_DB: fact_checker
    volumes:
      - pg_data:/var/lib/postgresql/data
      - ./migrations/schema.sql:/docker-entrypoint-initdb.d/schema.sql:ro
  
  api:
    build: .
    environment:
      DATABASE_URL: postgresql+asyncpg://fact_checker:password@db:5432/fact_checker
      OPENROUTER_API_KEY: ${OPENROUTER_API_KEY}
    ports:
      - "8000:8000"
    command: uvicorn fact_checker.api:app --host 0.0.0.0 --port 8000 --workers 2
```

### 7.2 Development Override
```bash
docker compose --profile dev up api-dev
# Mounts source, enables --reload, DEBUG logs
```

---

## 8. Testing Requirements

| Test Type | Coverage Target | Command |
|-----------|-----------------|---------|
| Unit (skills, agents) | ≥ 90% | `pytest tests/test_agents.py -v` |
| Integration (API) | ≥ 80% | `pytest tests/test_api.py -v` |
| E2E (harness) | Smoke | `pytest tests/test_pipeline.py -v` |
| Lint/Type | Clean | `ruff check . && mypy src/fact_checker` |

---

## 9. Future Extensions (v0.3+)

- [ ] Real-time streaming ingest (WebRTC, HLS)
- [ ] Multi-language support (Whisper + translation)
- [ ] Claim clustering across jobs (knowledge graph)
- [ ] Human review UI (TUI + web)
- [ ] Scheduled monitoring jobs (cron)
- [ ] Export formats: PDF report, JSONL, CSV, ClaimReview markup
- [ ] Model fine-tuning pipeline for domain-specific claims

---

## 10. Acceptance Criteria

| Scenario | Given | When | Then |
|----------|-------|------|------|
| AC-01 | YouTube URL submitted | Pipeline completes | Job status = `done`, ≥1 claim, ≥1 verdict |
| AC-02 | Local video file uploaded | Pipeline completes | Transcript extracted, frames analysed, verdicts rendered |
| AC-03 | Image-only job submitted | Pipeline completes | `visible_claims` extracted, promoted to claims, verdicts rendered |
| AC-04 | Google FC API key configured | Claim matches known fact-check | `is_factcheck_source=true`, high credibility |
| AC-05 | Contradictory evidence found | Verdict generated | `requires_human_review=true`, confidence calibrated down |
| AC-06 | API key missing | Any LLM call made | MockChatModel returns deterministic stub, pipeline completes |
| AC-07 | Job fails at stage 3 | Error occurs | Status = `failed`, error logged, partial context preserved |
| AC-08 | Retry failed job | `POST /jobs/{id}/retry` | Status reset to `pending`, pipeline re-runs |