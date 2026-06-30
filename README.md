# fact_checker

**AI-powered multi-modal fact-checking pipeline with deep research, citations, and local object detection.**

## Overview

fact_checker is an automated pipeline that ingests video, audio, images, documents, and web content; extracts verifiable factual claims; performs deep multi-source research using free APIs; and produces evidence-grounded verdicts with inline citations, exact source quotes, and confidence scores.

## Features

### 🎯 **Multi-Modal Input**
- **YouTube URLs** (watch, shorts, live, youtu.be) — 3-layer caption extraction
- **Direct media URLs** — video/audio download + Whisper ASR
- **Local files** — video, audio, PDF, images, text, DOCX, HTML
- **Web articles** — trafilatura/BeautifulSoup extraction
- **Image-only jobs** — vision analysis without transcript

### 🔬 **Deep Research Pipeline**
1. **Claim Extraction** — Atomic, checkable claims from transcript + OCR
2. **Multi-Source Search** — 8 free providers (DuckDuckGo, Brave, Semantic Scholar, Arxiv, PubMed, Crossref, GovInfo, Wikipedia)
3. **Quote Extraction** — Exact supporting quotes with context windows
4. **Stance Classification** — Supporting/refuting/neutral via LLM
5. **Diversity Gates** — ≥3 domains, ≥2 source types, temporal freshness
6. **Contradiction Detection** — Fact-check disagreement flagging
7. **Iterative Research** — Up to 2 rounds with adversarial queries

### 📝 **Citation-Rich Verdicts**
- **Inline citations** `[doc_id]` in explanations
- **Exact quotes** from sources with context
- **Structured citations** linked to evidence IDs
- **5 verdict labels**: Supported, Refuted, Misleading, Insufficient Evidence, Unverifiable
- **Confidence calibration** with source quality signals
- **Human-review routing** for low confidence/contradictions/sensitive topics

### 🖼️ **Vision + Local Object Detection**
- **Frame extraction** — ffmpeg keyframes (30s interval, scene detection planned)
- **Vision LLM** (Llama-4-Maverick) — Scene description, OCR, visible claims, manipulation risk
- **Local YOLO** (optional) — Fast object detection, structured context for Vision LLM
- **Frame-transcript correlation** — ±5s window alignment

### 👤 **User Context & Steering**
- **User context** — Background info, hypothesis
- **User question** — Specific question to answer
- **Focus areas** — Topics to prioritize
- Passed through entire pipeline for targeted research

### 🌐 **Interfaces**
- **FastAPI** — Async job submission, SSE streaming, citation/evidence endpoints
- **Textual TUI** — Claims/verdicts tables, citation panel, research graph
- **MCP Server** — Claude Desktop / Cursor integration
- **CLI** — `fact-checker submit/file/serve/tui`

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Input     │────▶│   Ingest    │────▶│   Vision    │────▶│   Embed     │
│  (URL/File) │     │  (3-layer)  │     │  (YOLO+LLM) │     │ (ChromaDB)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                       │
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌────────┴─────┐
│  Verdict    │◀───│  Research   │◀───│  Extract    │◀───┤  (context)   │
│  (Citations)│     │ (ReAct)     │     │  Claims     │     │  + User Ctx  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Quick Start

### Prerequisites
- Python 3.11+
- ffmpeg/ffprobe
- PostgreSQL 16+
- OpenRouter API key (for LLMs)

### Installation
```bash
git clone https://github.com/cbwinslow/fact_checker
cd fact_checker
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Configuration
```bash
cp .env.example .env
# Edit .env with your keys:
# OPENROUTER_API_KEY=sk-or-...
# DATABASE_URL=postgresql+asyncpg://user:pass@localhost/fact_checker
# Optional: BRAVE_SEARCH_API_KEY, GOOGLE_FACTCHECK_API_KEY
```

### Run API Server
```bash
# Development
docker compose --profile dev up

# Or locally
fact-checker serve
```

### Submit a Job
```bash
# CLI
fact-checker submit "https://youtube.com/watch?v=..."

# API
curl -X POST http://localhost:8000/submit \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/watch?v=...", "user_question": "Is claim X true?"}'
```

### Check Results
```bash
# Poll for status
curl http://localhost:8000/jobs/{job_id}

# Get citations
curl http://localhost:8000/jobs/{job_id}/citations

# Stream progress (SSE)
curl http://localhost:8000/jobs/{job_id}/stream
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Required |
| `OPENROUTER_MODEL` | Fallback model | `openai/gpt-oss-120b:free` |
| `MODEL_EXTRACTION` | Claim extraction model | `openai/gpt-oss-120b:free` |
| `MODEL_VERIFICATION` | Verification/research model | `nvidia/nemotron-3-ultra-253b:free` |
| `MODEL_ORCHESTRATION` | Pipeline orchestration | `nvidia/nemotron-3-super-49b-v1:free` |
| `MODEL_MULTIMODAL` | Vision model | `meta-llama/llama-4-maverick:free` |
| `DATABASE_URL` | PostgreSQL connection | `postgresql+asyncpg://...` |
| `WHISPER_MODEL_SIZE` | Whisper model | `base` |
| `WHISPER_DEVICE` | cpu/cuda | `cpu` |

## Project Structure

```
fact_checker/
├── src/fact_checker/
│   ├── agents/
│   │   ├── claim_extractor.py      # Claim extraction with user context
│   │   ├── deep_research_agent.py  # Original research agent
│   │   ├── research_workflow.py    # LangGraph ReAct workflow
│   │   ├── image_analyst.py        # Vision LLM + YOLO integration
│   │   └── verdict_agent.py        # Citation-rich verdicts
│   ├── agents/
│   │   ├── claim_extractor.py
│   │   ├── deep_research_agent.py
│   │   ├── research_workflow.py    # LangGraph ReAct workflow
│   │   ├── image_analyst.py        # Vision + YOLO
│   │   └── verdict_agent.py
│   ├── services/
│   │   ├── file_router.py          # Universal media router
│   │   ├── ingest.py               # 3-layer YouTube ingest
│   │   ├── audio_ingest.py         # Audio-only Whisper
│   │   ├── pdf_ingest.py           # PDF extraction
│   │   ├── web_scraper.py          # Article scraping
│   │   ├── vision.py               # Frame extraction, EXIF, data URLs
│   │   ├── yolo_detector.py        # Local YOLO object detection
│   │   ├── search_providers.py     # 8 free search providers
│   │   ├── embedder.py             # Embeddings (local/OpenRouter)
│   │   ├── vector_store.py         # ChromaDB wrapper
│   │   └── rate_limiter.py         # Token bucket rate limiting
│   ├── skills/
│   │   ├── claim_skills.py         # Claim normalization, dedup, typing
│   │   ├── evidence_skills.py      # Scoring, diversity, contradictions
│   │   ├── research_skills.py      # Query planning, gap analysis
│   │   └── verdict_skills.py       # Calibration, review routing
│   ├── prompts/
│   │   ├── claim_extraction.md
│   │   ├── verdict_draft.md
│   │   └── image_analysis.md
│   ├── models.py                   # Pydantic models (Citation, EvidenceItem, etc.)
│   ├── harness.py                  # Pipeline orchestrator
│   ├── api.py                      # FastAPI endpoints
│   ├── cli.py                      # Typer CLI
│   ├── tui.py                      # Textual dashboard
│   └── config.py                   # Settings + Model registry
├── tests/                          # 28 passing tests
├── mcp/                            # MCP server for AI assistants
├── alembic/                        # DB migrations
└── docker-compose.yml              # Dev + prod containers
```

## Free Search Providers

| Provider | Type | Free Tier | Auth |
|----------|------|-----------|------|
| DuckDuckGo | General | Unlimited (HTML) | None |
| Brave Search | General | 2,000/mo | API key |
| Semantic Scholar | Academic | 100/5min | Optional |
| Arxiv | Preprints | Unlimited | None |
| PubMed/NCBI | Biomedical | Unlimited | Email |
| Crossref | DOIs | Unlimited | None |
| GovInfo | US Gov | Unlimited | Optional |
| Wikipedia | Encyclopedia | Unlimited | None |

## Extending the Pipeline

### Add a Search Provider
```python
class MyProvider(SearchProvider):
    name = "my_provider"
    source_type = "academic"
    
    async def search(self, query, max_results=10):
        # Return List[SearchResult]
        pass
    
    async def fetch_full(self, url):
        # Return full text
        pass

# Register
get_registry().register(MyProvider())
```

### Add a Skill
```python
# skills/my_skills.py
def my_utility_function(data: List[Dict]) -> Dict:
    # Pure function, no I/O, no LLM calls
    pass
```

### Custom Prompt
```markdown
# prompts/custom.md
Your system prompt here...
# Use in agent:
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "custom.md"
```

## Testing

```bash
# All tests
pytest tests/ -v

# Specific suite
pytest tests/test_agents.py -v
pytest tests/test_api.py -v
pytest tests/test_pipeline.py -v

# With coverage
pytest --cov=src/fact_checker tests/
```

## Documentation

- **AGENTS.md** — Agent architecture & pipeline specification
- **SRS.md** — Software Requirements Specification
- **FEATURES.md** — Feature catalog & implementation status
- **IMPLEMENTATION_PLAN.md** — Detailed implementation plan

### Documentation map
- [Agent architecture](AGENTS.md)
- [Ingest skill](skills/ingest/SKILL.md)
- [Image analysis skill](skills/image-analysis/SKILL.md)
- [Claim extraction skill](skills/claim-extraction/SKILL.md)
- [Evidence retrieval skill](skills/evidence-retrieval/SKILL.md)
- [Deep research skill](skills/deep-research/SKILL.md)
- [Verdict skill](skills/verdict/SKILL.md)
- [MCP server skill](mcp/SKILL.md)
- [MCP server config](mcp/servers.md)

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit PR

## Roadmap

- [ ] Scene detection for smarter frame extraction
- [ ] Semantic evidence deduplication
- [ ] Source reliability learning from verdicts
- [ ] Cross-job claim clustering
- [ ] ClaimReview schema export
- [ ] Multi-language support
- [ ] Human review UI
- [ ] Scheduled monitoring jobs