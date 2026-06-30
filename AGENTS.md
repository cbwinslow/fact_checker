---
name: fact-checker-agent-architecture
version: 1
summary: Canonical agent registry and handoff contract for the fact_checker pipeline.
owners:
  - cbwinslow
repo: fact_checker
entrypoints:
  - src/factchecker/harness.py
  - src/factchecker/api.py
  - mcp/factcheckermcpserver.py
---

# AGENTS.md

## Purpose
This file is the canonical registry for the fact_checker pipeline. It defines each agent, its responsibilities, inputs, outputs, tool dependencies, and handoff contract.

## Pipeline order
1. MediaRouter
2. ImageAnalystAgent
3. Embedder
4. ClaimExtractorAgent
5. DeepResearchAgent
6. VerdictDraftAgent
7. Orchestrator / Harness

## Shared context
All stages read from and write to a shared `AnalysisContext`. Cross-stage identity keys should remain UUID based: `jobid`, `claimid`, `imageid`, and `segmentid`.

## Agent registry

### MediaRouter
- Module: `src/factchecker/services/filerouter.py`
- Purpose: Detect input type and route to the correct ingest path.
- Inputs: `jobid`, `url`, `localpath`, `imagepaths`, `sourcetype`
- Outputs: transcript segments, ingest source classification, image-only routing state
- Depends on: ingest service, audio ingest, pdf ingest, web scraper
- Handoff: populated `ctx.segments`, `ctx.ingestsource`

### ImageAnalystAgent
- Module: `src/factchecker/agents/imageanalyst.py`
- Purpose: Extract OCR text, visible claims, objects, and manipulation signals from uploaded images or frames.
- Inputs: `jobid`, `imagepaths`, optional frame timestamps
- Outputs: `ImageAnalysis[]`
- Depends on: `services/vision.py`, multimodal model slot, `prompts/imageanalysis.md`
- Handoff: populated `ctx.images`

### Embedder
- Module: `src/factchecker/services/embedder.py`
- Purpose: Chunk transcript content and create embeddings for semantic retrieval.
- Inputs: `jobid`, `ctx.segments`
- Outputs: embedded chunks, job vector store
- Depends on: vector store, extraction model slot
- Handoff: populated `ctx.chunks`, `ctx.vectorstore`

### ClaimExtractorAgent
- Module: `src/factchecker/agents/claimextractor.py`
- Purpose: Convert transcript and image-derived text into atomic, checkable claims.
- Inputs: `ctx.segments`, `ctx.images`
- Outputs: `Claim[]`
- Depends on: `prompts/claimextraction.md`, claim skills, extraction model slot
- Handoff: populated `ctx.claims`

### DeepResearchAgent
- Module: `src/factchecker/agents/deepresearchagent.py`
- Purpose: Gather, score, and deduplicate supporting and contradicting evidence for each claim.
- Inputs: `ctx.claims`, `ctx.vectorstore`
- Outputs: `ResearchResult[]`, flattened evidence items
- Depends on: search providers, web scraper, research skills, evidence skills, verification model slot
- Handoff: populated `ctx.researchresults`, `ctx.evidence`

### EvidenceAgent
- Module: `src/factchecker/agents/evidenceagent.py`
- Purpose: Provide a simpler evidence retrieval path when deep research is not needed.
- Inputs: claims or ad hoc evidence retrieval requests
- Outputs: evidence snippets and ranked sources
- Depends on: Google Fact Check, Serper, evidence skills
- Handoff: optional alternate evidence pipeline

### VerdictDraftAgent
- Module: `src/factchecker/agents/verdictagent.py`
- Purpose: Turn gathered evidence into evidence-grounded claim verdicts with calibrated confidence.
- Inputs: `ctx.claims`, claim-grouped evidence
- Outputs: `VerdictResult[]`
- Depends on: `prompts/verdictdraft.md`, verdict skills, verification model slot
- Handoff: populated `ctx.verdicts`

### Orchestrator / Harness
- Module: `src/factchecker/harness.py`
- Purpose: Execute the sequential pipeline, manage status transitions, and preserve partial results.
- Inputs: job submission payloads from CLI, API, or MCP
- Outputs: completed pipeline state, persisted job artifacts, final summary
- Depends on: all stages, DB persistence, webhook notifier
- Handoff: API / CLI / MCP response surfaces

## Invariants
- `AnalysisContext` is the single source of truth.
- Stages should be idempotent when possible.
- Partial results should survive failures.
- Status updates should be emitted between stages.
- Image-only inputs should still support claim generation through visible-claim promotion.

## Gaps to close next
- Remove or clearly deprecate duplicate root-level modules that mirror `src/factchecker`.
- Consolidate `settings.py` and `config.py` into one configuration story.
- Wire frame-to-transcript correlation into the harness.
- Add per-agent SKILL.md files so humans and agent frameworks can consume each component consistently.
