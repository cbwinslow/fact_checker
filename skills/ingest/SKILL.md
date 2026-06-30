---
name: ingest
version: 1
summary: Normalize raw media, document, article, and text inputs into transcript-like segments for downstream fact-checking.
module: src/factchecker/services/filerouter.py
owners:
  - cbwinslow
inputs:
  - YouTube URLs
  - direct media URLs
  - local video and audio files
  - PDFs
  - web articles
  - text, markdown, docx, html
outputs:
  - transcript segments
  - ingest source classification
  - image-only routing state
models:
  - multimodal
  - fast
mcp_tools:
  - detect_media_type
  - estimate_cost
---

# Ingest

## Purpose
Turn heterogeneous inputs into normalized segments and metadata that the rest of the pipeline can process.

## Responsibilities
- Detect source type.
- Route to the correct ingest service.
- Produce transcript-like segments for media and text inputs.
- Preserve enough metadata for cost estimation and later audit.

## Handoffs
Writes `ctx.segments` and `ctx.ingestsource`.

## Failure policy
If ingest partially succeeds, preserve recovered segments and mark status for review instead of dropping the whole job.
