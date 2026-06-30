---
name: claim-extraction
version: 1
summary: Extract atomic, verifiable claims from transcript and image-derived content.
module: src/factchecker/agents/claimextractor.py
owners:
  - cbwinslow
inputs:
  - transcript segments
  - promoted visible claims
outputs:
  - Claim records
models:
  - extraction
prompts:
  - src/factchecker/prompts/claimextraction.md
python_skills:
  - src/factchecker/skills/claimskills.py
---

# Claim Extraction

## Purpose
Transform raw text into discrete claims that can be researched and judged independently.

## Responsibilities
- Extract factual claims only.
- Normalize claim text.
- Deduplicate near-duplicates.
- Classify claim type.
- Score priority and checkability for downstream processing.

## Handoffs
Writes `ctx.claims` using stable `claimid` values and preserves source linkage where possible.
