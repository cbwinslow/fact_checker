---
name: evidence-retrieval
version: 1
summary: Retrieve and rank relevant evidence for each claim using fact-check APIs, web search, scraping, and source scoring.
module: src/factchecker/agents/evidenceagent.py
owners:
  - cbwinslow
inputs:
  - Claim records
outputs:
  - ranked evidence snippets
  - scored source metadata
models:
  - verification
python_skills:
  - src/factchecker/skills/evidenceskills.py
external_dependencies:
  - Google Fact Check API
  - Serper
---

# Evidence Retrieval

## Purpose
Provide a focused retrieval path for claims that need supporting or contradicting source material.

## Responsibilities
- Query fact-check sources and search providers.
- Gather source snippets.
- Score credibility and relevance.
- Deduplicate duplicate URLs and near-duplicate snippets.

## Handoffs
Provides ranked evidence into `ctx.evidence` or a simplified evidence response surface.
