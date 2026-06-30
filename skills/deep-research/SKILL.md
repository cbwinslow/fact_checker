---
name: deep-research
version: 1
summary: Perform multi-hop evidence retrieval, adversarial query expansion, page scraping, and evidence-gap analysis per claim.
module: src/factchecker/agents/deepresearchagent.py
owners:
  - cbwinslow
inputs:
  - Claim records
  - job vector store
outputs:
  - ResearchResult records
  - flattened evidence items
models:
  - verification
python_skills:
  - src/factchecker/skills/researchskills.py
  - src/factchecker/skills/evidenceskills.py
external_dependencies:
  - Google Fact Check API
  - Serper
  - Wikipedia
---

# Deep Research

## Purpose
Collect the strongest evidence package for each claim, including contradictory evidence and missing-context detection.

## Responsibilities
- Run semantic retrieval over job-local embeddings.
- Issue primary and adversarial web queries.
- Scrape top pages for richer evidence.
- Rank and deduplicate results.
- Produce an evidence brief suitable for verdict drafting.

## Handoffs
Writes `ctx.researchresults` and the canonical evidence collection used by verdict generation.
