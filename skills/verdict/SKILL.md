---
name: verdict
version: 1
summary: Produce calibrated verdicts and explanations for each claim using retrieved evidence.
module: src/factchecker/agents/verdictagent.py
owners:
  - cbwinslow
inputs:
  - Claim records
  - grouped evidence
outputs:
  - VerdictResult records
  - human review routing flags
models:
  - verification
prompts:
  - src/factchecker/prompts/verdictdraft.md
python_skills:
  - src/factchecker/skills/verdictskills.py
---

# Verdict

## Purpose
Convert evidence into claim-level judgments that are legible, calibrated, and reviewable.

## Responsibilities
- Draft claim verdicts.
- Calibrate confidence based on evidence quality.
- Route sensitive or weakly supported cases to human review.
- Aggregate job-level outcome summaries.

## Handoffs
Writes `ctx.verdicts` and exposes API- and MCP-friendly verdict payloads.
