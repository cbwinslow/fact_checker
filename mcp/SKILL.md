---
name: factchecker-mcp-server
version: 1
summary: MCP interface for submitting fact-check jobs, polling status, and reading pipeline outputs.
module: mcp/factcheckermcpserver.py
transport: stdio
owners:
  - cbwinslow
entrypoints:
  - python -m mcp.factcheckermcpserver
provided_tools:
  - submit_job
  - get_job_status
  - get_claims
  - get_verdicts
  - search_evidence
  - extract_claims_text
  - detect_media_type
  - estimate_cost
---

# MCP Server

## Purpose
Expose the fact_checker pipeline to MCP-compatible clients such as editors, agent shells, and desktop assistants.

## Responsibilities
- Accept job submissions.
- Return structured status and results.
- Expose lower-level utility tools for claim extraction and intake estimation.
- Keep request and response shapes stable enough for external agent clients.

## Contract notes
- Transport is stdio unless a future server mode is added.
- Tool names should remain stable or be versioned.
- Authentication and rate limits should be documented in the implementation README.
