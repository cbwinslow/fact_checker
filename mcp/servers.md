# MCP Servers and config

## Current server
- `mcp/factcheckermcpserver.py`: the primary MCP server entry point for the repository.

## Expected environment variables
- `OPENROUTER_API_KEY`
- `DATABASE_URL`
- `GOOGLE_FACT_CHECK_API_KEY`
- `SERPER_API_KEY`
- `API_SECRET_KEY`

## Recommended next additions
- Add an MCP-focused README section with example client config for Claude Desktop, Cursor, and Windsurf.
- Add JSON schema snapshots for each exposed tool.
- Add a compatibility matrix showing which tools need network access, DB access, or external provider keys.
- Decide whether tool names stay snake_case or mirror existing internal naming exactly.

## Suggested client config pattern
Use the repo's Python environment and launch:

```bash
python -m mcp.factcheckermcpserver
```

Then provide env vars through the client's MCP configuration layer instead of hardcoding secrets in repo files.
