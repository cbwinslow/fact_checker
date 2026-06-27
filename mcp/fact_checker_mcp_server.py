"""fact_checker_mcp_server.py - MCP server entry point for the fact-checker pipeline.

File: mcp/fact_checker_mcp_server.py

Exposes the fact_checker pipeline as a Model Context Protocol (MCP) server
using stdio transport. Any MCP-aware AI host (Claude Desktop, Cursor,
Windsurf, Continue, etc.) can connect to this server and invoke fact-checking
tools as structured function calls.

Available MCP tools
-------------------
submit_job          -- Submit a URL or file path for fact-checking.
get_job_status      -- Poll the status of a running or completed job.
get_claims          -- Retrieve extracted claims for a completed job.
get_verdicts        -- Retrieve verdicts for a completed job.
search_evidence     -- Search for evidence for a free-form claim text.
extract_claims_text -- Extract claims from a raw text string (no job).
detect_media_type   -- Detect the media type of a file path or URL.
estimate_cost       -- Estimate processing cost for a given input.

Usage
-----
    # stdio mode (Claude Desktop / Cursor)
    python mcp/fact_checker_mcp_server.py

    # Claude Desktop config snippet:
    {
      "mcpServers": {
        "fact_checker": {
          "command": "python",
          "args": ["/path/to/fact_checker/mcp/fact_checker_mcp_server.py"],
          "env": {
            "OPENROUTER_API_KEY": "your-key-here",
            "DATABASE_URL": "postgresql+asyncpg://user:pass@localhost/fact_checker"
          }
        }
      }
    }

Dependencies
------------
    pip install mcp httpx asyncpg sqlalchemy[asyncio]
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup: ensure repo root and src/ are importable regardless of cwd.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import mcp.server.stdio
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.types import TextContent, Tool
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    print(
        "[fact_checker_mcp] ERROR: 'mcp' package not found. "
        "Install with: pip install mcp",
        file=sys.stderr,
    )

from mcp.mcp_tools import (
    TOOL_DEFINITIONS,
    handle_submit_job,
    handle_get_job_status,
    handle_get_claims,
    handle_get_verdicts,
    handle_search_evidence,
    handle_extract_claims_text,
    handle_detect_media_type,
    handle_estimate_cost,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("fact_checker_mcp")

# Tool name -> async handler
_HANDLERS: dict[str, Any] = {
    "submit_job": handle_submit_job,
    "get_job_status": handle_get_job_status,
    "get_claims": handle_get_claims,
    "get_verdicts": handle_get_verdicts,
    "search_evidence": handle_search_evidence,
    "extract_claims_text": handle_extract_claims_text,
    "detect_media_type": handle_detect_media_type,
    "estimate_cost": handle_estimate_cost,
}


async def run_server() -> None:
    """Initialise and run the MCP server over stdio transport.

    Blocks until the parent process closes stdin. Registers all tool
    definitions and routes incoming tool/call requests to the appropriate
    handler function.
    """
    if not _MCP_AVAILABLE:
        log.error("Cannot start: 'mcp' package not installed.")
        sys.exit(1)

    server = Server("fact_checker")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return all tools this MCP server exposes."""
        return [Tool(**td) for td in TOOL_DEFINITIONS]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Dispatch an MCP tool call to the appropriate handler.

        Args:
            name:      Tool name from the MCP request.
            arguments: Keyword arguments dict from the MCP request.

        Returns:
            List with a single TextContent containing the JSON result.
        """
        handler = _HANDLERS.get(name)
        if handler is None:
            return [TextContent(type="text", text=json.dumps(
                {"error": f"Unknown tool: {name}"}
            ))]
        try:
            log.info("[mcp] tool=%s args=%s", name, list(arguments.keys()))
            result = await handler(**arguments)
            return [TextContent(type="text", text=json.dumps(result, default=str))]
        except Exception as exc:
            log.exception("[mcp] tool=%s raised: %s", name, exc)
            return [TextContent(type="text", text=json.dumps(
                {"error": str(exc), "tool": name}
            ))]

    log.info("[fact_checker_mcp] Server starting (stdio)...")
    async with mcp.server.stdio.stdio_server() as (r, w):
        await server.run(
            r, w,
            InitializationOptions(
                server_name="fact_checker",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(run_server())
