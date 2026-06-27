"""mcp - Model Context Protocol server for the fact_checker pipeline.

File: mcp/__init__.py

Exposes the fact_checker pipeline as an MCP-compatible tool server so that
any MCP-aware AI assistant (Claude, Cursor, Windsurf, etc.) can submit
fact-check jobs, poll status, and retrieve results via structured tool calls.

Package contents
----------------
fact_checker_mcp_server  -- Main MCP server entry point (stdio transport)
mcp_tools                -- Tool schemas and handler implementations

Quick start
-----------
    # Install MCP SDK
    pip install mcp

    # Run the server (stdio, for use with Claude Desktop / Cursor)
    python -m mcp.fact_checker_mcp_server

    # Or launch directly
    python mcp/fact_checker_mcp_server.py
"""
