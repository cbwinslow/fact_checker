"""Root main.py — thin shim that delegates to the package CLI entry point.

Usage:
    python main.py                      # starts the FastAPI server
    python main.py check <url>          # run a fact-check on a URL (async)
"""
from __future__ import annotations

import asyncio
import sys
import uvicorn


def main() -> None:
    """Run the FastAPI server or a one-shot CLI fact-check."""
    if len(sys.argv) >= 3 and sys.argv[1] == "check":
        # One-shot fact-check mode: python main.py check <url>
        import json
        from fact_checker.harness import run_pipeline

        url = sys.argv[2]
        print(f"[main] Running fact-check on: {url}")
        result = asyncio.run(run_pipeline(url=url))
        print(json.dumps(
            {
                "job_id":  str(result.job.id),
                "status":  result.job.status.value,
                "claims":  len(result.claims),
                "verdicts": len(result.verdicts),
            },
            indent=2,
        ))
    else:
        # Default: start the FastAPI server
        from fact_checker.config import settings
        uvicorn.run(
            "fact_checker.api:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=False,
            log_level=settings.log_level.lower(),
        )


if __name__ == "__main__":
    main()
