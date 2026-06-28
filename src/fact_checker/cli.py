"""Typer CLI for fact_checker.

Commands:
  fact-checker submit <url>   - Run full pipeline on a URL
  fact-checker file <path>    - Run pipeline on a local video file
  fact-checker jobs           - List recent jobs
  fact-checker serve          - Start FastAPI server
  fact-checker tui            - Launch Textual TUI dashboard
"""
from __future__ import annotations
import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

from .config import settings
from .harness import run_pipeline
from .models import Verdict

app = typer.Typer(
    name="fact-checker",
    help="AI-powered video fact-checking pipeline",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

logging.basicConfig(level=settings.log_level)


VERDICT_COLORS = {
    Verdict.SUPPORTED: "green",
    Verdict.REFUTED: "red",
    Verdict.MISLEADING: "yellow",
    Verdict.INSUFFICIENT: "dim white",
    Verdict.UNVERIFIABLE: "blue",
}

VERDICT_ICONS = {
    Verdict.SUPPORTED: "✅",
    Verdict.REFUTED: "❌",
    Verdict.MISLEADING: "⚠️ ",
    Verdict.INSUFFICIENT: "❓",
    Verdict.UNVERIFIABLE: "🔹",
}


def _print_results(result) -> None:
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Job ID:[/] {result.job.id}\n"
        f"[bold cyan]Status:[/] {result.job.status.value}\n"
        f"[bold cyan]Ingest:[/] {result.job.ingest_source}\n"
        f"[bold cyan]Segments:[/] {len(result.segments)}  "
        f"[bold cyan]Claims:[/] {len(result.claims)}  "
        f"[bold cyan]Verdicts:[/] {len(result.verdicts)}",
        title="[bold white on blue] FACT CHECK RESULTS [/]",
        border_style="blue",
    ))
    console.print()

    if not result.verdicts:
        console.print("[dim]No verdicts generated.[/dim]")
        return

    table = Table(
        show_header=True,
        header_style="bold white",
        box=box.ROUNDED,
        border_style="dim",
        expand=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Verdict", width=14)
    table.add_column("Conf", width=6)
    table.add_column("Claim", ratio=3)
    table.add_column("Explanation", ratio=4)
    table.add_column("Review", width=6)

    claim_map = {c.id: c for c in result.claims}

    for i, v in enumerate(result.verdicts, 1):
        claim = claim_map.get(v.claim_id)
        claim_text = claim.text[:80] + "..." if claim and len(claim.text) > 80 else (claim.text if claim else "")
        color = VERDICT_COLORS.get(v.verdict, "white")
        icon = VERDICT_ICONS.get(v.verdict, "")
        verdict_label = f"{icon} {v.verdict.value}"
        review_flag = "[red]YES[/red]" if v.requires_human_review else "[dim]no[/dim]"
        table.add_row(
            str(i),
            Text(verdict_label, style=color),
            f"{v.confidence:.0%}",
            claim_text,
            v.explanation[:120] + "..." if len(v.explanation) > 120 else v.explanation,
            review_flag,
        )

    console.print(table)
    console.print()


@app.command()
def submit(
    url: str = typer.Argument(..., help="YouTube URL or any video URL"),
):
    """[cyan]Submit a video URL for fact-checking.[/cyan]"""
    console.print(f"[bold green]Submitting:[/] {url}")
    result = asyncio.run(run_pipeline(url=url))
    _print_results(result)


@app.command()
def file(
    path: Path = typer.Argument(..., help="Path to local video file"),
):
    """[cyan]Fact-check a local video file.[/cyan]"""
    if not path.exists():
        console.print(f"[red]File not found:[/] {path}")
        raise typer.Exit(1)
    console.print(f"[bold green]Processing file:[/] {path}")
    result = asyncio.run(run_pipeline(local_path=path))
    _print_results(result)


@app.command()
def serve(
    host: str = typer.Option(settings.api_host, help="API host"),
    port: int = typer.Option(settings.api_port, help="API port"),
):
    """[cyan]Start the FastAPI server.[/cyan]"""
    import uvicorn
    console.print(f"[bold green]Starting API server[/] at http://{host}:{port}")
    uvicorn.run("fact_checker.api:app", host=host, port=port, reload=True)


@app.command()
def tui():
    """[cyan]Launch the Textual TUI dashboard.[/cyan]"""
    from .tui import FactCheckerApp
    FactCheckerApp().run()


if __name__ == "__main__":
    app()
