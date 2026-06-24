"""Textual TUI - htop-style fact-checking dashboard.

Layout:
  ┌ Header bar: title + model + keybindings ┐
  ├ URL input + Submit button                ┤
  ├ Pipeline status bar (4 stages)           ┤
  ├ Claims table (live updating)             ┤
  ├ Verdict table (color-coded)              ┤
  └ Footer: stage log                        ┘
"""
from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Button,
    Static,
    Log,
    ProgressBar,
)
from textual.reactive import reactive
from textual import work
from rich.text import Text

from .harness import run_pipeline
from .models import Verdict, PipelineResult


VERDICT_STYLE = {
    Verdict.SUPPORTED:   ("green",  "✓ SUPPORTED"),
    Verdict.REFUTED:     ("red",    "✗ REFUTED"),
    Verdict.MISLEADING:  ("yellow", "⚠ MISLEADING"),
    Verdict.INSUFFICIENT:("dim",    "? INSUFFICIENT"),
    Verdict.UNVERIFIABLE:("blue",   "● UNVERIFIABLE"),
}

STAGE_LABELS = ["Ingest", "Extract", "Evidence", "Verdict"]


class StageBar(Static):
    """4-stage pipeline progress indicator."""

    current_stage: reactive[int] = reactive(-1)

    def render(self) -> str:
        parts = []
        for i, label in enumerate(STAGE_LABELS):
            if i < self.current_stage:
                parts.append(f"[bold green]● {label}[/]")
            elif i == self.current_stage:
                parts.append(f"[bold yellow blink]● {label}...[/]")
            else:
                parts.append(f"[dim]○ {label}[/]")
        return "  →  ".join(parts)


class FactCheckerApp(App):
    """htop-style Textual TUI for fact_checker."""

    CSS = """
    Screen {
        background: $surface;
    }
    #url-bar {
        height: 3;
        padding: 0 1;
        background: $panel;
        border: solid $primary;
    }
    #url-input {
        width: 1fr;
    }
    #submit-btn {
        width: 12;
        background: $success;
    }
    #stage-bar {
        height: 3;
        padding: 1 2;
        background: $panel-darken-1;
        border-bottom: solid $primary;
    }
    #claims-table {
        height: 1fr;
        border: solid $primary-darken-2;
    }
    #verdict-table {
        height: 1fr;
        border: solid $primary-darken-2;
    }
    #log-panel {
        height: 8;
        border-top: solid $primary;
        background: $surface-darken-1;
    }
    DataTable {
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("r", "clear_results", "Clear"),
        Binding("?", "help", "Help"),
    ]

    TITLE = "Fact Checker TUI"
    SUB_TITLE = "AI-powered video fact-checking"

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="url-bar"):
            yield Input(placeholder="Paste YouTube URL or video URL...", id="url-input")
            yield Button("Submit ⏎", id="submit-btn", variant="success")

        yield StageBar(id="stage-bar")

        with Horizontal():
            with Vertical():
                yield Label("[bold cyan] Claims[/]", markup=True)
                yield DataTable(id="claims-table", zebra_stripes=True)

            with Vertical():
                yield Label("[bold cyan] Verdicts[/]", markup=True)
                yield DataTable(id="verdict-table", zebra_stripes=True)

        yield Log(id="log-panel", highlight=True, max_lines=200)
        yield Footer()

    def on_mount(self) -> None:
        self._setup_claims_table()
        self._setup_verdict_table()
        self.query_one("#log-panel", Log).write_line(
            f"[{datetime.now():%H:%M:%S}] Fact Checker TUI ready. Paste a URL and press Submit."
        )

    def _setup_claims_table(self) -> None:
        table = self.query_one("#claims-table", DataTable)
        table.add_columns("#", "Claim", "Checkable", "Confidence")

    def _setup_verdict_table(self) -> None:
        table = self.query_one("#verdict-table", DataTable)
        table.add_columns("Verdict", "Conf", "Claim (truncated)", "Review?")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit-btn":
            url = self.query_one("#url-input", Input).value.strip()
            if url:
                self._run_pipeline(url)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        url = event.value.strip()
        if url:
            self._run_pipeline(url)

    @work(exclusive=True, thread=False)
    async def _run_pipeline(self, url: str) -> None:
        log = self.query_one("#log-panel", Log)
        stage_bar = self.query_one("#stage-bar", StageBar)
        claims_table = self.query_one("#claims-table", DataTable)
        verdict_table = self.query_one("#verdict-table", DataTable)

        claims_table.clear()
        verdict_table.clear()

        log.write_line(f"[{datetime.now():%H:%M:%S}] Submitting: {url}")
        stage_bar.current_stage = 0

        try:
            result: PipelineResult = await run_pipeline(url=url)

            # Populate claims table
            stage_bar.current_stage = 1
            for i, claim in enumerate(result.claims, 1):
                claims_table.add_row(
                    str(i),
                    claim.text[:70] + "..." if len(claim.text) > 70 else claim.text,
                    "✓" if claim.is_checkable else "✗",
                    f"{claim.confidence:.0%}",
                )
            log.write_line(f"[{datetime.now():%H:%M:%S}] {len(result.claims)} claims extracted.")

            # Populate verdicts table
            stage_bar.current_stage = 3
            claim_map = {c.id: c for c in result.claims}
            for v in result.verdicts:
                claim = claim_map.get(v.claim_id)
                claim_text = claim.text[:55] + "..." if claim and len(claim.text) > 55 else (claim.text if claim else "")
                color, label = VERDICT_STYLE.get(v.verdict, ("white", v.verdict.value))
                verdict_table.add_row(
                    Text(label, style=color),
                    f"{v.confidence:.0%}",
                    claim_text,
                    Text("YES", style="red bold") if v.requires_human_review else Text("no", style="dim"),
                )

            stage_bar.current_stage = 4
            log.write_line(
                f"[{datetime.now():%H:%M:%S}] Done. "
                f"{len(result.verdicts)} verdicts | "
                f"Status: {result.job.status.value}"
            )

        except Exception as e:
            stage_bar.current_stage = -1
            log.write_line(f"[{datetime.now():%H:%M:%S}] ERROR: {e}")

    def action_clear_results(self) -> None:
        self.query_one("#claims-table", DataTable).clear()
        self.query_one("#verdict-table", DataTable).clear()
        self.query_one("#stage-bar", StageBar).current_stage = -1
        self.query_one("#log-panel", Log).write_line(
            f"[{datetime.now():%H:%M:%S}] Cleared results."
        )
