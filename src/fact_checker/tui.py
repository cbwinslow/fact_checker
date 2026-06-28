"""Textual TUI - htop-style fact-checking dashboard.

Layout:
  ┌ Header bar: title + model + keybindings ┐
  ├ URL input + Submit button                ┤
  ├ Pipeline status bar (4 stages)           ┤
  ├ Claims table (live updating)             ┤
  ├ Verdict table (color-coded)              ┤
  ├ Citations panel (expandable)             ┤
  ├ Research graph (evidence flow)           ┤
  └ Footer: stage log                        ┘
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Button,
    Static,
    Log,
    TabbedContent,
    TabPane,
    Tree,
)
from textual.reactive import reactive
from textual import work
from rich.text import Text

from .harness import run_pipeline
from .models import Verdict, PipelineResult, Citation


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


class CitationPanel(Static):
    """Expandable panel showing citations for selected verdict."""
    
    citations: reactive[list] = reactive([])
    
    def render(self) -> str:
        if not self.citations:
            return "[dim]No citations for selected verdict. Click a verdict to view citations.[/]"
        
        lines = ["[bold cyan]Citations[/]\n"]
        for cite in self.citations:
            lines.append(f"[bold]{cite.get('claim_fragment', 'Claim')}[/]")
            lines.append(f"  [blue]{cite.get('quote', 'No quote')}[/]")
            lines.append(f"  [dim]Evidence ID: {cite.get('evidence_id', 'N/A')}[/]\n")
        return "\n".join(lines)
    
    def update_citations(self, citations: list):
        self.citations = citations
        self.refresh()


class ResearchGraph(Tree):
    """Tree visualization of research flow: sub-questions → sources → quotes."""
    
    def show_research(self, research_data: dict):
        """Display research flow."""
        self.clear()
        root = self.root
        root.label = "[bold cyan]Research Flow[/]"
        
        # Sub-questions
        sq_node = root.add("[bold]Sub-Questions[/]")
        for sq in research_data.get("sub_questions", []):
            sq_node.add_leaf(f"• {sq}")
        
        # Sources
        src_node = root.add("[bold]Sources[/]")
        for src in research_data.get("sources", []):
            src_leaf = src_node.add_leaf(f"[blue]{src['domain']}[/] {src['title'][:60]}")
            src_leaf.add_leaf(f"[dim]Type: {src['type']} | Relevance: {src['relevance']:.0%}[/]")
        
        # Quotes
        q_node = root.add("[bold]Key Quotes[/]")
        for q in research_data.get("quotes", []):
            q_node.add_leaf(f'"{q[:100]}..."')
        
        root.expand_all()


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
    #main-panes {
        height: 1fr;
    }
    #left-pane {
        width: 50%;
        border-right: solid $primary-darken-2;
    }
    #right-pane {
        width: 50%;
    }
    #claims-table, #verdict-table {
        height: 1fr;
        border: solid $primary-darken-2;
    }
    #citations-panel {
        height: 12;
        border-top: solid $primary;
        background: $surface-darken-1;
        padding: 1;
    }
    #research-graph {
        height: 20;
        border-top: solid $primary;
        background: $surface-darken-1;
        padding: 1;
    }
    #log-panel {
        height: 8;
        border-top: solid $primary;
        background: $surface-darken-1;
    }
    DataTable {
        background: $surface;
    }
    TabbedContent {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("r", "clear_results", "Clear"),
        Binding("?", "help", "Help"),
        Binding("tab", "focus_next", "Next Pane"),
    ]

    TITLE = "Fact Checker TUI"
    SUB_TITLE = "AI-powered video fact-checking with citations"

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="url-bar"):
            yield Input(placeholder="Paste YouTube URL or video URL...", id="url-input")
            yield Button("Submit ⏎", id="submit-btn", variant="success")

        yield StageBar(id="stage-bar")

        # Main content area with tabbed panes
        with Horizontal(id="main-panes"):
            with Vertical(id="left-pane"):
                with TabbedContent():
                    with TabPane("Claims", id="tab-claims"):
                        yield Label("[bold cyan] Claims[/]", markup=True)
                        yield DataTable(id="claims-table", zebra_stripes=True)
                    with TabPane("Verdicts", id="tab-verdicts"):
                        yield Label("[bold cyan] Verdicts[/]", markup=True)
                        yield DataTable(id="verdict-table", zebra_stripes=True)
            
            with Vertical(id="right-pane"):
                with TabbedContent():
                    with TabPane("Citations", id="tab-citations"):
                        yield CitationPanel(id="citations-panel")
                    with TabPane("Research Graph", id="tab-research"):
                        yield ResearchGraph(id="research-graph")

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

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle verdict row selection to show citations."""
        if event.data_table.id == "verdict-table":
            self._show_citations_for_row(event.row_index)

    def _show_citations_for_row(self, row_index: int) -> None:
        """Show citations for selected verdict row."""
        # In a real implementation, we'd map row_index to the actual verdict
        # For now, just show a placeholder
        citations_panel = self.query_one("#citations-panel", CitationPanel)
        citations_panel.update_citations([
            {"evidence_id": "example-uuid", "quote": "Example quote from source", "claim_fragment": "Example claim fragment"},
            {"evidence_id": "example-uuid-2", "quote": "Another supporting quote", "claim_fragment": "Another claim part"},
        ])
        # Switch to citations tab
        tabbed = self.query_one(TabbedContent)
        tabbed.active = "tab-citations"

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

            # Show research graph if available
            if hasattr(result, 'research') and result.research:
                research_graph = self.query_one("#research-graph", ResearchGraph)
                # Build research data from result
                research_data = {
                    "sub_questions": getattr(result, 'sub_questions', []),
                    "sources": [
                        {
                            "domain": ev.domain,
                            "title": ev.title or ev.snippet[:50],
                            "type": ev.source_type,
                            "relevance": ev.relevance_score,
                        }
                        for ev in result.evidence[:10]
                    ],
                    "quotes": [ev.quote_text for ev in result.evidence if ev.quote_text][:5],
                }
                research_graph.show_research(research_data)

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
        self.query_one("#citations-panel", CitationPanel).update_citations([])
        self.query_one("#research-graph", ResearchGraph).clear()
        self.query_one("#log-panel", Log).write_line(
            f"[{datetime.now():%H:%M:%S}] Cleared results."
        )
