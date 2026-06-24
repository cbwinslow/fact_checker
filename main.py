import typer

from fact_checker.orchestrator.harness import FactCheckHarness

app = typer.Typer(help="Lightweight video fact-checking harness")


@app.command()
def submit(source: str) -> None:
    result = FactCheckHarness().run(source)
    typer.echo(result)
