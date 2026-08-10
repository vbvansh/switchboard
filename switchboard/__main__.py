"""CLI entry point: `python -m switchboard serve`."""

from __future__ import annotations

import typer
import uvicorn
from rich.console import Console

from switchboard.config import settings

cli = typer.Typer(add_completion=False, help="Switchboard - local AI model router.")
console = Console()


@cli.command()
def serve(
    host: str = typer.Option(settings.host, help="Bind address."),
    port: int = typer.Option(settings.port, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on edits."),
) -> None:
    """Run the OpenAI-compatible proxy."""
    console.print(f"[bold]Switchboard[/bold] -> {settings.ollama_base_url}")
    console.print(f"Default model: [cyan]{settings.default_model}[/cyan]")
    console.print(f"Point any OpenAI client at [green]http://{host}:{port}/v1[/green]\n")
    uvicorn.run("switchboard.api:app", host=host, port=port, reload=reload)


@cli.command()
def check() -> None:
    """Verify Ollama is reachable and list the models available as tiers."""
    import httpx

    url = f"{settings.openai_compat_url}/models"
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        console.print(f"[red]Cannot reach Ollama[/red] at {settings.ollama_base_url}")
        console.print(f"  {exc}")
        raise typer.Exit(code=1)

    models = [m["id"] for m in response.json().get("data", [])]
    console.print(f"[green]Ollama reachable[/green] - {len(models)} model(s):")
    for name in sorted(models):
        marker = " [cyan](default)[/cyan]" if name == settings.default_model else ""
        console.print(f"  - {name}{marker}")


if __name__ == "__main__":
    cli()
