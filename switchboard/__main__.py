"""CLI: `python -m switchboard <command>`."""

from __future__ import annotations

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from switchboard.config import settings
from switchboard.ledger import Database, LedgerError, LedgerService
from switchboard.pricing import PriceTable

cli = typer.Typer(add_completion=False, help="Switchboard - local AI model router.")
users_cli = typer.Typer(help="Manage developers and their budgets.")
cli.add_typer(users_cli, name="users")

console = Console()


def _ledger() -> LedgerService:
    database = Database(settings.database_url)
    database.create_all()
    return LedgerService(
        database, PriceTable.load(settings.prices_file), settings.store_prompts
    )


# --- Server ----------------------------------------------------------------


@cli.command()
def serve(
    host: str = typer.Option(settings.host, help="Bind address."),
    port: int = typer.Option(settings.port, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on edits."),
) -> None:
    """Run the OpenAI-compatible proxy."""
    console.print(f"[bold]Switchboard[/bold] -> {settings.ollama_base_url}")
    console.print(f"Default model: [cyan]{settings.default_model}[/cyan]")
    console.print(f"Ledger:        [cyan]{settings.database_url}[/cyan]")
    if settings.store_prompts:
        console.print(
            "[yellow]Prompt text IS being stored[/yellow] (store_prompts=true)"
        )
    console.print(f"Point any OpenAI client at [green]http://{host}:{port}/v1[/green]\n")
    uvicorn.run("switchboard.api:app", host=host, port=port, reload=reload)


@cli.command()
def check() -> None:
    """Verify Ollama is reachable and list the models available as tiers."""
    import httpx

    prices = PriceTable.load(settings.prices_file)
    try:
        response = httpx.get(f"{settings.openai_compat_url}/models", timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        console.print(f"[red]Cannot reach Ollama[/red] at {settings.ollama_base_url}")
        console.print(f"  {exc}")
        raise typer.Exit(code=1) from exc

    models = sorted(m["id"] for m in response.json().get("data", []))
    console.print(f"[green]Ollama reachable[/green] - {len(models)} model(s):")
    for name in models:
        price = prices.for_model(name)
        flags = []
        if name == settings.default_model:
            flags.append("default")
        if name == prices.baseline_model:
            flags.append("baseline")
        suffix = f" [dim]({', '.join(flags)})[/dim]" if flags else ""
        console.print(
            f"  {price.tier:<8} {name:<28} "
            f"[dim]${price.input_per_mtok:.2f}/${price.output_per_mtok:.2f} "
            f"per Mtok (simulated)[/dim]{suffix}"
        )


# --- Users -----------------------------------------------------------------


@users_cli.command("add")
def users_add(
    name: str = typer.Argument(..., help="Developer name, e.g. alice."),
    budget: float = typer.Option(50.0, help="Monthly budget in simulated USD."),
) -> None:
    """Create a developer and print their API key once."""
    try:
        created = _ledger().create_user(name, budget)
    except LedgerError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"\n[green]Created[/green] {created.name} "
                  f"- budget ${created.monthly_budget_usd:.2f}/month (simulated)")
    console.print(f"\n  API key: [bold cyan]{created.api_key}[/bold cyan]\n")
    console.print(
        "[yellow]Save it now.[/yellow] Only a hash is stored, so this key "
        "cannot be shown again. Losing it means creating a new user.\n"
    )


@users_cli.command("list")
def users_list() -> None:
    """Show all developers and their budgets."""
    rows = _ledger().usage()
    if not rows:
        console.print("No users yet. Create one with: switchboard users add <name>")
        return

    table = Table(title="Users")
    table.add_column("Name")
    table.add_column("Budget/month", justify="right")
    table.add_column("Requests (MTD)", justify="right")
    for row in rows:
        table.add_row(row.name, f"${row.budget_usd:.2f}", str(row.requests))
    console.print(table)


@users_cli.command("budget")
def users_budget(
    name: str = typer.Argument(...),
    amount: float = typer.Argument(..., help="New monthly budget in simulated USD."),
) -> None:
    """Change a developer's monthly budget."""
    try:
        _ledger().set_budget(name, amount)
    except LedgerError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]{name}[/green] budget set to ${amount:.2f}/month")


@users_cli.command("deactivate")
def users_deactivate(name: str = typer.Argument(...)) -> None:
    """Block a developer without deleting their history."""
    try:
        _ledger().set_active(name, False)
    except LedgerError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[yellow]{name} deactivated[/yellow]")


@users_cli.command("activate")
def users_activate(name: str = typer.Argument(...)) -> None:
    """Re-enable a deactivated developer."""
    try:
        _ledger().set_active(name, True)
    except LedgerError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]{name} activated[/green]")


# --- Reporting -------------------------------------------------------------


@cli.command()
def usage() -> None:
    """Month-to-date spend and savings per developer."""
    rows = _ledger().usage()
    if not rows:
        console.print("No users yet. Create one with: switchboard users add <name>")
        return

    table = Table(title="Usage this month (all money SIMULATED)")
    table.add_column("User")
    table.add_column("Requests", justify="right")
    table.add_column("Spent", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Saved", justify="right")
    table.add_column("Budget left", justify="right")

    for row in rows:
        remaining = row.remaining_usd
        colour = "red" if remaining <= 0 else "green"
        table.add_row(
            row.name,
            str(row.requests),
            f"${row.spent_usd:.4f}",
            f"${row.baseline_usd:.4f}",
            f"{row.saved_pct:.0f}%",
            f"[{colour}]${remaining:.2f}[/{colour}]",
        )

    console.print(table)
    console.print(
        "[dim]'Baseline' is what these requests would have cost on "
        f"{PriceTable.load(settings.prices_file).baseline_model}. "
        "No real money is involved.[/dim]"
    )


if __name__ == "__main__":
    cli()
