"""CLI: `python -m switchboard <command>`."""

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from switchboard import schema
from switchboard.catalog import CatalogError, ModelCatalog
from switchboard.config import settings
from switchboard.ledger import Database, LedgerError, LedgerService
from switchboard.providers import LocalOnlyViolation, ProviderPool
from switchboard.schema import SchemaOutOfDate, require_up_to_date

cli = typer.Typer(add_completion=False, help="Switchboard - local AI model router.")
users_cli = typer.Typer(help="Manage developers and their budgets.")
eval_cli = typer.Typer(help="Measure routing strategies against known answers.")
db_cli = typer.Typer(help="Database schema management.")
cli.add_typer(users_cli, name="users")
cli.add_typer(eval_cli, name="eval")
cli.add_typer(db_cli, name="db")

console = Console()


def _ledger(require_schema: bool = True) -> LedgerService:
    if require_schema:
        try:
            require_up_to_date(settings.database_url)
        except SchemaOutOfDate as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    database = Database(settings.database_url)
    return LedgerService(
        database, ModelCatalog.load(settings.providers_file), settings.store_prompts
    )


# --- Database --------------------------------------------------------------


@db_cli.command("status")
def db_status() -> None:
    """Show whether the database schema matches this version of the code."""
    state = schema.status(settings.database_url)
    colour = "green" if state.up_to_date else "yellow"
    console.print(f"Database: [cyan]{settings.database_url}[/cyan]")
    console.print(f"Schema:   [{colour}]{state.describe()}[/{colour}]")
    if not state.up_to_date:
        console.print(
            "\nRun [cyan]switchboard db upgrade[/cyan] to bring it up to date."
        )


@db_cli.command("upgrade")
def db_upgrade() -> None:
    """Create or update the database schema.

    Safe to run repeatedly - migrations already applied are skipped.
    """
    before = schema.status(settings.database_url)
    if before.up_to_date:
        console.print(f"[green]Already up to date[/green] (revision {before.head})")
        return

    console.print(f"Upgrading {settings.database_url}")
    console.print(f"  {before.describe()}")
    schema.upgrade(settings.database_url)
    console.print(f"[green]Done[/green] - now at revision {schema.head_revision()}")


@db_cli.command("stamp-baseline")
def db_stamp_baseline() -> None:
    """Mark an existing pre-migrations database as being at the first revision.

    For databases created before migrations existed. Their tables are already
    the right shape, so running the first migration would fail on "table
    already exists". This records the version without touching any data.
    """
    state = schema.status(settings.database_url)
    if not state.unmanaged:
        console.print(
            f"[yellow]Nothing to do[/yellow] - already stamped at {state.current}."
        )
        return

    schema.stamp(settings.database_url, "0001")
    console.print(
        f"[green]Stamped[/green] {settings.database_url} at revision 0001.\n"
        "Now run [cyan]switchboard db upgrade[/cyan] to apply anything newer."
    )


# --- Server ----------------------------------------------------------------


@cli.command()
def serve(
    host: str = typer.Option(settings.host, help="Bind address."),
    port: int = typer.Option(settings.port, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on edits."),
) -> None:
    """Run the OpenAI-compatible proxy."""
    catalog = _catalog()
    enabled = ", ".join(p.id for p in catalog.enabled_providers()) or "(none)"

    console.print("[bold]Switchboard[/bold]")
    console.print(f"Providers:     [cyan]{enabled}[/cyan]")
    console.print(f"Default model: [cyan]{settings.default_model}[/cyan]")
    console.print(f"Ledger:        [cyan]{settings.database_url}[/cyan]")
    if settings.local_only:
        console.print("[green]Local-only mode ON[/green] - no prompt leaves this host")
    if settings.store_prompts:
        console.print(
            "[yellow]Prompt text IS being stored[/yellow] (store_prompts=true)"
        )
    console.print(f"Point any OpenAI client at [green]http://{host}:{port}/v1[/green]\n")
    uvicorn.run("switchboard.api:app", host=host, port=port, reload=reload)


def _catalog() -> ModelCatalog:
    try:
        return ModelCatalog.load(settings.providers_file)
    except CatalogError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@cli.command()
def providers() -> None:
    """Show configured providers and the models they offer."""
    catalog = _catalog()
    console.print(f"Catalog: [cyan]{settings.providers_file}[/cyan]")
    if settings.local_only:
        console.print("[yellow]Local-only mode is ON[/yellow] - remote providers "
                      "will be refused at startup.\n")

    table = Table(title="Providers")
    table.add_column("Provider")
    table.add_column("Type")
    table.add_column("Endpoint")
    table.add_column("Models", justify="right")
    table.add_column("State")

    for spec in catalog.providers.values():
        if not spec.enabled:
            state = "[dim]disabled[/dim]"
        elif not spec.key_is_available:
            state = f"[red]no key ({spec.api_key_env})[/red]"
        elif spec.is_local:
            state = "[green]enabled, local[/green]"
        else:
            state = "[green]enabled, remote[/green]"

        table.add_row(
            spec.id, spec.type, spec.base_url, str(len(spec.model_ids)), state
        )
    console.print(table)

    models = Table(title="Models")
    models.add_column("Tier")
    models.add_column("Model")
    models.add_column("Provider")
    models.add_column("In $/Mtok", justify="right")
    models.add_column("Out $/Mtok", justify="right")
    models.add_column("Notes")

    for model_id in catalog.known_models():
        spec = catalog.models[model_id]
        notes = []
        if model_id in catalog.ladder:
            notes.append(f"ladder #{catalog.ladder.index(model_id)}")
        if model_id == catalog.baseline_model:
            notes.append("baseline")
        if model_id == settings.default_model:
            notes.append("default")
        if spec.emits_thinking:
            notes.append("thinking")
        models.add_row(
            spec.tier,
            model_id,
            spec.provider_id,
            f"{spec.input_per_mtok:.2f}",
            f"{spec.output_per_mtok:.2f}",
            ", ".join(notes),
        )
    console.print(models)

    if catalog.has_simulated_pricing:
        console.print(
            "[yellow]Some enabled providers use SIMULATED pricing.[/yellow] "
            "Cost and savings figures are illustrative, not real money."
        )


@cli.command()
def check() -> None:
    """Verify every enabled provider is reachable."""
    import asyncio

    catalog = _catalog()
    try:
        pool = ProviderPool(catalog, local_only=settings.local_only)
    except LocalOnlyViolation as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    # Read everything off the pool before closing it: aclose() drops the
    # provider clients, and available_models() is derived from them.
    available = pool.available_models()
    unconfigured = pool.unconfigured()

    async def probe() -> dict[str, bool]:
        try:
            return await pool.health()
        finally:
            await pool.aclose()

    health = asyncio.run(probe())

    if not health:
        console.print(
            "[red]No usable providers.[/red] Enable one in providers.yaml and "
            "make sure its API key environment variable is set."
        )
        raise typer.Exit(code=1)

    failures = 0
    for provider_id, healthy in sorted(health.items()):
        spec = catalog.providers[provider_id]
        if healthy:
            console.print(f"  [green]OK  [/green] {provider_id:<16} {spec.base_url}")
        else:
            failures += 1
            console.print(f"  [red]DOWN[/red] {provider_id:<16} {spec.base_url}")

    for provider_id, reason in unconfigured.items():
        console.print(f"  [yellow]SKIP[/yellow] {provider_id:<16} {reason}")

    console.print(f"\n{len(available)} model(s) available: {', '.join(available)}")

    if failures:
        raise typer.Exit(code=1)


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
        f"{_catalog().baseline_model}. "
        "No real money is involved.[/dim]"
    )


# --- Evaluation ------------------------------------------------------------


@eval_cli.command("tasks")
def eval_tasks(
    taskset: str = typer.Option("builtin", help="Task set name."),
) -> None:
    """Show what is in a task set."""
    from eval.datasets import load_taskset

    tasks = load_taskset(taskset)
    counts = tasks.counts_by_difficulty()
    console.print(f"[bold]{tasks.name}[/bold] - {len(tasks)} tasks")
    console.print(
        "  " + "  ".join(f"{level}: {n}" for level, n in counts.items())
    )
    categories = sorted({t.category for t in tasks})
    console.print(f"  categories: {', '.join(categories)}")


@eval_cli.command("run")
def eval_run(
    taskset: str = typer.Option("builtin", help="Task set name."),
    strategies: str = typer.Option(
        "always-cheap,always-expensive,random,keyword",
        help="Comma-separated strategy names.",
    ),
    limit: int = typer.Option(0, help="Only run the first N tasks (0 = all)."),
    difficulty: str = typer.Option("", help="Only run one difficulty level."),
    max_tokens: int = typer.Option(600, help="Cap on generated tokens per task."),
    out: str = typer.Option("runs/latest.jsonl", help="Where to stream results."),
) -> None:
    """Run task sets through strategies and record every outcome.

    Slow by design on this hardware - the top tier does not fit in VRAM. Results
    stream to disk as they complete, so an interrupted run is not lost.
    """
    import asyncio

    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from eval.datasets import load_taskset
    from eval.report import summarise, to_markdown
    from eval.runner import EvalRunner
    from switchboard.routing import build_strategy

    catalog = _catalog()
    names = [n.strip() for n in strategies.split(",") if n.strip()]

    try:
        chosen = [build_strategy(name, catalog) for name in names]
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    tasks = load_taskset(taskset).filtered(
        difficulty=difficulty or None, limit=limit or None
    )
    if not len(tasks):
        console.print("[red]No tasks matched those filters.[/red]")
        raise typer.Exit(code=1)

    total = len(tasks) * len(chosen)
    console.print(
        f"Running [cyan]{len(tasks)}[/cyan] tasks x "
        f"[cyan]{len(chosen)}[/cyan] strategies = [bold]{total}[/bold] generations"
    )
    console.print(f"Ladder: {' -> '.join(catalog.ladder)}\n")

    runner = EvalRunner(settings, catalog, max_tokens=max_tokens)
    output = Path(out)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("evaluating", total=total)

        def tick(result) -> None:
            mark = "[green]ok[/green]" if result.correct else "[red]x [/red]"
            progress.update(
                bar,
                advance=1,
                description=f"{mark} {result.strategy}/{result.task_id} "
                f"[dim]{result.model}[/dim]",
            )

        try:
            results, _ = asyncio.run(runner.run(tasks, chosen, output, progress=tick))
        except KeyboardInterrupt:
            console.print(
                f"\n[yellow]Interrupted. Partial results in {output}[/yellow]"
            )
            raise typer.Exit(code=130) from None

    console.print(f"\n[green]Done.[/green] Raw results: {output}\n")
    console.print(to_markdown(summarise(results)))
    console.print(
        f"\nRender the full report with: "
        f"[cyan]switchboard eval report --run {output}[/cyan]"
    )


@eval_cli.command("report")
def eval_report(
    run: str = typer.Option("runs/latest.jsonl", help="Run file to summarise."),
    out_dir: str = typer.Option("results", help="Where to write report artefacts."),
    plot: bool = typer.Option(True, help="Also render the cost/accuracy chart."),
) -> None:
    """Summarise a run: comparison table, per-tier breakdown, and chart."""
    from eval.report import pareto_plot, summarise, to_markdown
    from eval.runner import load_results

    run_path = Path(run)
    if not run_path.exists():
        console.print(f"[red]No run file at {run_path}[/red]")
        raise typer.Exit(code=1)

    results, metadata = load_results(run_path)
    if not results:
        console.print("[red]That run file contains no results.[/red]")
        raise typer.Exit(code=1)

    summaries = summarise(results)

    table = Table(title="Strategy comparison (money SIMULATED)")
    table.add_column("Strategy")
    table.add_column("Accuracy", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Saved", justify="right")
    table.add_column("Avg latency", justify="right")
    table.add_column("Switches", justify="right")
    table.add_column("Marked", justify="right")
    table.add_column("Risky", justify="right")

    for s in summaries:
        table.add_row(
            s.strategy,
            f"{s.accuracy:.1f}% ({s.correct}/{s.tasks})",
            f"${s.cost_usd:.4f}",
            f"{s.saved_vs_baseline_pct:.1f}%",
            f"{s.avg_latency_ms} ms",
            str(s.model_switches),
            f"{s.marked_answers}/{s.tasks}",
            str(s.risky_extractions),
        )
    console.print(table)
    console.print(
        "[dim]Marked = used the ANSWER: format. Risky = answer mined out of "
        "prose, where a grading mistake is plausible.[/dim]"
    )

    breakdown = Table(title="Accuracy by difficulty")
    breakdown.add_column("Strategy")
    for level in ("easy", "medium", "hard"):
        breakdown.add_column(level, justify="right")
    for s in summaries:
        cells = []
        for level in ("easy", "medium", "hard"):
            correct, total = s.accuracy_by_difficulty.get(level, (0, 0))
            cells.append(f"{correct}/{total}" if total else "-")
        breakdown.add_row(s.strategy, *cells)
    console.print(breakdown)

    for s in summaries:
        usage = ", ".join(
            f"{model} x{count}" for model, count in sorted(s.model_usage.items())
        )
        console.print(f"[dim]{s.strategy}: {usage}[/dim]")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "comparison.md").write_text(
        to_markdown(summaries) + "\n", encoding="utf-8"
    )
    console.print(f"\nWrote [cyan]{out_path / 'comparison.md'}[/cyan]")

    if plot:
        chart = pareto_plot(summaries, out_path / "cost_vs_accuracy.png")
        console.print(f"Wrote [cyan]{chart}[/cyan]")

    if metadata:
        console.print(
            f"\n[dim]Run: {metadata.get('task_set')}, "
            f"max_tokens={metadata.get('max_tokens')}, "
            f"temperature={metadata.get('temperature')}, "
            f"started {metadata.get('started_at')}[/dim]"
        )


if __name__ == "__main__":
    cli()
