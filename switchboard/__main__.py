"""CLI: `python -m switchboard <command>`."""

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from switchboard.config import settings
from switchboard.ledger import Database, LedgerError, LedgerService
from switchboard.pricing import PriceTable

cli = typer.Typer(add_completion=False, help="Switchboard - local AI model router.")
users_cli = typer.Typer(help="Manage developers and their budgets.")
eval_cli = typer.Typer(help="Measure routing strategies against known answers.")
cli.add_typer(users_cli, name="users")
cli.add_typer(eval_cli, name="eval")

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

    prices = PriceTable.load(settings.prices_file)
    names = [n.strip() for n in strategies.split(",") if n.strip()]

    try:
        chosen = [build_strategy(name, prices) for name in names]
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
    console.print(f"Ladder: {' -> '.join(prices.ladder)}\n")

    runner = EvalRunner(settings, prices, max_tokens=max_tokens)
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
            console.print(f"\n[yellow]Interrupted. Partial results in {output}[/yellow]")
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
    table.add_column("Format misses", justify="right")

    for s in summaries:
        table.add_row(
            s.strategy,
            f"{s.accuracy:.1f}% ({s.correct}/{s.tasks})",
            f"${s.cost_usd:.4f}",
            f"{s.saved_vs_baseline_pct:.1f}%",
            f"{s.avg_latency_ms} ms",
            str(s.model_switches),
            str(s.format_failures),
        )
    console.print(table)

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
