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
bench_cli = typer.Typer(help="Public routing benchmarks: real models, real costs.")
router_cli = typer.Typer(help="Train and inspect the live routing model.")
cli.add_typer(users_cli, name="users")
cli.add_typer(eval_cli, name="eval")
cli.add_typer(db_cli, name="db")
cli.add_typer(bench_cli, name="bench")
cli.add_typer(router_cli, name="router")

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
    graceful_timeout: int = typer.Option(
        60,
        help="Seconds to let in-flight requests finish on shutdown. A slow "
        "local model can take minutes, so the usual 10s cuts real answers off.",
    ),
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
    uvicorn.run(
        "switchboard.api:app",
        host=host,
        port=port,
        reload=reload,
        timeout_graceful_shutdown=graceful_timeout,
    )


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


def _require_eval() -> None:
    """Fail clearly when the evaluation extras are not installed.

    The Docker image ships runtime dependencies only - the evaluation harness
    pulls in a large scientific stack that a server never uses. A missing
    import should explain that, not produce a traceback.
    """
    try:
        import eval  # noqa: F401
    except ImportError as exc:
        console.print(
            "[red]The evaluation harness is not available in this "
            "installation.[/red]\n"
            "It is excluded from the container image because it needs a large "
            "set of extra packages.\n"
            "Install it with: [cyan]pip install -r requirements-dev.txt[/cyan]"
        )
        raise typer.Exit(code=1) from exc


@eval_cli.command("tasks")
def eval_tasks(
    taskset: str = typer.Option("builtin", help="Task set name."),
) -> None:
    """Show what is in a task set."""
    _require_eval()
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
    _require_eval()
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
    _require_eval()
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


def _score_cascades(test_grid, train_grid, texts, levels, feature_mode, seed):
    """Score cascade strategies on the held-out split.

    Every cascade result is priced with `score_for_paths`, which charges for
    each call made rather than only the final one. A cascade that escalates has
    genuinely paid twice.
    """
    from eval.benchmarks import cascade as cascade_mod
    from eval.benchmarks.features import FeatureExtractor
    from eval.benchmarks.replay import ReplayResult

    results = []

    def as_result(name, paths):
        scored = test_grid.score_for_paths(paths)
        usage = {}
        for model in scored.pop("final_models"):
            usage[model] = usage.get(model, 0) + 1
        return ReplayResult(
            strategy=name,
            accuracy=scored["accuracy"],
            cost_usd=scored["cost_usd"],
            mean_latency_s=scored["mean_latency_s"],
            n_queries=scored["n_queries"],
            model_usage=usage,
        )

    if cascade_mod.has_answers(test_grid):
        try:
            paths = cascade_mod.agreement_paths(test_grid)
            results.append(as_result("cascade-agree", paths))
        except ValueError as exc:
            console.print(f"[yellow]Agreement cascade skipped: {exc}[/yellow]")
    else:
        console.print(
            "[dim]Agreement cascade skipped: this source records no parsed "
            "answers to compare between models.[/dim]"
        )

    template = cascade_mod.VerifierCascade.train(
        train_grid, texts, FeatureExtractor(mode=feature_mode), seed=seed
    )
    for variant in cascade_mod.cascades_for_thresholds(template, levels):
        results.append(as_result(variant.name, variant.paths(test_grid, texts)))

    return results


def _explain_empty_grid(frame) -> None:
    """Say WHICH models are shrinking the grid, not just that it is empty."""
    counts = frame.why_grid_is_empty()
    console.print(
        "\n[yellow]No question was answered by every selected model.[/yellow]\n"
        "These benchmarks ship several question splits and not every model ran "
        "on all of them, so the overlap can be empty.\n"
    )
    table = Table(title="Questions attempted, per model")
    table.add_column("Model")
    table.add_column("Questions", justify="right")
    for model, n in counts.items():
        colour = "red" if n < counts.max() else "green"
        table.add_row(str(model), f"[{colour}]{int(n):,}[/{colour}]")
    console.print(table)

    keep = counts[counts == counts.max()].index.tolist()
    console.print(
        f"\nTry dropping the low-coverage models. These {len(keep)} share the "
        f"largest split:\n  [cyan]{','.join(keep)}[/cyan]"
    )


# --- Benchmarks ------------------------------------------------------------


@bench_cli.command("build")
def bench_build(
    source: str = typer.Argument(
        ..., help="llmrouterbench | xroutebench | all"
    ),
    rebuild: bool = typer.Option(False, help="Rebuild even if already cached."),
) -> None:
    """Normalise a downloaded benchmark into the fast Parquet cache.

    Reading the raw sources takes minutes; the cache loads in seconds. Phase C
    re-runs experiments constantly, so this is done once up front.
    """
    _require_eval()
    from eval import benchmarks

    wanted = list(benchmarks.SOURCES) if source == "all" else [source]

    for name in wanted:
        if benchmarks.is_cached(name) and not rebuild:
            console.print(f"[dim]{name}: already cached (use --rebuild to redo)[/dim]")
            continue

        console.print(f"Building [cyan]{name}[/cyan] ...")
        counted = {"n": 0}

        def tick(label: str, count: int, total=counted) -> None:
            # `total` is bound as a default so the closure captures this
            # iteration's dict, not whatever the loop variable holds later.
            total["n"] += count
            console.print(f"  {label:<28} {count:>8,} rows")

        try:
            path = benchmarks.build(name, progress=tick)
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

        size_mb = path.stat().st_size / 1e6
        console.print(
            f"[green]{name}[/green]: {counted['n']:,} rows -> "
            f"{path} ({size_mb:.1f} MB)\n"
        )


@bench_cli.command("list")
def bench_list() -> None:
    """Show which benchmarks are cached and what they contain."""
    _require_eval()
    from eval import benchmarks

    table = Table(title="Cached benchmarks")
    table.add_column("Source")
    table.add_column("Rows", justify="right")
    table.add_column("Models", justify="right")
    table.add_column("Suites", justify="right")
    table.add_column("Latency?")

    any_cached = False
    for name in benchmarks.SOURCES:
        if not benchmarks.is_cached(name):
            table.add_row(name, "-", "-", "-", "[dim]not built[/dim]")
            continue
        any_cached = True
        frame = benchmarks.load(name)
        table.add_row(
            name,
            f"{len(frame):,}",
            str(len(frame.models)),
            str(len(frame.benchmarks)),
            "[green]yes[/green]" if frame.has_latency else "[dim]no[/dim]",
        )
    console.print(table)

    if not any_cached:
        console.print(
            "\nNothing cached yet. Download a source, then run:\n"
            "  [cyan]python scripts/fetch_llmrouterbench.py --extract[/cyan]\n"
            "  [cyan]switchboard bench build all[/cyan]"
        )


@bench_cli.command("summary")
def bench_summary(
    source: str = typer.Argument(..., help="Which cached source to summarise."),
    by: str = typer.Option("benchmark", help="benchmark | model"),
    top: int = typer.Option(25, help="Rows to show."),
) -> None:
    """Break a cached benchmark down by suite or by model."""
    _require_eval()
    from eval import benchmarks

    if not benchmarks.is_cached(source):
        console.print(
            f"[red]{source} is not cached.[/red] "
            f"Run: switchboard bench build {source}"
        )
        raise typer.Exit(code=1)

    frame = benchmarks.load(source)

    if by == "model":
        data = frame.model_summary().head(top)
        table = Table(title=f"{source}: models")
        table.add_column("Model")
        table.add_column("Questions", justify="right")
        table.add_column("Suites", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Cost", justify="right")
        table.add_column("Latency", justify="right")
        for model, row in data.iterrows():
            latency = (
                f"{row.mean_latency_s:.2f}s"
                if row.mean_latency_s == row.mean_latency_s
                else "-"
            )
            table.add_row(
                str(model),
                f"{int(row.answered):,}",
                str(int(row.benchmarks)),
                f"{row.accuracy:.1%}",
                f"${row.total_cost:,.2f}",
                latency,
            )
    else:
        data = frame.summary().head(top)
        table = Table(title=f"{source}: benchmark suites")
        table.add_column("Suite")
        table.add_column("Rows", justify="right")
        table.add_column("Questions", justify="right")
        table.add_column("Models", justify="right")
        table.add_column("Mean score", justify="right")
        table.add_column("Cost", justify="right")
        for name, row in data.iterrows():
            table.add_row(
                str(name),
                f"{int(row.rows):,}",
                f"{int(row.queries):,}",
                str(int(row.models)),
                f"{row.mean_score:.1%}",
                f"${row.total_cost:,.2f}",
            )

    console.print(table)


@bench_cli.command("headroom")
def bench_headroom(
    source: str = typer.Argument(..., help="Which cached source to analyse."),
    suite: str = typer.Option("", help="Restrict to one benchmark suite."),
    models: str = typer.Option("", help="Comma-separated models to compare."),
) -> None:
    """How much could perfect routing win?

    Builds a complete model x question grid, then reports the ceiling (an
    oracle that always picks a model that answers correctly) against the best
    and cheapest single models. This is the prize any router is chasing.
    """
    _require_eval()
    from eval import benchmarks

    if not benchmarks.is_cached(source):
        console.print(
            f"[red]{source} is not cached.[/red] "
            f"Run: switchboard bench build {source}"
        )
        raise typer.Exit(code=1)

    frame = benchmarks.load(source)
    if suite:
        frame = frame.filter(benchmark=suite)
    if models:
        wanted = [m.strip() for m in models.split(",") if m.strip()]
        frame = frame.filter(models=wanted)
        missing = set(wanted) - set(frame.models)
        if missing:
            console.print(f"[yellow]Not in this source: {sorted(missing)}[/yellow]")

    if not len(frame):
        console.print("[red]No rows matched those filters.[/red]")
        raise typer.Exit(code=1)

    grid = frame.grid()
    if grid.n_queries == 0:
        console.print(
            "[red]No question was answered by every selected model.[/red]\n"
            "Coverage is uneven - narrow the model list or pick one suite."
        )
        raise typer.Exit(code=1)

    best_model, best_acc = grid.best_single_model()
    cheap_model, cheap_cost = grid.cheapest_model()
    oracle_acc = grid.oracle_accuracy()
    oracle_cost = grid.oracle_cost()
    best_cost = float(grid.cost[best_model].sum())

    table = Table(title=f"{source}{f' / {suite}' if suite else ''} - routing headroom")
    table.add_column("Reference point")
    table.add_column("Accuracy", justify="right")
    table.add_column("Cost", justify="right")

    table.add_row(
        f"cheapest model ({cheap_model})",
        f"{grid.correct[cheap_model].mean():.1%}",
        f"${cheap_cost:,.4f}",
    )
    table.add_row(
        f"best single model ({best_model})", f"{best_acc:.1%}", f"${best_cost:,.4f}"
    )
    table.add_row(
        "[bold]ORACLE (perfect routing)[/bold]",
        f"[bold]{oracle_acc:.1%}[/bold]",
        f"[bold]${oracle_cost:,.4f}[/bold]",
    )
    console.print(table)

    console.print(
        f"\n[bold]{grid.n_queries:,}[/bold] questions x "
        f"[bold]{len(grid.models)}[/bold] models (complete grid)"
    )
    console.print(
        f"accuracy headroom over the best single model: "
        f"[green]+{100 * (oracle_acc - best_acc):.1f} points[/green]"
    )
    if best_cost > 0:
        console.print(
            f"cost at oracle quality: [green]{100 * (1 - oracle_cost / best_cost):.0f}%"
            f" cheaper[/green] than always using {best_model}"
        )
    console.print(
        f"questions routing could win: [green]{grid.routable_fraction():.1%}[/green]"
        f"  (no model solves {1 - grid.solvable_fraction():.1%})"
    )


@bench_cli.command("replay")
def bench_replay(
    source: str = typer.Argument(..., help="Which cached source to replay against."),
    suite: str = typer.Option("", help="Restrict to one benchmark suite."),
    models: str = typer.Option("", help="Comma-separated models to route between."),
    strategies: str = typer.Option(
        "random,keyword", help="Comma-separated strategies to score."
    ),
    seed: int = typer.Option(0, help="Seed for the random baseline."),
    out_dir: str = typer.Option("results", help="Where to write the report."),
    plot: bool = typer.Option(True, help="Also render the cost/accuracy chart."),
) -> None:
    """Score routing strategies against recorded answers - no API calls.

    Every strategy is measured against three fixed reference points: the
    cheapest model, the best single model (what companies do today), and an
    oracle that always picks a model which answers correctly. A routing score
    means nothing without that ceiling to compare it to.
    """
    _require_eval()
    import pandas as pd

    from eval import benchmarks
    from eval.benchmarks import replay as replay_mod

    if not benchmarks.is_cached(source):
        console.print(
            f"[red]{source} is not cached.[/red] "
            f"Run: switchboard bench build {source}"
        )
        raise typer.Exit(code=1)

    frame = benchmarks.load(source)
    if suite:
        frame = frame.filter(benchmark=suite)
    if models:
        wanted = [m.strip() for m in models.split(",") if m.strip()]
        frame = frame.filter(models=wanted)
        missing = set(wanted) - set(frame.models)
        if missing:
            console.print(f"[yellow]Not in this source: {sorted(missing)}[/yellow]")

    if not len(frame):
        console.print("[red]No rows matched those filters.[/red]")
        raise typer.Exit(code=1)

    grid = frame.grid()
    if grid.n_queries == 0:
        console.print(
            "[red]No question was answered by every selected model.[/red]\n"
            "Coverage is uneven - narrow the model list or pick one suite."
        )
        raise typer.Exit(code=1)

    # Strategies see only the question text, exactly as they would live.
    query_table = benchmarks.load_queries(source)
    texts = {
        (row.benchmark, row.query_id): row.query
        for row in query_table.itertuples()
    }

    names = [s.strip() for s in strategies.split(",") if s.strip()]
    console.print(
        f"Replaying [cyan]{grid.n_queries:,}[/cyan] questions x "
        f"[cyan]{len(grid.models)}[/cyan] models"
    )

    try:
        results = replay_mod.replay(grid, texts, names, seed=seed)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = replay_mod.compare(results)

    display = Table(title=f"{source}{f' / {suite}' if suite else ''} - routing replay")
    display.add_column("Strategy")
    display.add_column("Accuracy", justify="right")
    display.add_column("Cost", justify="right")
    display.add_column("Saving vs best", justify="right")
    display.add_column("Gap closed", justify="right")
    display.add_column("Models", justify="right")
    display.add_column("Trade-off curve", justify="center")

    for name, row in table.sort_values("cost_usd").iterrows():
        reference = name in replay_mod.REFERENCE_STRATEGIES
        label = f"[bold]{name}[/bold]" if reference else str(name)

        gap = row.get("gap_closed")
        if pd.isna(gap):
            gap_text = "-"
        elif gap < 0:
            gap_text = f"[red]{gap:.0%}[/red]"
        elif gap >= 0.5:
            gap_text = f"[green]{gap:.0%}[/green]"
        else:
            gap_text = f"{gap:.0%}"

        saving = row.get("saving_vs_best")
        display.add_row(
            label,
            f"{row.accuracy:.1%}",
            f"${row.cost_usd:,.4f}",
            "-" if pd.isna(saving) else f"{saving:.1%}",
            gap_text,
            str(int(row.models_used)),
            "[green]on[/green]" if row.get("pareto") else "[dim]dominated[/dim]",
        )
    console.print(display)

    console.print(
        "[dim]Gap closed = of the accuracy available between the best single "
        "model and perfect routing, how much this captured. Negative means "
        "less accurate than always using the best model - which can still be a "
        "good trade if it saves enough.\n"
        "Trade-off curve = no other achievable strategy is both cheaper AND "
        "more accurate. 'Dominated' means there is no reason to pick it.[/dim]"
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = f"replay-{source}" + (f"-{suite}" if suite else "")

    (out_path / f"{stem}.md").write_text(
        replay_mod.to_markdown(table) + "\n", encoding="utf-8"
    )
    table.to_csv(out_path / f"{stem}.csv")
    console.print(f"\nWrote [cyan]{out_path / f'{stem}.md'}[/cyan] and .csv")

    if plot:
        chart = replay_mod.plot(table, out_path / f"{stem}.png")
        console.print(f"Wrote [cyan]{chart}[/cyan]")


@bench_cli.command("train")
def bench_train(
    source: str = typer.Argument(..., help="Which cached source to train on."),
    suite: str = typer.Option("", help="Restrict to one benchmark suite."),
    models: str = typer.Option("", help="Comma-separated models to route between."),
    test_size: float = typer.Option(0.3, help="Fraction of questions held out."),
    thresholds: str = typer.Option(
        "0.3,0.4,0.5,0.6,0.7,0.8", help="Confidence levels to sweep."
    ),
    seed: int = typer.Option(0, help="Seed for the split and the classifiers."),
    features: str = typer.Option(
        "tfidf",
        help="surface | tfidf | embedding. Embeddings are far richer but very "
        "slow without a GPU.",
    ),
    baselines: str = typer.Option(
        "random,keyword", help="Baselines to compare against."
    ),
    cascades: bool = typer.Option(
        True,
        help="Also score cascades: call a cheap model, inspect the answer, "
        "escalate only if unconvinced.",
    ),
    out_dir: str = typer.Option("results", help="Where to write the report."),
    plot: bool = typer.Option(True, help="Also render the cost/accuracy chart."),
) -> None:
    """Train a learned router and score it on questions it has never seen.

    Learns, per model, the probability it answers a given question correctly,
    then routes to the cheapest model clearing a confidence threshold. Sweeping
    that threshold traces a whole cost/quality curve from one trained model.

    Everything is measured on a held-out split. A router scored on its own
    training questions measures memory, not judgement.
    """
    _require_eval()
    import pandas as pd

    from eval import benchmarks
    from eval.benchmarks import learned
    from eval.benchmarks import replay as replay_mod

    if not benchmarks.is_cached(source):
        console.print(
            f"[red]{source} is not cached.[/red] "
            f"Run: switchboard bench build {source}"
        )
        raise typer.Exit(code=1)

    frame = benchmarks.load(source)
    if suite:
        frame = frame.filter(benchmark=suite)
    if models:
        wanted = [m.strip() for m in models.split(",") if m.strip()]
        frame = frame.filter(models=wanted)

    if not len(frame):
        console.print("[red]No rows matched those filters.[/red]")
        raise typer.Exit(code=1)

    grid = frame.grid()
    if grid.n_queries < 50:
        console.print(
            f"[red]Only {grid.n_queries} complete questions.[/red] "
            "Too few to train and test on."
        )
        _explain_empty_grid(frame)
        raise typer.Exit(code=1)

    query_table = benchmarks.load_queries(source)
    texts = {
        (row.benchmark, row.query_id): row.query for row in query_table.itertuples()
    }

    train_index, test_index = learned.split_questions(grid, test_size, seed)
    train_grid, test_grid = grid.subset(train_index), grid.subset(test_index)

    console.print(
        f"[bold]{grid.n_queries:,}[/bold] questions x "
        f"[bold]{len(grid.models)}[/bold] models  ->  "
        f"train {len(train_index):,} / test {len(test_index):,}"
    )
    console.print("Extracting features and training ...")

    from eval.benchmarks.features import FeatureExtractor

    predictor = learned.SuccessPredictor.train(
        train_grid, texts, FeatureExtractor(mode=features), seed=seed
    )

    test_texts = [texts.get(key, "") for key in test_grid.correct.index]
    console.print(f"Features: [cyan]{predictor.extractor.describe()}[/cyan]")

    report = learned.training_report(predictor, test_grid, texts)
    mean_auc = report["auc"].mean()

    auc_table = Table(title="Can we predict success? (held-out AUC)")
    auc_table.add_column("Model")
    auc_table.add_column("Gets it right", justify="right")
    auc_table.add_column("AUC", justify="right")
    for model, row in report.head(12).iterrows():
        auc = row["auc"]
        colour = "green" if auc >= 0.65 else "yellow" if auc >= 0.55 else "red"
        auc_table.add_row(
            str(model),
            f"{row['base_rate']:.1%}",
            "-" if pd.isna(auc) else f"[{colour}]{auc:.3f}[/{colour}]",
        )
    console.print(auc_table)
    console.print(
        f"[dim]AUC 0.5 = no better than guessing, 1.0 = perfect. "
        f"Mean across models: {mean_auc:.3f}. If these sit near 0.5 the "
        f"features carry no signal and no routing rule on top will help.[/dim]\n"
    )

    # --- Score everything on the held-out split ---------------------------
    levels = [float(t) for t in thresholds.split(",") if t.strip()]
    routers = learned.routers_for_thresholds(
        predictor, train_grid.mean_cost_per_model(), levels
    )

    baseline_names = [b.strip() for b in baselines.split(",") if b.strip()]
    results = replay_mod.replay(test_grid, texts, baseline_names, seed=seed)

    for router in routers:
        # Predict for the whole split in one pass; each decision then costs a
        # dictionary lookup instead of a fresh model call.
        router.warm(test_texts)
        choices = replay_mod.strategy_choices(router, test_grid, texts)
        results.append(replay_mod._result(router.name, test_grid, choices))

    if cascades:
        results.extend(
            _score_cascades(test_grid, train_grid, texts, levels, features, seed)
        )

    table = replay_mod.compare(results)

    heading = f"{source}{f' / {suite}' if suite else ''} - held-out results"
    display = Table(title=heading)
    display.add_column("Strategy")
    display.add_column("Accuracy", justify="right")
    display.add_column("Cost", justify="right")
    display.add_column("Saving vs best", justify="right")
    display.add_column("Gap closed", justify="right")
    display.add_column("Models", justify="right")
    display.add_column("Curve", justify="center")

    for name, row in table.sort_values("cost_usd").iterrows():
        is_learned = str(name).startswith(("learned", "cascade"))
        is_ref = name in replay_mod.REFERENCE_STRATEGIES
        label = (
            f"[cyan]{name}[/cyan]" if is_learned
            else f"[bold]{name}[/bold]" if is_ref
            else str(name)
        )
        gap = row.get("gap_closed")
        if pd.isna(gap):
            gap_text = "-"
        elif gap < 0:
            gap_text = f"[red]{gap:.0%}[/red]"
        elif gap >= 0.3:
            gap_text = f"[green]{gap:.0%}[/green]"
        else:
            gap_text = f"{gap:.0%}"

        saving = row.get("saving_vs_best")
        display.add_row(
            label,
            f"{row.accuracy:.1%}",
            f"${row.cost_usd:,.4f}",
            "-" if pd.isna(saving) else f"{saving:.1%}",
            gap_text,
            str(int(row.models_used)),
            "[green]on[/green]" if row.get("pareto") else "[dim]dominated[/dim]",
        )
    console.print(display)

    # --- Did it actually work? --------------------------------------------
    learned_rows = table[table.index.str.startswith(("learned", "cascade"))]
    beaten = []
    for name in baseline_names + ["always-cheapest", "always-best"]:
        if name not in table.index:
            continue
        base = table.loc[name]
        better = learned_rows[
            (learned_rows["accuracy"] >= base["accuracy"])
            & (learned_rows["cost_usd"] <= base["cost_usd"])
        ]
        if len(better):
            beaten.append(name)

    if beaten:
        console.print(
            f"\n[green]Learned strategies dominate: {', '.join(beaten)}[/green] "
            "(at least as accurate, and no more expensive)"
        )
    else:
        console.print(
            "\n[yellow]The learned router does not dominate any baseline.[/yellow] "
            "It may still sit on the trade-off curve - check the Curve column."
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = f"trained-{source}" + (f"-{suite}" if suite else "")
    (out_path / f"{stem}.md").write_text(
        replay_mod.to_markdown(table) + "\n", encoding="utf-8"
    )
    table.to_csv(out_path / f"{stem}.csv")
    report.to_csv(out_path / f"{stem}-auc.csv")
    console.print(f"\nWrote [cyan]{out_path / f'{stem}.md'}[/cyan], .csv and -auc.csv")

    if plot:
        chart = replay_mod.plot(table, out_path / f"{stem}.png")
        console.print(f"Wrote [cyan]{chart}[/cyan]")


@bench_cli.command("sla")
def bench_sla(
    source: str = typer.Argument(..., help="Cached source (needs per-query latency)."),
    suite: str = typer.Option("", help="Restrict to one benchmark suite."),
    models: str = typer.Option("", help="Comma-separated models to route between."),
    budgets: str = typer.Option(
        "1.0,1.5,2.0,3.0,5.0", help="Latency budgets in seconds to sweep."
    ),
    quality: float = typer.Option(0.5, help="Minimum predicted success to accept."),
    test_size: float = typer.Option(0.3, help="Fraction of questions held out."),
    seed: int = typer.Option(0, help="Seed for the split and the classifiers."),
    features: str = typer.Option("tfidf", help="surface | tfidf | embedding."),
    out_dir: str = typer.Option("results", help="Where to write the report."),
) -> None:
    """What does a latency promise cost you?

    Sweeps a range of latency budgets. For each, the router may only choose
    models whose typical response time fits, then takes the cheapest one likely
    enough to be right. Tightening the budget forces it off slower - often
    better - models, and the table shows exactly what that costs in accuracy.

    Violations are measured against the latency ACTUALLY recorded per request,
    not against the model averages the router used to decide.
    """
    _require_eval()
    import pandas as pd

    from eval import benchmarks
    from eval.benchmarks import constraints as constraints_mod
    from eval.benchmarks import learned
    from eval.benchmarks import replay as replay_mod
    from eval.benchmarks.features import FeatureExtractor

    if not benchmarks.is_cached(source):
        console.print(
            f"[red]{source} is not cached.[/red] "
            f"Run: switchboard bench build {source}"
        )
        raise typer.Exit(code=1)

    frame = benchmarks.load(source)
    if suite:
        frame = frame.filter(benchmark=suite)
    if models:
        frame = frame.filter(models=[m.strip() for m in models.split(",") if m.strip()])

    if not frame.has_latency:
        console.print(
            f"[red]{source} records no per-request latency.[/red]\n"
            "Latency SLAs can only be measured where response times were "
            "recorded. Try: [cyan]switchboard bench sla xroutebench[/cyan]"
        )
        raise typer.Exit(code=1)

    grid = frame.grid()
    if grid.n_queries < 50:
        console.print(f"[red]Only {grid.n_queries} complete questions.[/red]")
        _explain_empty_grid(frame)
        raise typer.Exit(code=1)

    query_table = benchmarks.load_queries(source)
    texts = {
        (row.benchmark, row.query_id): row.query for row in query_table.itertuples()
    }

    train_index, test_index = learned.split_questions(grid, test_size, seed)
    train_grid, test_grid = grid.subset(train_index), grid.subset(test_index)

    predictor = learned.SuccessPredictor.train(
        train_grid, texts, FeatureExtractor(mode=features), seed=seed
    )
    profile = constraints_mod.ModelProfile.from_grid(train_grid)
    test_texts = [texts.get(key, "") for key in test_grid.correct.index]

    console.print(
        f"[bold]{test_grid.n_queries:,}[/bold] held-out questions x "
        f"[bold]{len(grid.models)}[/bold] models"
    )

    speeds = Table(title="Latency per model (from training)")
    speeds.add_column("Model")
    speeds.add_column("Median", justify="right")
    speeds.add_column(
        f"p{constraints_mod.SLA_PERCENTILE} (used for eligibility)", justify="right"
    )
    speeds.add_column("Accuracy", justify="right")
    accuracy = train_grid.model_accuracy()
    for model in profile.latency_tail.sort_values().index:
        speeds.add_row(
            str(model),
            f"{profile.latency[model]:.2f}s",
            f"{profile.latency_tail[model]:.2f}s",
            f"{accuracy.get(model, float('nan')):.1%}",
        )
    console.print(speeds)

    levels = [float(b) for b in budgets.split(",") if b.strip()]
    rows = []

    # Unconstrained reference: what you get with no promise at all.
    for label, limits in [
        ("no SLA", constraints_mod.Constraints(min_quality=quality)),
        *[
            (
                f"<= {budget:g}s",
                constraints_mod.Constraints(
                    max_latency_s=budget, min_quality=quality
                ),
            )
            for budget in levels
        ],
    ]:
        router = constraints_mod.ConstrainedRouter(predictor, profile, limits)
        router.warm(test_texts)
        choices = replay_mod.strategy_choices(router, test_grid, texts)
        scored = test_grid.score_for(choices)
        budget = limits.max_latency_s
        latency = constraints_mod.latency_report(test_grid, choices, budget)

        rows.append(
            {
                "sla": label,
                "accuracy": scored["accuracy"],
                "cost_usd": scored["cost_usd"],
                "p95_latency_s": latency.get("p95_latency_s", float("nan")),
                "violation_rate": latency.get("sla_violation_rate", float("nan")),
                "unsatisfiable": router.unsatisfiable,
                "n_queries": scored["n_queries"],
                "models_used": len(set(choices)),
            }
        )

    table = pd.DataFrame(rows).set_index("sla")

    display = Table(title=f"{source} - what a latency promise costs")
    display.add_column("SLA")
    display.add_column("Accuracy", justify="right")
    display.add_column("Cost", justify="right")
    display.add_column("p95 latency", justify="right")
    display.add_column("Violations", justify="right")
    display.add_column("Models", justify="right")
    display.add_column("Impossible", justify="right")

    for label, row in table.iterrows():
        violations = row.violation_rate
        if pd.isna(violations):
            violation_text = "-"
        elif violations <= 0.05:
            violation_text = f"[green]{violations:.1%}[/green]"
        elif violations <= 0.20:
            violation_text = f"[yellow]{violations:.1%}[/yellow]"
        else:
            violation_text = f"[red]{violations:.1%}[/red]"

        # A promise no model can keep is not a trade-off, it is a promise
        # that was never achievable. Showing it as an ordinary row would hide
        # that the router simply fell back on every single request.
        share = row.unsatisfiable / max(int(row.get("n_queries", 0) or 0), 1)
        if row.unsatisfiable:
            impossible = f"[red]{share:.0%}[/red]"
        else:
            impossible = "[green]-[/green]"

        display.add_row(
            str(label),
            f"{row.accuracy:.1%}",
            f"${row.cost_usd:,.4f}",
            "-" if pd.isna(row.p95_latency_s) else f"{row.p95_latency_s:.2f}s",
            violation_text,
            str(int(row.models_used)),
            impossible,
        )
    console.print(display)

    unconstrained = table.loc["no SLA"]
    console.print(
        "[dim]Violations are measured against the latency ACTUALLY recorded "
        "per request, not the model averages the router used to decide - a "
        "fast model can still answer slowly. 5% or under is the usual target "
        f"for a p{constraints_mod.SLA_PERCENTILE} promise.[/dim]"
    )

    # Compare against the tightest budget that was actually ACHIEVABLE. A
    # promise no model can keep tells you nothing about the trade-off - the
    # router simply fell back on every request.
    achievable = table.iloc[1:][table.iloc[1:]["unsatisfiable"] == 0]
    if len(achievable):
        strictest = achievable.iloc[0]
        lost = unconstrained.accuracy - strictest.accuracy
        colour = "red" if lost > 0.05 else "green"
        console.print(
            f"\nThe tightest achievable promise ([cyan]{strictest.name}[/cyan]) "
            f"gives up [{colour}]{lost:.1%}[/] accuracy versus routing with no "
            f"SLA at all, and keeps violations at "
            f"{strictest.violation_rate:.1%}."
        )

    impossible = table[table["unsatisfiable"] > 0]
    if len(impossible):
        console.print(
            f"[yellow]No model is fast enough to promise: "
            f"{', '.join(str(i) for i in impossible.index)}.[/yellow]\n"
            "Those rows fell back to the fastest model on every request - the "
            "promise was never achievable with this pool, which is itself the "
            "answer. Add a faster model or loosen the budget."
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = f"sla-{source}" + (f"-{suite}" if suite else "")
    table.to_csv(out_path / f"{stem}.csv")
    console.print(f"\nWrote [cyan]{out_path / f'{stem}.csv'}[/cyan]")


# --- Router artifacts ------------------------------------------------------


@router_cli.command("train")
def router_train(
    source: str = typer.Argument("llmrouterbench", help="Benchmark to train on."),
    suite: str = typer.Option("mmlupro", help="Benchmark suite."),
    models: str = typer.Option("", help="Comma-separated models to train over."),
    features: str = typer.Option("tfidf", help="surface | tfidf | embedding."),
    test_size: float = typer.Option(0.3, help="Fraction held out for scoring."),
    seed: int = typer.Option(0, help="Seed for the split and the classifiers."),
    out: str = typer.Option("", help="Where to write the artifact."),
) -> None:
    """Train a router and save it for the server to load.

    The artifact learns over BENCHMARK model names. Your catalog almost
    certainly uses different ones, so each model in providers.yaml declares a
    `benchmark_alias` saying which benchmark model it stands in for. Run
    `switchboard router info` afterwards to see what actually mapped.
    """
    _require_eval()
    from eval import benchmarks
    from eval.benchmarks import learned
    from eval.benchmarks.features import FeatureExtractor
    from switchboard.routing import artifact as artifact_mod

    if not benchmarks.is_cached(source):
        console.print(
            f"[red]{source} is not cached.[/red] "
            f"Run: switchboard bench build {source}"
        )
        raise typer.Exit(code=1)

    frame = benchmarks.load(source)
    if suite:
        frame = frame.filter(benchmark=suite)
    if models:
        frame = frame.filter(models=[m.strip() for m in models.split(",") if m.strip()])

    grid = frame.grid()
    if grid.n_queries < 50:
        console.print(f"[red]Only {grid.n_queries} complete questions.[/red]")
        _explain_empty_grid(frame)
        raise typer.Exit(code=1)

    query_table = benchmarks.load_queries(source)
    texts = {
        (row.benchmark, row.query_id): row.query for row in query_table.itertuples()
    }

    train_index, test_index = learned.split_questions(grid, test_size, seed)
    train_grid, test_grid = grid.subset(train_index), grid.subset(test_index)

    console.print(
        f"Training on [cyan]{len(train_index):,}[/cyan] questions over "
        f"[cyan]{len(grid.models)}[/cyan] models ..."
    )
    predictor = learned.SuccessPredictor.train(
        train_grid, texts, FeatureExtractor(mode=features), seed=seed
    )

    # Scored on held-out questions so the recorded AUC is honest.
    report = learned.training_report(predictor, test_grid, texts)
    mean_auc = float(report["auc"].mean())

    metadata = artifact_mod.RouterMetadata(
        source=source,
        benchmark=suite,
        features=predictor.extractor.describe(),
        models=list(grid.models),
        n_train_questions=len(train_index),
        mean_auc=mean_auc,
    )

    destination = Path(out) if out else Path(settings.router_path)
    artifact_mod.save(destination, predictor, metadata)

    colour = "green" if mean_auc >= 0.65 else "yellow" if mean_auc >= 0.55 else "red"
    console.print(
        f"[green]Saved[/green] {destination}\n"
        f"  {metadata.describe()}\n"
        f"  held-out AUC: [{colour}]{mean_auc:.3f}[/{colour}] "
        f"(0.5 = guessing)"
    )

    # Show immediately whether this artifact can drive the local catalog -
    # finding out at server start would be a worse time.
    _report_router_mapping(metadata.models)


@router_cli.command("info")
def router_info(
    path: str = typer.Option("", help="Artifact to inspect."),
) -> None:
    """Show what a router was trained on, and which local models it can drive."""
    from switchboard.routing import artifact as artifact_mod

    target = Path(path) if path else Path(settings.router_path)
    metadata = artifact_mod.read_metadata(target)

    if metadata is None:
        if not target.exists():
            console.print(
                f"[yellow]No router at {target}.[/yellow]\n"
                "Train one with: [cyan]switchboard router train[/cyan]"
            )
            raise typer.Exit(code=1)
        console.print(f"[yellow]{target} has no readable metadata sidecar.[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"Artifact: [cyan]{target}[/cyan]")
    console.print(f"  {metadata.describe()}")
    colour = (
        "green"
        if metadata.mean_auc >= 0.65
        else "yellow"
        if metadata.mean_auc >= 0.55
        else "red"
    )
    console.print(f"  held-out AUC: [{colour}]{metadata.mean_auc:.3f}[/{colour}]\n")

    _report_router_mapping(metadata.models)


def _report_router_mapping(trained_models: list[str]) -> None:
    """Which trained models map onto something this catalog can actually serve."""
    from switchboard.providers import ProviderPool
    from switchboard.routing.live import build_model_map

    catalog = _catalog()
    try:
        pool = ProviderPool(catalog, local_only=settings.local_only)
        available = pool.available_models()
    except Exception:  # noqa: BLE001 - inspection must not depend on providers
        available = list(catalog.models)

    mapping = build_model_map(catalog, trained_models, available)

    table = Table(title="Trained model -> local model")
    table.add_column("Benchmark model")
    table.add_column("Serves as")
    for name in sorted(trained_models):
        local = mapping.get(name)
        table.add_row(
            name,
            f"[green]{local}[/green]" if local else "[dim]not mapped[/dim]",
        )
    console.print(table)

    if len(mapping) >= 2:
        console.print(
            f"[green]Routing will be enabled[/green] over "
            f"{', '.join(sorted(mapping.values()))}."
        )
    else:
        console.print(
            f"[yellow]Routing will be DISABLED[/yellow] - only {len(mapping)} "
            "model mapped, and at least 2 are needed.\n"
            "Add `benchmark_alias:` to models in providers.yaml to say which "
            "benchmark model each one stands in for."
        )


@cli.command()
def shadow() -> None:
    """What would routing have done to the traffic you actually served?

    Reads the requests recorded while shadow mode was on. Every figure here is
    a PROJECTION: the shadow model was never called, so its cost is estimated
    from the tokens the real model produced, and nothing here can say whether
    its answer would have been as good.
    """
    from switchboard.shadow import summarise

    ledger = _ledger()
    summary = summarise(ledger.shadow_rows())

    if not summary.requests:
        console.print(
            "[yellow]No shadowed requests recorded this month.[/yellow]\n"
            "Turn shadow mode on with [cyan]SWITCHBOARD_SHADOW_MODE=true[/cyan], "
            "send some traffic, then run this again."
        )
        return

    table = Table(title="Shadow mode - what routing WOULD have done")
    table.add_column("Measure")
    table.add_column("Value", justify="right")

    table.add_row("Requests shadowed", f"{summary.requests:,}")
    table.add_row("Cost as served", f"${summary.actual_cost_usd:,.4f}")
    table.add_row("Cost if routed (estimated)", f"${summary.shadow_cost_usd:,.4f}")

    colour = "green" if summary.projected_saving_usd >= 0 else "red"
    table.add_row(
        "Projected saving",
        f"[{colour}]${summary.projected_saving_usd:,.4f} "
        f"({summary.projected_saving_pct:.1f}%)[/{colour}]",
    )
    table.add_row("Different model chosen", f"{summary.changed:,} "
                  f"({summary.changed_pct:.0f}%)")
    table.add_row("  ... to something cheaper", f"{summary.downgraded:,}")
    table.add_row("  ... to something dearer", f"{summary.upgraded:,}")
    console.print(table)

    if summary.model_counts:
        picks = Table(title="Models routing would have chosen")
        picks.add_column("Model")
        picks.add_column("Times", justify="right")
        for model, count in sorted(
            summary.model_counts.items(), key=lambda kv: -kv[1]
        ):
            picks.add_row(str(model), f"{count:,}")
        console.print(picks)

    console.print(
        "\n[dim]Projections, not measurements. The shadow model was never "
        "called: its cost is estimated from the tokens the real model produced, "
        "so a chattier model would truly have cost more. And no answer was "
        "produced to grade, so nothing here says whether quality would have "
        "held up.[/dim]"
    )


if __name__ == "__main__":
    cli()
