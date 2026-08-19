"""Scoring routing strategies against recorded benchmark data.

The trick that makes this work: the benchmarks record what EVERY model did on
EVERY question. So to find out how a routing strategy would have performed, we
do not have to call anything. We ask the strategy which model it would pick,
then look up what that model actually did.

A full comparison across 40 models and hundreds of thousands of answers takes
under a second and costs nothing. The same experiment run live would take days
and hundreds of dollars.

Three reference points are always included, because a routing score means
nothing on its own:

    always-cheapest  the cost floor
    always-best      what a company does today, and the quality to beat
    oracle           the impossible ceiling - see `oracle_choices`
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import pandas as pd

from eval.benchmarks.schema import Grid
from switchboard.routing import RoutingContext, RoutingStrategy
from switchboard.routing.baselines import AlwaysModel, KeywordHeuristic, RandomModel

#: Strategy names that need the grid's own outcomes, so cannot be a
#: RoutingStrategy - they are computed directly against the recorded results.
REFERENCE_STRATEGIES = ("always-cheapest", "always-best", "oracle")

BASELINE_STRATEGIES = ("random", "keyword")


@dataclass(frozen=True)
class ReplayResult:
    strategy: str
    accuracy: float
    cost_usd: float
    mean_latency_s: float
    n_queries: int
    model_usage: dict[str, int] = field(default_factory=dict)

    def cost_per_query(self) -> float:
        return self.cost_usd / self.n_queries if self.n_queries else 0.0


# --- Ladders and reference choices ------------------------------------------


def cost_ordered_models(grid: Grid) -> list[str]:
    """Models cheapest first, by what they actually charged on this grid.

    Strategies assume position 0 is cheapest, so the order is derived from
    measured spend rather than from a price list that might not match how the
    models were actually used - a verbose model can cost more than a nominally
    pricier terse one.
    """
    return list(grid.cost.sum().sort_values().index)


def oracle_choices(grid: Grid) -> pd.Series:
    """The impossible router: for each question, the cheapest model that got it right.

    This cannot be built. Knowing which model will answer correctly requires
    already knowing the answer. It exists only to establish the ceiling - a
    routing score with no ceiling cannot be interpreted.

    Where no model answers correctly there is nothing to pick, so it falls back
    to the cheapest model. Those questions are lost by every strategy equally,
    so the choice does not affect any comparison.
    """
    correct = grid.correct > 0
    masked = grid.cost.where(correct)
    fallback = grid.cost.sum().idxmin()

    # idxmin raises on a row that is entirely NA, which is exactly what an
    # unsolvable question looks like here. So those rows are filled in
    # separately rather than asking pandas to pick a minimum of nothing.
    solvable = correct.any(axis=1)
    choices = pd.Series(fallback, index=grid.correct.index, dtype=object)
    if solvable.any():
        choices[solvable] = masked[solvable].idxmin(axis=1)
    return choices


def always_choices(grid: Grid, model: str) -> pd.Series:
    return pd.Series(model, index=grid.correct.index)


# --- Running strategies ------------------------------------------------------


def strategy_choices(
    strategy: RoutingStrategy, grid: Grid, texts: Mapping[tuple[str, str], str]
) -> pd.Series:
    """Ask a real routing strategy what it would pick for every question.

    The strategy sees only the question text - exactly what it would see in
    production. It has no access to the recorded outcomes, which is what keeps
    the evaluation honest.
    """
    picks = []
    for benchmark, query_id in grid.correct.index:
        text = texts.get((benchmark, query_id), "")
        context = RoutingContext(messages=[{"role": "user", "content": text}])
        picks.append(strategy.choose(context).model)
    return pd.Series(picks, index=grid.correct.index)


def build_for_ladder(
    name: str, ladder: list[str], seed: int = 0
) -> RoutingStrategy:
    """Build a baseline strategy over an arbitrary list of models.

    The strategies in switchboard.routing normally take their ladder from the
    provider catalog. Here the ladder is whichever models a benchmark happens
    to cover, so it is injected instead.
    """
    if name == "random":
        return RandomModel(ladder, seed=seed)
    if name == "keyword":
        return KeywordHeuristic(ladder)
    if name.startswith("always:"):
        model = name.split(":", 1)[1]
        if model not in ladder:
            raise ValueError(f"{model!r} is not in this grid. Available: {ladder}")
        return AlwaysModel(model, name=name)
    raise ValueError(
        f"Unknown strategy {name!r}. Available: "
        f"{', '.join(BASELINE_STRATEGIES)}, always:<model>, "
        f"or a reference point: {', '.join(REFERENCE_STRATEGIES)}"
    )


def _result(name: str, grid: Grid, choices: pd.Series) -> ReplayResult:
    scored = grid.score_for(choices)
    return ReplayResult(
        strategy=name,
        accuracy=scored["accuracy"],
        cost_usd=scored["cost_usd"],
        mean_latency_s=scored["mean_latency_s"],
        n_queries=scored["n_queries"],
        model_usage=choices.value_counts().to_dict(),
    )


def replay(
    grid: Grid,
    texts: Mapping[tuple[str, str], str],
    strategies: Iterable[str] = BASELINE_STRATEGIES,
    seed: int = 0,
) -> list[ReplayResult]:
    """Score the reference points plus every named strategy."""
    ladder = cost_ordered_models(grid)
    cheapest = ladder[0]
    best = grid.model_accuracy().index[0]

    results = [
        _result("always-cheapest", grid, always_choices(grid, cheapest)),
        _result("always-best", grid, always_choices(grid, best)),
        _result("oracle", grid, oracle_choices(grid)),
    ]

    for name in strategies:
        if name in REFERENCE_STRATEGIES:
            continue  # already included above
        strategy = build_for_ladder(name, ladder, seed=seed)
        results.append(_result(name, grid, strategy_choices(strategy, grid, texts)))

    return results


# --- Comparison --------------------------------------------------------------


def compare(results: list[ReplayResult]) -> pd.DataFrame:
    """Turn raw results into the table that actually answers the question.

    Two derived columns carry the meaning:

    `saving_vs_best`  - how much cheaper than what a company does today.
    `gap_closed`      - of the accuracy available between "best single model"
                        and "perfect routing", what fraction did this capture?
                        Negative means it did worse than just picking one good
                        model, which is the outcome the ACL 2026 survey found
                        for several commercial routers.
    """
    by_name = {r.strategy: r for r in results}
    best = by_name.get("always-best")
    oracle = by_name.get("oracle")

    rows = []
    for r in results:
        row = {
            "strategy": r.strategy,
            "accuracy": r.accuracy,
            "cost_usd": r.cost_usd,
            "cost_per_query": r.cost_per_query(),
            "mean_latency_s": r.mean_latency_s,
            "models_used": len(r.model_usage),
        }
        if best is not None and best.cost_usd > 0:
            row["saving_vs_best"] = 1.0 - (r.cost_usd / best.cost_usd)
            row["accuracy_vs_best"] = r.accuracy - best.accuracy
        if best is not None and oracle is not None:
            headroom = oracle.accuracy - best.accuracy
            gained = r.accuracy - best.accuracy
            row["gap_closed"] = gained / headroom if headroom > 0 else float("nan")
        rows.append(row)

    table = pd.DataFrame(rows).set_index("strategy")
    table["pareto"] = _pareto_optimal(table)
    return table


def _pareto_optimal(table: pd.DataFrame) -> pd.Series:
    """Mark strategies no other ACHIEVABLE strategy beats outright.

    A strategy is dominated when another is at least as accurate AND at least
    as cheap, with one of those strictly better - meaning there is no reason to
    ever choose it. Anything not dominated sits on the trade-off curve, and
    choosing between those is a judgement about what accuracy is worth.

    This matters because "gap closed" only measures accuracy. A strategy that
    gives up 7 points to save 97% of the bill looks like a failure by that
    column alone, and is not one.

    The oracle is excluded from the comparison: it is impossible to build, so
    letting it dominate real strategies would mark everything as pointless.
    """
    achievable = table.drop(index="oracle", errors="ignore")

    flags = {}
    for name, row in table.iterrows():
        if name == "oracle":
            flags[name] = True  # the ceiling is on the curve by definition
            continue
        others = achievable.drop(index=name, errors="ignore")
        dominated = (
            (others["accuracy"] >= row.accuracy)
            & (others["cost_usd"] <= row.cost_usd)
            & (
                (others["accuracy"] > row.accuracy)
                | (others["cost_usd"] < row.cost_usd)
            )
        )
        flags[name] = not bool(dominated.any())
    return pd.Series(flags)


def to_markdown(table: pd.DataFrame) -> str:
    lines = [
        "| Strategy | Accuracy | Cost | Saving vs best | Gap to oracle closed "
        "| On trade-off curve |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in table.sort_values("cost_usd").iterrows():
        saving = row.get("saving_vs_best")
        gap = row.get("gap_closed")
        lines.append(
            f"| {name} | {row.accuracy:.1%} | ${row.cost_usd:,.4f} | "
            f"{'-' if pd.isna(saving) else f'{saving:.1%}'} | "
            f"{'-' if pd.isna(gap) else f'{gap:.1%}'} | "
            f"{'yes' if row.get('pareto') else 'no - dominated'} |"
        )
    return "\n".join(lines)


def plot(table: pd.DataFrame, output) -> object:
    """Cost against accuracy. Up and to the left is better."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))

    for name, row in table.iterrows():
        reference = name in REFERENCE_STRATEGIES
        ax.scatter(
            row.cost_usd,
            row.accuracy * 100,
            s=170 if reference else 110,
            marker="*" if reference else "o",
            zorder=3,
        )
        ax.annotate(
            name,
            (row.cost_usd, row.accuracy * 100),
            textcoords="offset points",
            xytext=(9, 6),
            fontsize=9,
            fontweight="bold" if reference else "normal",
        )

    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Total cost for the whole question set (USD, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Routing strategies - better is up and to the left")
    ax.grid(True, alpha=0.3, zorder=0)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output
