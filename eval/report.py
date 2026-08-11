"""Turning raw results into the comparison that matters.

The headline question this answers: for each strategy, how much accuracy do you
give up, and how much simulated money do you save, relative to sending
everything to the top-tier model?
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from eval.runner import TaskResult

REFERENCE_STRATEGY = "always-expensive"


@dataclass(frozen=True)
class StrategySummary:
    strategy: str
    tasks: int
    correct: int
    cost_usd: float
    baseline_usd: float
    latency_ms_total: int
    model_switches: int
    format_failures: int
    truncations: int
    errors: int
    model_usage: dict[str, int]
    accuracy_by_difficulty: dict[str, tuple[int, int]]

    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / self.tasks if self.tasks else 0.0

    @property
    def avg_latency_ms(self) -> int:
        return self.latency_ms_total // self.tasks if self.tasks else 0

    @property
    def saved_vs_baseline_pct(self) -> float:
        """Saving against 'route everything to the top tier'."""
        if self.baseline_usd <= 0:
            return 0.0
        return 100.0 * (self.baseline_usd - self.cost_usd) / self.baseline_usd


def summarise(results: list[TaskResult]) -> list[StrategySummary]:
    by_strategy: dict[str, list[TaskResult]] = {}
    for result in results:
        by_strategy.setdefault(result.strategy, []).append(result)

    summaries = []
    for strategy, rows in by_strategy.items():
        difficulty: dict[str, tuple[int, int]] = {}
        for level in ("easy", "medium", "hard"):
            subset = [r for r in rows if r.difficulty == level]
            difficulty[level] = (sum(1 for r in subset if r.correct), len(subset))

        summaries.append(
            StrategySummary(
                strategy=strategy,
                tasks=len(rows),
                correct=sum(1 for r in rows if r.correct),
                cost_usd=sum(r.simulated_cost_usd for r in rows),
                baseline_usd=sum(r.baseline_cost_usd for r in rows),
                latency_ms_total=sum(r.latency_ms for r in rows),
                model_switches=sum(1 for r in rows if r.caused_model_switch),
                format_failures=sum(1 for r in rows if not r.followed_format),
                truncations=sum(1 for r in rows if r.truncated),
                errors=sum(1 for r in rows if r.error),
                model_usage=dict(Counter(r.model for r in rows)),
                accuracy_by_difficulty=difficulty,
            )
        )

    # Cheapest first: the trade-off reads naturally down the column.
    return sorted(summaries, key=lambda s: s.cost_usd)


def to_markdown(summaries: list[StrategySummary]) -> str:
    lines = [
        "| Strategy | Accuracy | Cost (sim.) | Saved vs top tier | Avg latency | Switches |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.strategy} | {s.accuracy:.1f}% ({s.correct}/{s.tasks}) | "
            f"${s.cost_usd:.4f} | {s.saved_vs_baseline_pct:.1f}% | "
            f"{s.avg_latency_ms} ms | {s.model_switches} |"
        )
    return "\n".join(lines)


def pareto_plot(summaries: list[StrategySummary], output: Path) -> Path:
    """Cost against accuracy. Up and to the left is better.

    Matplotlib's Agg backend is forced: this runs headless and must never try to
    open a window.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for summary in summaries:
        ax.scatter(summary.cost_usd, summary.accuracy, s=110, zorder=3)
        ax.annotate(
            summary.strategy,
            (summary.cost_usd, summary.accuracy),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    ax.set_xlabel("Simulated cost for the whole task set (USD)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cost vs accuracy - better is up and to the left")
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output
