"""The common shape every benchmark is normalised into.

The two public routing datasets we use are structured completely differently -
one is 700 JSON files in inconsistent directory layouts, the other is Parquet
with different column names. Rather than teach every analysis about both, each
is converted once into the shape below. Everything downstream then works on
either source without knowing which it is.

Query TEXT is deliberately kept out of the outcome table. The same question
appears once per model - up to 38 times - so storing it inline would multiply
the text by 38 and turn a small table into gigabytes. It lives in a separate
lookup keyed by query_id instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Columns of the outcome table. One row = one model's attempt at one question.
COLUMNS = (
    "source",         # which dataset it came from
    "benchmark",      # which test suite, e.g. "gpqa"
    "query_id",       # stable id, unique within (source, benchmark)
    "model",          # which model answered
    "correct",        # score in 0..1 - graded against ground truth
    "cost_usd",       # what this answer cost
    "prompt_tokens",
    "output_tokens",
    "latency_s",      # measured; NaN where the source does not record it
)

#: Columns of the query lookup table.
QUERY_COLUMNS = ("source", "benchmark", "query_id", "query")

DTYPES = {
    "source": "string",
    "benchmark": "string",
    "query_id": "string",
    "model": "string",
    "correct": "float64",
    "cost_usd": "float64",
    "prompt_tokens": "int64",
    "output_tokens": "int64",
    "latency_s": "float64",
}


class BenchmarkError(ValueError):
    """A loaded benchmark violates the contract above."""


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce the schema, loudly.

    A benchmark table with a silent duplicate or a missing column produces
    plausible-looking numbers that are wrong, which is far worse than a crash.
    """
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise BenchmarkError(f"Missing columns: {missing}")

    frame = frame[list(COLUMNS)].astype(DTYPES)

    duplicated = frame.duplicated(subset=["source", "benchmark", "query_id", "model"])
    if duplicated.any():
        example = frame[duplicated].iloc[0]
        raise BenchmarkError(
            f"{duplicated.sum()} duplicate (benchmark, query_id, model) rows - "
            f"e.g. {example.benchmark}/{example.query_id}/{example.model}. "
            "Each model must answer each question exactly once."
        )

    out_of_range = ~frame["correct"].between(0.0, 1.0)
    if out_of_range.any():
        raise BenchmarkError(
            f"{out_of_range.sum()} rows have `correct` outside 0..1. "
            "Scores must be normalised before loading."
        )

    return frame.reset_index(drop=True)


@dataclass(frozen=True)
class Grid:
    """A complete model x question matrix.

    Every cell is filled: each model in `models` attempted every question in
    the index. That completeness is what makes an oracle computable - you
    cannot ask "what would the best possible router have done" unless you know
    what every model would have done.
    """

    correct: pd.DataFrame     # questions x models, values 0..1
    cost: pd.DataFrame        # questions x models, USD
    latency: pd.DataFrame     # questions x models, seconds (may be all NaN)

    @property
    def models(self) -> list[str]:
        return list(self.correct.columns)

    @property
    def n_queries(self) -> int:
        return len(self.correct)

    # --- Reference points --------------------------------------------------

    def model_accuracy(self) -> pd.Series:
        return self.correct.mean().sort_values(ascending=False)

    def model_cost(self) -> pd.Series:
        return self.cost.sum().sort_values()

    def oracle_accuracy(self) -> float:
        """The ceiling: a router that always picks a model that gets it right."""
        return float(self.correct.max(axis=1).mean())

    def oracle_cost(self) -> float:
        """The floor: always the CHEAPEST model that answers correctly.

        Questions no model solves cost nothing here - there is no correct
        choice to price, and charging for them would distort the comparison.
        """
        masked = self.cost.where(self.correct > 0)
        return float(masked.min(axis=1).fillna(0.0).sum())

    def best_single_model(self) -> tuple[str, float]:
        accuracy = self.model_accuracy()
        return accuracy.index[0], float(accuracy.iloc[0])

    def cheapest_model(self) -> tuple[str, float]:
        cost = self.model_cost()
        return cost.index[0], float(cost.iloc[0])

    def solvable_fraction(self) -> float:
        """Share of questions at least one model gets right."""
        return float((self.correct.max(axis=1) > 0).mean())

    def routable_fraction(self) -> float:
        """Share of questions the cheapest model misses but some model solves.

        This is the size of the prize: questions where routing can add value.
        """
        cheapest, _ = self.cheapest_model()
        winnable = (self.correct.max(axis=1) > 0) & (self.correct[cheapest] <= 0)
        return float(winnable.mean())

    def score_for(self, choices: pd.Series) -> dict[str, float]:
        """Evaluate a routing decision: one chosen model per question.

        `choices` maps question -> model name. This is how every strategy gets
        measured, from the trivial baselines to the learned router.
        """
        aligned = choices.reindex(self.correct.index)
        if aligned.isna().any():
            missing = int(aligned.isna().sum())
            raise BenchmarkError(
                f"{missing} questions have no routing choice."
            )

        unknown = set(aligned.unique()) - set(self.models)
        if unknown:
            raise BenchmarkError(
                f"Routed to models not in this grid: {sorted(unknown)}"
            )

        rows = range(len(aligned))
        picked_correct = self.correct.to_numpy()[
            rows, [self.correct.columns.get_loc(m) for m in aligned]
        ]
        picked_cost = self.cost.to_numpy()[
            rows, [self.cost.columns.get_loc(m) for m in aligned]
        ]
        picked_latency = self.latency.to_numpy()[
            rows, [self.latency.columns.get_loc(m) for m in aligned]
        ]

        return {
            "accuracy": float(picked_correct.mean()),
            "cost_usd": float(picked_cost.sum()),
            "mean_latency_s": float(pd.Series(picked_latency).mean()),
            "n_queries": len(aligned),
        }
