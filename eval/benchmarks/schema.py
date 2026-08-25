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
    # A fingerprint of what the model answered - see `answer_key`. Not the
    # ground truth: this is what a cascade may look at after paying for a
    # cheap call. Empty where a source records no parsed answer.
    "prediction",
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
    "prediction": "string",
}


#: Answers longer than this are stored as a hash instead of verbatim.
#: Multiple-choice answers ("B") stay readable; a 1.7 MB code submission
#: becomes 18 characters.
ANSWER_KEY_MAX_CHARS = 64


def answer_key(prediction: str | None) -> str:
    """A short, comparable fingerprint of a model's answer.

    Cascades compare answers between models to gauge confidence: two models
    agreeing is evidence the answer is right. Only equality matters, never the
    content - so long answers are hashed rather than stored.

    That keeps the cache small AND keeps the semantics exact. Two identical
    essays hash identically; two different ones do not. Truncating instead
    would make different answers collide.
    """
    if not prediction:
        return ""

    normalised = " ".join(str(prediction).split()).lower()
    if len(normalised) <= ANSWER_KEY_MAX_CHARS:
        return normalised

    import hashlib

    return "h:" + hashlib.sha1(normalised.encode("utf-8", "replace")).hexdigest()[:16]


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
    output_tokens: pd.DataFrame | None = None
    prediction: pd.DataFrame | None = None

    @property
    def models(self) -> list[str]:
        return list(self.correct.columns)

    @property
    def n_queries(self) -> int:
        return len(self.correct)

    def subset(self, index) -> Grid:
        """A grid over a subset of the questions.

        Used to split into training and held-out test halves. A learned router
        can memorise the questions it was trained on, so it must be scored on
        questions it has never seen - otherwise the number measures recall, not
        the ability to judge a new question.
        """
        def slice_of(frame):
            return None if frame is None else frame.loc[index]

        return Grid(
            correct=self.correct.loc[index],
            cost=self.cost.loc[index],
            latency=self.latency.loc[index],
            output_tokens=slice_of(self.output_tokens),
            prediction=slice_of(self.prediction),
        )

    def mean_cost_per_model(self) -> pd.Series:
        """Average spend per question, per model. The router's price list."""
        return self.cost.mean()

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

    def score_for_paths(self, paths: pd.Series) -> dict[str, float]:
        """Evaluate a cascade: the ordered list of models actually called.

        A cascade that escalates has paid for BOTH calls, so cost and latency
        sum over the whole path. Charging only for the final model would make
        every cascade look cheaper than it is - the single easiest way to
        produce a flattering, wrong result here.

        Correctness comes from the last model called, which is the answer the
        cascade ultimately returns.
        """
        aligned = paths.reindex(self.correct.index)
        if aligned.isna().any():
            raise BenchmarkError(
                f"{int(aligned.isna().sum())} questions have no routing path."
            )

        total_cost = 0.0
        total_latency = 0.0
        correct = 0.0
        calls = 0
        final_models: list[str] = []

        for question, path in aligned.items():
            if not path:
                raise BenchmarkError(f"Empty routing path for {question}.")

            unknown = set(path) - set(self.models)
            if unknown:
                raise BenchmarkError(
                    f"Cascade called models not in this grid: {sorted(unknown)}"
                )

            for model in path:
                total_cost += float(self.cost.at[question, model])
                latency = self.latency.at[question, model]
                if latency == latency:  # not NaN
                    total_latency += float(latency)
            calls += len(path)

            final = path[-1]
            final_models.append(final)
            correct += float(self.correct.at[question, final])

        n = len(aligned)
        return {
            "accuracy": correct / n if n else 0.0,
            "cost_usd": total_cost,
            "mean_latency_s": total_latency / n if n else 0.0,
            "n_queries": n,
            "calls_per_query": calls / n if n else 0.0,
            "final_models": final_models,
        }

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
