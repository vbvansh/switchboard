"""Building and loading the normalised benchmark cache.

Reading 6.5 GB of JSON takes minutes. Phase C will re-run experiments hundreds
of times, and a loop that starts with a two-minute wait is a loop people stop
using. So each source is normalised ONCE into Parquet - columnar, compressed,
and typically 50-100x faster to read - and everything afterwards reads that.

The cache is derived data. It lives under data/ which is gitignored, and can be
rebuilt from the original sources at any time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from eval.benchmarks import llmrouterbench, xroutebench
from eval.benchmarks.schema import COLUMNS, Grid, validate

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/benchmarks/cache")
SOURCES = ("llmrouterbench", "xroutebench")


def outcomes_path(source: str) -> Path:
    return CACHE_DIR / f"{source}.parquet"


def queries_path(source: str) -> Path:
    return CACHE_DIR / f"{source}-queries.parquet"


def is_cached(source: str) -> bool:
    return outcomes_path(source).exists()


# --- Building --------------------------------------------------------------


def build(source: str, progress=None) -> Path:
    """Normalise a source into the cache. Returns the outcomes file."""
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}. Known: {', '.join(SOURCES)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if source == "llmrouterbench":
        outcomes, queries = _build_llmrouterbench(progress)
    else:
        outcomes, queries = xroutebench.load()
        if progress:
            progress(source, len(outcomes))

    outcomes = validate(outcomes)
    outcomes.to_parquet(outcomes_path(source), index=False, compression="zstd")
    queries.to_parquet(queries_path(source), index=False, compression="zstd")
    return outcomes_path(source)


def _build_llmrouterbench(progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream benchmark directories so peak memory stays bounded."""
    outcome_parts: list[pd.DataFrame] = []
    query_parts: list[pd.DataFrame] = []

    for name, frame, queries in llmrouterbench.iter_benchmarks():
        outcome_parts.append(frame)
        query_parts.append(queries)
        if progress:
            progress(name, len(frame))

    if not outcome_parts:
        raise RuntimeError("No benchmarks loaded from LLMRouterBench.")

    return (
        pd.concat(outcome_parts, ignore_index=True),
        pd.concat(query_parts, ignore_index=True),
    )


# --- Loading ---------------------------------------------------------------


def load(source: str, rebuild: bool = False) -> BenchmarkFrame:
    if rebuild or not is_cached(source):
        logger.info("Building cache for %s ...", source)
        build(source)
    return BenchmarkFrame(pd.read_parquet(outcomes_path(source)), source)


def load_queries(source: str) -> pd.DataFrame:
    """Question text, kept apart from the outcome table to save space."""
    path = queries_path(source)
    if not path.exists():
        raise FileNotFoundError(
            f"No query cache for {source}. Run: switchboard bench build"
        )
    return pd.read_parquet(path)


class BenchmarkFrame:
    """A normalised benchmark, with the filters analysis actually needs."""

    def __init__(self, frame: pd.DataFrame, source: str) -> None:
        self.frame = frame
        self.source = source

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def benchmarks(self) -> list[str]:
        return sorted(self.frame["benchmark"].unique())

    @property
    def models(self) -> list[str]:
        return sorted(self.frame["model"].unique())

    @property
    def has_latency(self) -> bool:
        return bool(self.frame["latency_s"].notna().any())

    def filter(
        self,
        benchmark: str | list[str] | None = None,
        models: list[str] | None = None,
    ) -> BenchmarkFrame:
        frame = self.frame
        if benchmark is not None:
            wanted = [benchmark] if isinstance(benchmark, str) else benchmark
            frame = frame[frame["benchmark"].isin(wanted)]
        if models is not None:
            frame = frame[frame["model"].isin(models)]
        return BenchmarkFrame(frame.reset_index(drop=True), self.source)

    def summary(self) -> pd.DataFrame:
        """Per-benchmark overview: size, coverage, difficulty."""
        return (
            self.frame.groupby("benchmark")
            .agg(
                rows=("model", "size"),
                queries=("query_id", "nunique"),
                models=("model", "nunique"),
                mean_score=("correct", "mean"),
                total_cost=("cost_usd", "sum"),
            )
            .sort_values("rows", ascending=False)
        )

    def model_summary(self) -> pd.DataFrame:
        return (
            self.frame.groupby("model")
            .agg(
                answered=("query_id", "nunique"),
                benchmarks=("benchmark", "nunique"),
                accuracy=("correct", "mean"),
                total_cost=("cost_usd", "sum"),
                mean_latency_s=("latency_s", "mean"),
            )
            .sort_values("accuracy", ascending=False)
        )

    # --- The important one -------------------------------------------------

    def grid(self, min_models: int | None = None) -> Grid:
        """Build a COMPLETE model x question matrix.

        Coverage is uneven: flagship models appear on fewer benchmarks than the
        small open ones. Comparing a model that answered 200 questions against
        one that answered 2000 would compare their exams, not the models. So
        this keeps only questions that EVERY selected model attempted, and
        drops the rest.

        Filter to the models you care about first, then call this - otherwise a
        single sparsely-covered model can shrink the grid to nothing.
        """
        frame = self.frame
        if min_models is not None:
            keep = (
                frame.groupby("query_id")["model"].transform("nunique") >= min_models
            )
            frame = frame[keep]

        def pivot(column: str) -> pd.DataFrame:
            # dropna=False is load-bearing: by default pivot_table discards
            # columns that are entirely NaN, so a source with no latency data
            # (LLMRouterBench records none) would produce an EMPTY frame and
            # break the alignment below.
            return frame.pivot_table(
                index=["benchmark", "query_id"],
                columns="model",
                values=column,
                aggfunc="first",
                dropna=False,
            )

        correct = pivot("correct")
        complete = correct.notna().all(axis=1)
        correct = correct[complete]

        # reindex, not .loc: a source missing a column entirely should yield
        # NaN for it, not raise.
        return Grid(
            correct=correct,
            cost=pivot("cost_usd").reindex(correct.index),
            latency=pivot("latency_s").reindex(correct.index).reindex(
                columns=correct.columns
            ),
        )

    def why_grid_is_empty(self) -> pd.Series:
        """Questions attempted per model, fewest first.

        When a complete grid comes out empty, it is because the selected models
        did not all answer the same questions - benchmarks here ship several
        splits and not every model ran on every one. Showing the counts points
        straight at the model to drop.
        """
        return (
            self.frame.groupby("model")["query_id"].nunique().sort_values()
        )

    def coverage(self) -> pd.DataFrame:
        """Which models answered which benchmarks - for spotting sparse ones."""
        return (
            self.frame.pivot_table(
                index="model",
                columns="benchmark",
                values="query_id",
                aggfunc="nunique",
            )
            .fillna(0)
            .astype(int)
        )


__all__ = [
    "CACHE_DIR",
    "COLUMNS",
    "SOURCES",
    "BenchmarkFrame",
    "build",
    "is_cached",
    "load",
    "load_queries",
]
