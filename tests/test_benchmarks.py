"""The benchmark loaders and the grid analysis.

Tests run on synthetic data, not the real downloads. The real datasets are
gigabytes, gitignored, and not redistributable - a test suite that needed them
would be unrunnable for anyone cloning the repo. Integration tests that do use
them are skipped automatically when the cache is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from eval.benchmarks import BenchmarkError, BenchmarkFrame, validate
from eval.benchmarks.llmrouterbench import load_benchmark_dir
from eval.benchmarks.schema import COLUMNS, Grid


def rows(*records) -> pd.DataFrame:
    """Build an outcome frame from (benchmark, query, model, correct, cost)."""
    return pd.DataFrame(
        [
            {
                "source": "test",
                "benchmark": b,
                "query_id": q,
                "model": m,
                "correct": c,
                "cost_usd": cost,
                "prompt_tokens": 100,
                "output_tokens": 50,
                "latency_s": latency,
            }
            for b, q, m, c, cost, latency in records
        ],
        columns=list(COLUMNS),
    )


# --- Schema validation -----------------------------------------------------


def test_valid_frame_passes() -> None:
    frame = validate(rows(("b", "1", "cheap", 1.0, 0.01, 0.5)))
    assert len(frame) == 1


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(BenchmarkError, match="Missing columns"):
        validate(pd.DataFrame({"source": ["x"]}))


def test_duplicates_are_rejected() -> None:
    """A duplicate means one model answered one question twice.

    Left alone it silently doubles that model's weight in every average.
    """
    duped = rows(
        ("b", "1", "cheap", 1.0, 0.01, 0.5),
        ("b", "1", "cheap", 0.0, 0.01, 0.5),
    )
    with pytest.raises(BenchmarkError, match="duplicate"):
        validate(duped)


def test_scores_outside_zero_to_one_are_rejected() -> None:
    with pytest.raises(BenchmarkError, match="outside 0..1"):
        validate(rows(("b", "1", "cheap", 5.0, 0.01, 0.5)))


# --- Grid analysis ---------------------------------------------------------


@pytest.fixture
def grid() -> Grid:
    """Three questions, two models, with a deliberate structure:

      q1: both right          -> paying for `big` is waste
      q2: only `big` right    -> the winnable one
      q3: neither right       -> unwinnable
    """
    frame = BenchmarkFrame(
        validate(
            rows(
                ("b", "q1", "cheap", 1.0, 0.01, 1.0),
                ("b", "q1", "big", 1.0, 1.00, 2.0),
                ("b", "q2", "cheap", 0.0, 0.01, 1.0),
                ("b", "q2", "big", 1.0, 1.00, 2.0),
                ("b", "q3", "cheap", 0.0, 0.01, 1.0),
                ("b", "q3", "big", 0.0, 1.00, 2.0),
            )
        ),
        "test",
    )
    return frame.grid()


def test_grid_shape(grid: Grid) -> None:
    assert grid.n_queries == 3
    assert sorted(grid.models) == ["big", "cheap"]


def test_model_accuracy(grid: Grid) -> None:
    accuracy = grid.model_accuracy()
    assert accuracy["big"] == pytest.approx(2 / 3)
    assert accuracy["cheap"] == pytest.approx(1 / 3)


def test_oracle_beats_every_single_model(grid: Grid) -> None:
    """The whole premise: no one model is right about everything."""
    assert grid.oracle_accuracy() == pytest.approx(2 / 3)
    assert grid.oracle_accuracy() >= grid.model_accuracy().max()


def test_oracle_cost_picks_the_cheapest_correct_model(grid: Grid) -> None:
    """q1 -> cheap ($0.01), q2 -> only big is right ($1.00), q3 -> nothing."""
    assert grid.oracle_cost() == pytest.approx(1.01)


def test_unsolvable_questions_cost_nothing(grid: Grid) -> None:
    """q3 has no correct answer to price; charging for it would distort."""
    assert grid.solvable_fraction() == pytest.approx(2 / 3)


def test_routable_fraction_is_the_prize(grid: Grid) -> None:
    """Only q2 is winnable: cheap misses it, big gets it."""
    assert grid.routable_fraction() == pytest.approx(1 / 3)


def test_best_and_cheapest(grid: Grid) -> None:
    assert grid.best_single_model()[0] == "big"
    assert grid.cheapest_model()[0] == "cheap"


def test_score_for_evaluates_a_routing_decision(grid: Grid) -> None:
    """A perfect router: cheap where it suffices, big where it is needed."""
    choices = pd.Series(["cheap", "big", "cheap"], index=grid.correct.index)
    result = grid.score_for(choices)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["cost_usd"] == pytest.approx(0.01 + 1.00 + 0.01)
    assert result["n_queries"] == 3


def test_score_for_rejects_an_unknown_model(grid: Grid) -> None:
    choices = pd.Series(["imaginary"] * 3, index=grid.correct.index)
    with pytest.raises(BenchmarkError, match="not in this grid"):
        grid.score_for(choices)


def test_score_for_rejects_incomplete_choices(grid: Grid) -> None:
    choices = pd.Series(["cheap"], index=grid.correct.index[:1])
    with pytest.raises(BenchmarkError, match="no routing choice"):
        grid.score_for(choices)


# --- Completeness ----------------------------------------------------------


def test_grid_drops_questions_not_every_model_answered() -> None:
    """Comparing models on different questions compares their exams.

    Coverage is genuinely uneven in the real data - flagship models appear on
    fewer benchmarks than small open ones.
    """
    frame = BenchmarkFrame(
        validate(
            rows(
                ("b", "q1", "cheap", 1.0, 0.01, 1.0),
                ("b", "q1", "big", 1.0, 1.00, 2.0),
                ("b", "q2", "cheap", 1.0, 0.01, 1.0),  # `big` never saw q2
            )
        ),
        "test",
    )
    grid = frame.grid()
    assert grid.n_queries == 1


def test_grid_survives_a_source_with_no_latency() -> None:
    """LLMRouterBench records none. An all-NaN column must not break the grid.

    Regression: pivot_table drops all-NaN columns by default, which produced an
    empty frame and a KeyError on alignment.
    """
    frame = BenchmarkFrame(
        validate(
            rows(
                ("b", "q1", "cheap", 1.0, 0.01, float("nan")),
                ("b", "q1", "big", 1.0, 1.00, float("nan")),
            )
        ),
        "test",
    )
    grid = frame.grid()
    assert grid.n_queries == 1
    assert list(grid.latency.columns) == list(grid.correct.columns)


# --- Frame helpers ---------------------------------------------------------


@pytest.fixture
def frame() -> BenchmarkFrame:
    return BenchmarkFrame(
        validate(
            rows(
                ("maths", "q1", "cheap", 1.0, 0.01, 1.0),
                ("maths", "q1", "big", 1.0, 1.00, 2.0),
                ("code", "q2", "cheap", 0.0, 0.02, 1.5),
                ("code", "q2", "big", 1.0, 2.00, 3.0),
            )
        ),
        "test",
    )


def test_filter_by_benchmark(frame: BenchmarkFrame) -> None:
    assert frame.filter(benchmark="maths").benchmarks == ["maths"]


def test_filter_by_model(frame: BenchmarkFrame) -> None:
    assert frame.filter(models=["cheap"]).models == ["cheap"]


def test_summary_counts_per_benchmark(frame: BenchmarkFrame) -> None:
    summary = frame.summary()
    assert set(summary.index) == {"maths", "code"}
    assert summary.loc["maths", "models"] == 2


def test_has_latency_detects_presence(frame: BenchmarkFrame) -> None:
    assert frame.has_latency is True


# --- LLMRouterBench loader -------------------------------------------------


def write_result(root: Path, benchmark: str, split: str, model: str, n: int) -> None:
    directory = root / benchmark / split / model
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model,
        "dataset_name": benchmark,
        "split": split,
        "counts": n,
        "records": [
            {
                "index": i + 1,
                "origin_query": f"{split} question {i + 1}",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cost": 0.001,
                "score": 1.0,
            }
            for i in range(n)
        ],
    }
    (directory / f"{benchmark}-{split}-{model}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_query_ids_are_split_qualified(tmp_path: Path) -> None:
    """The bug that schema validation caught.

    Several benchmarks ship more than one question set - `hle` has both
    `subset_500` and `test` - and `index` restarts at 1 in each. Keying on the
    raw index merged two different questions into one row, producing 25,563
    duplicates on the real data.
    """
    write_result(tmp_path, "hle", "test", "gpt-5", n=2)
    write_result(tmp_path, "hle", "subset_500", "gpt-5", n=2)

    frame, queries = load_benchmark_dir(tmp_path / "hle")

    assert len(frame) == 4
    assert not frame.duplicated(subset=["benchmark", "query_id", "model"]).any()
    assert set(frame["query_id"]) == {
        "test:1",
        "test:2",
        "subset_500:1",
        "subset_500:2",
    }
    assert len(queries) == 4


def test_loader_reads_metadata_from_the_file_not_the_path(tmp_path: Path) -> None:
    """Directory layouts are inconsistent; the file states what it is."""
    write_result(tmp_path, "gpqa", "test", "claude-sonnet-4", n=3)
    frame, _ = load_benchmark_dir(tmp_path / "gpqa")
    assert set(frame["benchmark"]) == {"gpqa"}
    assert set(frame["model"]) == {"claude-sonnet-4"}


def test_loader_output_passes_validation(tmp_path: Path) -> None:
    write_result(tmp_path, "gpqa", "test", "gpt-5", n=3)
    write_result(tmp_path, "gpqa", "test", "claude-sonnet-4", n=3)
    frame, _ = load_benchmark_dir(tmp_path / "gpqa")
    assert len(validate(frame)) == 6


def test_loader_skips_unreadable_files(tmp_path: Path) -> None:
    """One corrupt file must not abandon an hours-long build."""
    write_result(tmp_path, "gpqa", "test", "gpt-5", n=2)
    broken = tmp_path / "gpqa" / "test" / "broken"
    broken.mkdir(parents=True)
    (broken / "junk.json").write_text("{not json", encoding="utf-8")

    frame, _ = load_benchmark_dir(tmp_path / "gpqa")
    assert len(frame) == 2


def test_loader_records_no_latency(tmp_path: Path) -> None:
    """This source has none; it must be NaN rather than a fabricated zero."""
    write_result(tmp_path, "gpqa", "test", "gpt-5", n=2)
    frame, _ = load_benchmark_dir(tmp_path / "gpqa")
    assert frame["latency_s"].isna().all()


# --- Integration (skipped without the real cache) --------------------------

from eval.benchmarks import is_cached, load  # noqa: E402

needs_cache = pytest.mark.skipif(
    not is_cached("llmrouterbench"),
    reason="run `switchboard bench build llmrouterbench` first",
)


@needs_cache
def test_real_cache_loads_and_validates() -> None:
    frame = load("llmrouterbench")
    assert len(frame) > 100_000
    assert "gpt-5" in frame.models
    validate(frame.frame)


@needs_cache
def test_real_gpqa_grid_is_complete() -> None:
    grid = load("llmrouterbench").filter(benchmark="gpqa").grid()
    assert grid.n_queries == 198
    assert grid.oracle_accuracy() > grid.model_accuracy().max()
