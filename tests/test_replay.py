"""Offline replay: scoring routing strategies against recorded outcomes."""

from __future__ import annotations

import pandas as pd
import pytest

from eval.benchmarks import BenchmarkFrame, validate
from eval.benchmarks.replay import (
    REFERENCE_STRATEGIES,
    ReplayResult,
    _pareto_optimal,
    always_choices,
    build_for_ladder,
    compare,
    cost_ordered_models,
    oracle_choices,
    replay,
    strategy_choices,
    to_markdown,
)
from eval.benchmarks.schema import Grid
from tests.test_benchmarks import rows


def make_grid(*records) -> Grid:
    return BenchmarkFrame(validate(rows(*records)), "test").grid()


@pytest.fixture
def grid() -> Grid:
    """Four questions over three models, arranged to exercise every case.

      q1  all three right           -> the cheapest should be chosen
      q2  only `mid` right          -> routing must escalate, but not to `big`
      q3  only `big` right          -> full escalation needed
      q4  nobody right              -> unwinnable, must not crash anything
    """
    return make_grid(
        ("b", "q1", "cheap", 1.0, 0.01, 1.0),
        ("b", "q1", "mid", 1.0, 0.10, 2.0),
        ("b", "q1", "big", 1.0, 1.00, 3.0),
        ("b", "q2", "cheap", 0.0, 0.01, 1.0),
        ("b", "q2", "mid", 1.0, 0.10, 2.0),
        ("b", "q2", "big", 0.0, 1.00, 3.0),
        ("b", "q3", "cheap", 0.0, 0.01, 1.0),
        ("b", "q3", "mid", 0.0, 0.10, 2.0),
        ("b", "q3", "big", 1.0, 1.00, 3.0),
        ("b", "q4", "cheap", 0.0, 0.01, 1.0),
        ("b", "q4", "mid", 0.0, 0.10, 2.0),
        ("b", "q4", "big", 0.0, 1.00, 3.0),
    )


@pytest.fixture
def texts(grid: Grid) -> dict:
    return {key: f"question {key[1]}" for key in grid.correct.index}


# --- Ladder ----------------------------------------------------------------


def test_ladder_is_ordered_by_measured_spend(grid: Grid) -> None:
    """Ordered by what models actually charged, not a nominal price list.

    A verbose cheap model can outspend a terse expensive one.
    """
    assert cost_ordered_models(grid) == ["cheap", "mid", "big"]


# --- The oracle ------------------------------------------------------------


def test_oracle_picks_the_cheapest_correct_model(grid: Grid) -> None:
    choices = oracle_choices(grid)
    assert choices[("b", "q1")] == "cheap"  # all right, so take the cheapest
    assert choices[("b", "q2")] == "mid"    # only mid is right
    assert choices[("b", "q3")] == "big"    # only big is right


def test_oracle_survives_a_question_nobody_solves(grid: Grid) -> None:
    """Regression: idxmin raises on an all-NA row.

    An unsolvable question masks to NaN for every model, which crashed the
    whole replay before this was handled.
    """
    choices = oracle_choices(grid)
    assert choices[("b", "q4")] == "cheap"  # nothing to pick, so the cheapest
    assert not choices.isna().any()


def test_oracle_is_the_ceiling(grid: Grid) -> None:
    """No achievable strategy may beat it - that is what makes it a ceiling."""
    oracle = grid.score_for(oracle_choices(grid))
    for model in grid.models:
        assert oracle["accuracy"] >= grid.correct[model].mean()


def test_oracle_accuracy_matches_the_grid(grid: Grid) -> None:
    assert grid.score_for(oracle_choices(grid))["accuracy"] == pytest.approx(0.75)


# --- Strategies ------------------------------------------------------------


def test_always_choices_is_constant(grid: Grid) -> None:
    choices = always_choices(grid, "mid")
    assert set(choices) == {"mid"}
    assert len(choices) == grid.n_queries


def test_strategy_sees_only_the_question_text(grid: Grid, texts: dict) -> None:
    """The strategy must not have access to recorded outcomes."""
    seen = []

    class Spy:
        name = "spy"

        def choose(self, context):
            seen.append(context.messages[0]["content"])
            from switchboard.routing import RoutingDecision

            return RoutingDecision(model="cheap", strategy="spy")

    strategy_choices(Spy(), grid, texts)
    assert seen == [f"question {q}" for _, q in grid.correct.index]


def test_build_for_ladder_makes_baselines() -> None:
    ladder = ["cheap", "mid", "big"]
    assert build_for_ladder("random", ladder).ladder == ladder
    assert build_for_ladder("keyword", ladder).ladder == ladder
    assert build_for_ladder("always:mid", ladder).model == "mid"


def test_build_for_ladder_rejects_a_model_not_present() -> None:
    with pytest.raises(ValueError, match="not in this grid"):
        build_for_ladder("always:imaginary", ["cheap"])


def test_build_for_ladder_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_for_ladder("telepathy", ["cheap"])


# --- Replay ----------------------------------------------------------------


def test_replay_always_includes_the_reference_points(grid: Grid, texts: dict) -> None:
    """A routing score is uninterpretable without floor, current practice, ceiling."""
    names = {r.strategy for r in replay(grid, texts, ["random"])}
    assert set(REFERENCE_STRATEGIES) <= names


def test_replay_does_not_duplicate_a_reference_point(grid: Grid, texts: dict) -> None:
    results = replay(grid, texts, ["oracle", "random"])
    assert [r.strategy for r in results].count("oracle") == 1


def test_replay_records_which_models_were_used(grid: Grid, texts: dict) -> None:
    results = {r.strategy: r for r in replay(grid, texts, [])}
    assert results["always-cheapest"].model_usage == {"cheap": 4}
    assert len(results["oracle"].model_usage) > 1


def test_replay_is_reproducible(grid: Grid, texts: dict) -> None:
    """An unseeded random baseline would move between runs."""
    first = [r.accuracy for r in replay(grid, texts, ["random"], seed=7)]
    second = [r.accuracy for r in replay(grid, texts, ["random"], seed=7)]
    assert first == second


# --- Comparison ------------------------------------------------------------


def test_gap_closed_is_zero_for_the_best_model_and_one_for_the_oracle(
    grid: Grid, texts: dict
) -> None:
    table = compare(replay(grid, texts, []))
    assert table.loc["always-best", "gap_closed"] == pytest.approx(0.0)
    assert table.loc["oracle", "gap_closed"] == pytest.approx(1.0)


def test_gap_closed_goes_negative_below_the_best_model(
    grid: Grid, texts: dict
) -> None:
    """The honest signal that a strategy lost accuracy to save money."""
    table = compare(replay(grid, texts, []))
    assert table.loc["always-cheapest", "gap_closed"] < 0


def test_saving_vs_best_is_measured_against_current_practice(
    grid: Grid, texts: dict
) -> None:
    table = compare(replay(grid, texts, []))
    assert table.loc["always-best", "saving_vs_best"] == pytest.approx(0.0)
    assert table.loc["always-cheapest", "saving_vs_best"] > 0.9


# --- Pareto ----------------------------------------------------------------


def frame_of(**strategies) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"strategy": name, "accuracy": acc, "cost_usd": cost}
            for name, (acc, cost) in strategies.items()
        ]
    ).set_index("strategy")


def test_a_strictly_worse_strategy_is_dominated() -> None:
    """Cheaper AND more accurate elsewhere means there is no reason to pick it."""
    table = frame_of(good=(0.80, 1.0), bad=(0.70, 2.0))
    # pandas stores these as numpy booleans, so compare by truthiness.
    flags = _pareto_optimal(table)
    assert flags["good"]
    assert not flags["bad"]


def test_a_cheaper_but_worse_strategy_stays_on_the_curve() -> None:
    """Trading accuracy for cost is a valid choice, not a defeat."""
    flags = _pareto_optimal(frame_of(expensive=(0.90, 10.0), thrifty=(0.70, 0.1)))
    assert flags["expensive"]
    assert flags["thrifty"]


def test_the_oracle_never_dominates_real_strategies() -> None:
    """It cannot be built, so letting it dominate would mark everything futile."""
    table = frame_of(oracle=(0.99, 0.5), real=(0.80, 1.0))
    flags = _pareto_optimal(table)
    assert flags["real"]
    assert flags["oracle"]


def test_identical_strategies_are_not_mutually_dominated() -> None:
    flags = _pareto_optimal(frame_of(a=(0.8, 1.0), b=(0.8, 1.0)))
    assert flags["a"] and flags["b"]


# --- Rendering -------------------------------------------------------------


def test_markdown_includes_every_strategy(grid: Grid, texts: dict) -> None:
    markdown = to_markdown(compare(replay(grid, texts, ["random"])))
    for name in (*REFERENCE_STRATEGIES, "random"):
        assert f"| {name} |" in markdown


def test_markdown_marks_dominated_strategies(grid: Grid, texts: dict) -> None:
    markdown = to_markdown(compare(replay(grid, texts, [])))
    assert "yes" in markdown


def test_result_cost_per_query() -> None:
    result = ReplayResult("x", 0.5, 10.0, 1.0, 4)
    assert result.cost_per_query() == pytest.approx(2.5)


def test_result_cost_per_query_handles_empty() -> None:
    assert ReplayResult("x", 0.0, 0.0, 0.0, 0).cost_per_query() == 0.0
