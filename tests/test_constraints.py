"""Routing under hard limits: latency SLAs, budget caps, quality floors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval.benchmarks import BenchmarkFrame, validate
from eval.benchmarks.constraints import (
    SLA_PERCENTILE,
    ConstrainedRouter,
    Constraints,
    ModelProfile,
    latency_report,
)
from eval.benchmarks.features import FeatureExtractor
from eval.benchmarks.learned import SuccessPredictor
from eval.benchmarks.schema import Grid
from switchboard.routing import RoutingContext
from tests.test_cascade import rows_with_answers


def ctx(text: str = "a question") -> RoutingContext:
    return RoutingContext(messages=[{"role": "user", "content": text}])


def make_grid(*records) -> Grid:
    return BenchmarkFrame(validate(rows_with_answers(*records)), "test").grid()


# --- Constraints ------------------------------------------------------------


def test_describe_lists_every_limit() -> None:
    limits = Constraints(max_latency_s=2, max_cost_usd=0.5, min_quality=0.7)
    described = limits.describe()
    assert "latency<=2s" in described
    assert "cost<=$0.5" in described
    assert "quality>=0.70" in described


def test_no_limits_describes_itself_as_unconstrained() -> None:
    assert Constraints().describe() == "unconstrained"


def test_a_quality_floor_alone_is_not_a_hard_limit() -> None:
    """Quality is a preference among eligible models, not an eligibility rule.

    A hard limit can rule every model out; a quality floor never should.
    """
    assert Constraints(min_quality=0.9).any_hard_limit is False
    assert Constraints(max_latency_s=1.0).any_hard_limit is True


# --- Profiles ---------------------------------------------------------------


def spread_grid() -> Grid:
    """`steady` is reliably ~1s. `spiky` is usually fast but has a long tail.

    Their medians say `spiky` is the faster model. Their tails say the opposite,
    and the tail is what an SLA is about.
    """
    records = []
    for i in range(20):
        slow = i >= 18  # 10% of requests are pathological
        records.append(
            ("b", f"q{i:02d}", "steady", 1.0, 0.10, 1.0, 20, "a")
        )
        records.append(
            ("b", f"q{i:02d}", "spiky", 1.0, 0.01, 30.0 if slow else 0.2, 20, "a")
        )
    return make_grid(*records)


def test_profile_records_both_median_and_tail() -> None:
    profile = ModelProfile.from_grid(spread_grid())
    assert profile.latency["spiky"] < profile.latency["steady"]      # median
    assert profile.latency_tail["spiky"] > profile.latency_tail["steady"]  # tail


def test_eligibility_uses_the_tail_not_the_median() -> None:
    """The bug this replaced: selecting on the median chose a model whose p95
    was far outside the budget, so the promise was broken constantly."""
    profile = ModelProfile.from_grid(spread_grid())
    eligible = profile.eligible(["steady", "spiky"], Constraints(max_latency_s=2.0))
    assert eligible == ["steady"]


def test_a_loose_budget_admits_everything() -> None:
    profile = ModelProfile.from_grid(spread_grid())
    eligible = profile.eligible(["steady", "spiky"], Constraints(max_latency_s=60.0))
    assert set(eligible) == {"steady", "spiky"}


def test_eligible_models_come_back_cheapest_first() -> None:
    profile = ModelProfile.from_grid(spread_grid())
    assert profile.eligible(["steady", "spiky"], Constraints())[0] == "spiky"


def test_a_cost_cap_excludes_expensive_models() -> None:
    profile = ModelProfile.from_grid(spread_grid())
    assert profile.eligible(
        ["steady", "spiky"], Constraints(max_cost_usd=0.05)
    ) == ["spiky"]


def test_unknown_latency_is_not_treated_as_fast() -> None:
    """Absence of evidence is not evidence of speed."""
    profile = ModelProfile(
        cost=pd.Series({"a": 0.1}),
        latency=pd.Series({"a": 1.0}),
        latency_tail=pd.Series({"a": float("nan")}),
    )
    assert profile.eligible(["a"], Constraints(max_latency_s=10.0)) == []


def test_a_source_without_latency_yields_an_empty_profile() -> None:
    grid = make_grid(
        ("b", "q1", "cheap", 1.0, 0.01, float("nan"), 20, "a"),
        ("b", "q1", "big", 1.0, 1.00, float("nan"), 50, "a"),
    )
    assert ModelProfile.from_grid(grid).has_latency is False


# --- The router -------------------------------------------------------------


def trained_setup():
    records, texts = [], {}
    for i in range(80):
        hard = i % 2 == 1
        key = ("b", f"q{i:02d}")
        texts[key] = "hard tricky proof" if hard else "easy simple question"
        records.append(
            ("b", f"q{i:02d}", "fast", 0.0 if hard else 1.0, 0.01, 0.5, 20, "a")
        )
        records.append(("b", f"q{i:02d}", "slow", 1.0, 1.00, 9.0, 50, "b"))
    grid = make_grid(*records)
    predictor = SuccessPredictor.train(grid, texts, FeatureExtractor(mode="surface"))
    return predictor, ModelProfile.from_grid(grid), grid, texts


@pytest.fixture(scope="module")
def setup():
    return trained_setup()


def test_without_limits_the_cheapest_capable_model_wins(setup) -> None:
    predictor, profile, _, _ = setup
    router = ConstrainedRouter(predictor, profile, Constraints(min_quality=0.5))
    assert router.choose(ctx("easy simple question")).model == "fast"


def test_a_hard_question_escalates_when_allowed(setup) -> None:
    predictor, profile, _, _ = setup
    router = ConstrainedRouter(predictor, profile, Constraints(min_quality=0.5))
    assert router.choose(ctx("hard tricky proof")).model == "slow"


def test_a_latency_limit_blocks_escalation(setup) -> None:
    """The point of an SLA: the accurate model is off the table if it is slow."""
    predictor, profile, _, _ = setup
    router = ConstrainedRouter(
        predictor, profile, Constraints(max_latency_s=2.0, min_quality=0.5)
    )
    assert router.choose(ctx("hard tricky proof")).model == "fast"


def test_an_impossible_promise_is_flagged_not_hidden(setup) -> None:
    """Serving it silently would make the SLA report a lie."""
    predictor, profile, _, _ = setup
    router = ConstrainedRouter(
        predictor, profile, Constraints(max_latency_s=0.001, min_quality=0.5)
    )
    decision = router.choose(ctx("anything"))
    assert router.unsatisfiable == 1
    assert decision.features["unsatisfiable"] is True
    assert "no model satisfies" in decision.reason


def test_an_impossible_promise_still_answers(setup) -> None:
    """Dropping the request would be worse than answering late."""
    predictor, profile, grid, _ = setup
    router = ConstrainedRouter(
        predictor, profile, Constraints(max_latency_s=0.001)
    )
    assert router.choose(ctx("anything")).model in grid.models


def test_the_fallback_is_the_fastest_model_when_latency_is_the_limit(setup) -> None:
    predictor, profile, _, _ = setup
    router = ConstrainedRouter(
        predictor, profile, Constraints(max_latency_s=0.001)
    )
    assert router.choose(ctx("anything")).model == "fast"


def test_when_nothing_clears_quality_it_takes_the_best_eligible(setup) -> None:
    """Falling back to the cheapest would abandon the hard questions."""
    predictor, profile, _, _ = setup
    router = ConstrainedRouter(predictor, profile, Constraints(min_quality=1.01))
    decision = router.choose(ctx("hard tricky proof"))
    assert "best of" in decision.reason


def test_decisions_explain_themselves(setup) -> None:
    predictor, profile, _, _ = setup
    router = ConstrainedRouter(
        predictor, profile, Constraints(max_latency_s=2.0, min_quality=0.5)
    )
    assert router.choose(ctx("easy simple question")).reason


def test_warming_does_not_change_decisions(setup) -> None:
    """Batching is an optimisation; it must not alter what gets chosen."""
    predictor, profile, grid, texts = setup
    limits = Constraints(min_quality=0.5)
    prompts = ["easy simple question", "hard tricky proof"]

    cold = [ConstrainedRouter(predictor, profile, limits).choose(ctx(p)).model
            for p in prompts]

    warm_router = ConstrainedRouter(predictor, profile, limits)
    warm_router.warm(prompts)
    warm = [warm_router.choose(ctx(p)).model for p in prompts]

    assert cold == warm


# --- Measuring what happened ------------------------------------------------


def test_violations_are_measured_against_real_latency() -> None:
    """Not against the averages the router used to decide.

    A model that is usually fast can still answer slowly, and the user
    experienced the slow answer.
    """
    grid = spread_grid()
    choices = pd.Series("spiky", index=grid.correct.index)
    report = latency_report(grid, choices, budget_s=2.0)
    # 2 of 20 recorded requests took 30s.
    assert report["sla_violation_rate"] == pytest.approx(0.1)


def test_report_includes_the_sla_percentile() -> None:
    grid = spread_grid()
    choices = pd.Series("steady", index=grid.correct.index)
    report = latency_report(grid, choices, budget_s=2.0)
    assert f"p{SLA_PERCENTILE}_latency_s" in report
    assert report["sla_violation_rate"] == 0.0


def test_no_budget_means_no_violation_figure() -> None:
    grid = spread_grid()
    choices = pd.Series("steady", index=grid.correct.index)
    assert "sla_violation_rate" not in latency_report(grid, choices, None)


def test_a_source_without_latency_reports_nothing() -> None:
    grid = make_grid(
        ("b", "q1", "cheap", 1.0, 0.01, float("nan"), 20, "a"),
        ("b", "q1", "big", 1.0, 1.00, float("nan"), 50, "a"),
    )
    choices = pd.Series("cheap", index=grid.correct.index)
    assert latency_report(grid, choices, budget_s=1.0) == {}


def test_report_survives_missing_measurements() -> None:
    """Some rows can lack a timing without invalidating the rest."""
    grid = make_grid(
        ("b", "q1", "m", 1.0, 0.01, 1.0, 20, "a"),
        ("b", "q2", "m", 1.0, 0.01, float("nan"), 20, "a"),
    )
    choices = pd.Series("m", index=grid.correct.index)
    report = latency_report(grid, choices, budget_s=2.0)
    assert np.isfinite(report["mean_latency_s"])
