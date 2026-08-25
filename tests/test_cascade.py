"""Cascades: call cheap, inspect the answer, escalate if unconvinced.

The load-bearing property in this file is the ACCOUNTING. A cascade that
escalates has paid for both calls, and charging only for the final one is the
easiest way to make a cascade look better than it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval.benchmarks import BenchmarkError, BenchmarkFrame, validate
from eval.benchmarks.cascade import (
    VerifierCascade,
    agreement_paths,
    cascades_for_thresholds,
    cheapest_models,
    has_answers,
    observation_features,
    strongest_model,
)
from eval.benchmarks.features import FeatureExtractor
from eval.benchmarks.schema import COLUMNS, Grid, answer_key


def rows_with_answers(*records) -> pd.DataFrame:
    """(benchmark, query, model, correct, cost, latency, tokens, prediction)."""
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
                "output_tokens": tokens,
                "latency_s": latency,
                "prediction": pred,
            }
            for b, q, m, c, cost, latency, tokens, pred in records
        ],
        columns=list(COLUMNS),
    )


def make_grid(*records) -> Grid:
    return BenchmarkFrame(validate(rows_with_answers(*records)), "test").grid()


# --- The answer fingerprint -------------------------------------------------


def test_short_answers_are_kept_verbatim() -> None:
    assert answer_key("B") == "b"
    assert answer_key("  The Answer ") == "the answer"


def test_long_answers_become_a_hash() -> None:
    """A 1.7 MB code submission must not be stored in the cache verbatim."""
    key = answer_key("x" * 5000)
    assert key.startswith("h:")
    assert len(key) < 30


def test_identical_long_answers_hash_identically() -> None:
    """Equality is the only thing cascades ask of an answer."""
    assert answer_key("y" * 5000) == answer_key("y" * 5000)


def test_different_long_answers_do_not_collide() -> None:
    """Truncating instead of hashing would make these look identical."""
    assert answer_key("a" * 5000 + "one") != answer_key("a" * 5000 + "two")


def test_missing_answers_are_empty() -> None:
    assert answer_key(None) == ""
    assert answer_key("") == ""


# --- Path accounting --------------------------------------------------------


@pytest.fixture
def grid() -> Grid:
    """Three models. q1 the cheap model gets right, q2 it does not."""
    return make_grid(
        ("b", "q1", "cheap", 1.0, 0.01, 1.0, 20, "a"),
        ("b", "q1", "peer", 1.0, 0.02, 1.0, 20, "a"),
        ("b", "q1", "big", 1.0, 1.00, 3.0, 50, "a"),
        ("b", "q2", "cheap", 0.0, 0.01, 1.0, 90, "b"),
        ("b", "q2", "peer", 0.0, 0.02, 1.0, 90, "c"),
        ("b", "q2", "big", 1.0, 1.00, 3.0, 50, "d"),
    )


def test_a_single_call_path_costs_one_call(grid: Grid) -> None:
    paths = pd.Series([("cheap",), ("cheap",)], index=grid.correct.index)
    scored = grid.score_for_paths(paths)
    assert scored["cost_usd"] == pytest.approx(0.02)
    assert scored["calls_per_query"] == 1.0


def test_escalating_pays_for_both_calls(grid: Grid) -> None:
    """The whole point. Charging only the final model would understate this."""
    paths = pd.Series([("cheap",), ("cheap", "big")], index=grid.correct.index)
    scored = grid.score_for_paths(paths)
    assert scored["cost_usd"] == pytest.approx(0.01 + 0.01 + 1.00)
    assert scored["calls_per_query"] == 1.5


def test_correctness_comes_from_the_last_model_called(grid: Grid) -> None:
    """That is the answer the cascade actually returns."""
    paths = pd.Series([("cheap",), ("cheap", "big")], index=grid.correct.index)
    assert grid.score_for_paths(paths)["accuracy"] == pytest.approx(1.0)


def test_latency_sums_over_the_path(grid: Grid) -> None:
    paths = pd.Series([("cheap",), ("cheap", "big")], index=grid.correct.index)
    scored = grid.score_for_paths(paths)
    assert scored["mean_latency_s"] == pytest.approx((1.0 + 4.0) / 2)


def test_an_empty_path_is_rejected(grid: Grid) -> None:
    paths = pd.Series([(), ("cheap",)], index=grid.correct.index)
    with pytest.raises(BenchmarkError, match="Empty routing path"):
        grid.score_for_paths(paths)


def test_an_unknown_model_in_a_path_is_rejected(grid: Grid) -> None:
    paths = pd.Series([("imaginary",), ("cheap",)], index=grid.correct.index)
    with pytest.raises(BenchmarkError, match="not in this grid"):
        grid.score_for_paths(paths)


def test_a_missing_path_is_rejected(grid: Grid) -> None:
    paths = pd.Series([("cheap",)], index=grid.correct.index[:1])
    with pytest.raises(BenchmarkError, match="no routing path"):
        grid.score_for_paths(paths)


# --- Model selection --------------------------------------------------------


def test_cheapest_and_strongest_are_identified(grid: Grid) -> None:
    assert cheapest_models(grid, 2) == ["cheap", "peer"]
    assert strongest_model(grid) == "big"


def test_has_answers_detects_a_source_without_predictions() -> None:
    """xRouteBench records full response text, not a comparable answer."""
    blank = make_grid(
        ("b", "q1", "cheap", 1.0, 0.01, 1.0, 20, ""),
        ("b", "q1", "big", 1.0, 1.00, 3.0, 50, ""),
    )
    assert has_answers(blank) is False
    assert has_answers(grid_with_answers()) is True


def grid_with_answers() -> Grid:
    return make_grid(
        ("b", "q1", "cheap", 1.0, 0.01, 1.0, 20, "a"),
        ("b", "q1", "big", 1.0, 1.00, 3.0, 50, "a"),
    )


# --- Agreement cascade ------------------------------------------------------


def test_agreement_accepts_without_escalating(grid: Grid) -> None:
    """Two models reaching the same answer is evidence it is right."""
    paths = agreement_paths(grid)
    assert paths[("b", "q1")] == ("cheap", "peer")


def test_disagreement_escalates(grid: Grid) -> None:
    paths = agreement_paths(grid)
    assert paths[("b", "q2")] == ("cheap", "peer", "big")


def test_two_blank_answers_are_not_agreement() -> None:
    """Unknown is not confirmation. Two blanks must escalate, not accept.

    Treating "" == "" as agreement would make every unparseable answer look
    like two models confirming each other - confidently accepting exactly the
    cases where the cheap model produced nothing usable.
    """
    blank = make_grid(
        ("b", "q1", "cheap", 1.0, 0.01, 1.0, 20, ""),
        ("b", "q1", "peer", 1.0, 0.02, 1.0, 20, ""),
        ("b", "q1", "big", 1.0, 1.00, 3.0, 50, "a"),
    )
    assert agreement_paths(blank)[("b", "q1")] == ("cheap", "peer", "big")


def test_agreement_refuses_a_source_with_no_answers_at_all() -> None:
    """Nothing to compare, so it says so rather than guessing."""
    nothing = make_grid(
        ("b", "q1", "cheap", 1.0, 0.01, 1.0, 20, ""),
        ("b", "q1", "big", 1.0, 1.00, 3.0, 50, ""),
    )
    with pytest.raises(ValueError, match="no parsed answers"):
        agreement_paths(nothing)


def test_agreement_always_pays_for_two_calls(grid: Grid) -> None:
    scored = grid.score_for_paths(agreement_paths(grid))
    assert scored["calls_per_query"] == pytest.approx(2.5)  # (2 + 3) / 2


# --- Observation features ---------------------------------------------------


def test_observations_exclude_correctness(grid: Grid) -> None:
    """A cascade may look at the answer's shape, never at its grade.

    If the correctness label leaked in here, the cascade would be an oracle
    wearing a disguise and every result would be meaningless.
    """
    easy = observation_features(grid, "cheap", "peer")[0]
    hard = observation_features(grid, "cheap", "peer")[1]
    # q1 and q2 differ in token count and agreement, both observable.
    assert not np.allclose(easy, hard)
    # Three columns: log tokens, tokens vs typical, agreement flag.
    assert observation_features(grid, "cheap", "peer").shape == (2, 3)


def test_agreement_flag_is_set_only_when_answers_match(grid: Grid) -> None:
    observations = observation_features(grid, "cheap", "peer")
    assert observations[0, 2] == 1.0  # q1 agrees
    assert observations[1, 2] == 0.0  # q2 disagrees


def test_observations_work_without_a_peer(grid: Grid) -> None:
    observations = observation_features(grid, "cheap", None)
    assert observations.shape == (2, 3)
    assert (observations[:, 2] == 0).all()


# --- Learned verifier -------------------------------------------------------


def training_grid(n: int = 120):
    """Cheap model fails the 'hard' questions; that must be learnable."""
    records = []
    texts = {}
    for i in range(n):
        hard = i % 2 == 1
        key = ("b", f"q{i:03d}")
        texts[key] = "hard tricky proof problem" if hard else "easy simple question"
        records.append(
            ("b", f"q{i:03d}", "cheap", 0.0 if hard else 1.0, 0.01, 1.0,
             200 if hard else 20, "x" if hard else "a")
        )
        records.append(
            ("b", f"q{i:03d}", "peer", 0.0 if hard else 1.0, 0.02, 1.0,
             200 if hard else 20, "y" if hard else "a")
        )
        records.append(("b", f"q{i:03d}", "big", 1.0, 1.00, 3.0, 50, "a"))
    return make_grid(*records), texts


@pytest.fixture(scope="module")
def verifier():
    grid, texts = training_grid()
    cascade = VerifierCascade.train(
        grid, texts, FeatureExtractor(mode="surface"), threshold=0.5
    )
    return cascade, grid, texts


def test_verifier_picks_cheap_and_strong_ends(verifier) -> None:
    cascade, _, _ = verifier
    assert cascade.first == "cheap"
    assert cascade.escalate_to == "big"


def test_verifier_learns_when_the_cheap_model_fails(verifier) -> None:
    """The load-bearing test: it must tell easy from hard after one call."""
    cascade, grid, texts = verifier
    confidence = cascade.confidence(grid, texts)
    easy = confidence[::2].mean()   # even indices are the easy questions
    hard = confidence[1::2].mean()
    assert easy > hard


def test_verifier_escalates_only_where_needed(verifier) -> None:
    cascade, grid, texts = verifier
    paths = cascade.paths(grid, texts)
    lengths = paths.map(len)
    assert lengths.min() == 1  # some accepted after one call
    assert lengths.max() == 2  # others escalated


def test_raising_the_threshold_escalates_more(verifier) -> None:
    """Monotonicity: demanding more confidence means paying more."""
    cascade, grid, texts = verifier
    spend = []
    for threshold in (0.1, 0.5, 0.9):
        variant = cascades_for_thresholds(cascade, [threshold])[0]
        spend.append(grid.score_for_paths(variant.paths(grid, texts))["cost_usd"])
    assert spend == sorted(spend)


def test_threshold_sweep_reuses_one_trained_classifier(verifier) -> None:
    cascade, _, _ = verifier
    variants = cascades_for_thresholds(cascade, [0.2, 0.5, 0.8])
    assert [v.threshold for v in variants] == [0.2, 0.5, 0.8]
    assert all(v.classifier is cascade.classifier for v in variants)


def test_a_cascade_never_costs_less_than_its_first_call(verifier) -> None:
    """It always pays for the cheap model, even when it accepts the answer."""
    cascade, grid, texts = verifier
    scored = grid.score_for_paths(cascade.paths(grid, texts))
    assert scored["cost_usd"] >= grid.cost["cheap"].sum()
    assert scored["calls_per_query"] >= 1.0
