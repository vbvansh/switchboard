"""Baseline strategies and the tier ladder."""

from __future__ import annotations

import pytest

from switchboard.pricing import PriceTable
from switchboard.routing import (
    BASELINE_NAMES,
    AlwaysModel,
    KeywordHeuristic,
    RandomModel,
    RoutingContext,
    build_strategy,
)


def ctx(text: str) -> RoutingContext:
    return RoutingContext(messages=[{"role": "user", "content": text}])


# --- Ladder ----------------------------------------------------------------


def test_ladder_is_ordered_cheapest_first(prices: PriceTable) -> None:
    costs = [prices.cost(m, 1000, 1000) for m in prices.ladder]
    assert costs == sorted(costs)


def test_ladder_endpoints(prices: PriceTable) -> None:
    assert prices.cheapest == "qwen2.5:1.5b"
    assert prices.most_expensive == "qwen2.5:7b"


def test_ladder_is_sorted_even_if_the_file_is_not(prices: PriceTable) -> None:
    """A mis-ordered ladder in prices.json must not corrupt routing."""
    shuffled = PriceTable(
        prices={m: prices.for_model(m) for m in prices.known_models()},
        default=prices.for_model("unknown-model"),
        baseline_model="qwen2.5:7b",
        ladder=["qwen2.5:7b", "qwen2.5:1.5b", "qwen3:4b", "qwen2.5:3b"],
    )
    assert shuffled.cheapest == "qwen2.5:1.5b"
    assert shuffled.most_expensive == "qwen2.5:7b"


# --- Context ---------------------------------------------------------------


def test_system_messages_are_excluded_from_the_prompt_text() -> None:
    """Boilerplate system prompts would swamp any signal about difficulty."""
    context = RoutingContext(
        messages=[
            {"role": "system", "content": "You are a helpful assistant. " * 50},
            {"role": "user", "content": "hi"},
        ]
    )
    assert context.prompt_text == "hi"


def test_last_user_message_is_found() -> None:
    context = RoutingContext(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
    )
    assert context.last_user_message == "second"


# --- Always ----------------------------------------------------------------


def test_always_ignores_the_prompt() -> None:
    strategy = AlwaysModel("qwen2.5:7b")
    assert strategy.choose(ctx("anything")).model == "qwen2.5:7b"
    assert strategy.choose(ctx("x" * 5000)).model == "qwen2.5:7b"


# --- Random ----------------------------------------------------------------


def test_random_is_reproducible(prices: PriceTable) -> None:
    """An unseeded baseline would move between runs and ruin comparisons."""
    first = [RandomModel(prices.ladder, seed=7).choose(ctx("q")).model for _ in range(1)]
    second = [
        RandomModel(prices.ladder, seed=7).choose(ctx("q")).model for _ in range(1)
    ]
    assert first == second


def test_random_stays_inside_the_ladder(prices: PriceTable) -> None:
    strategy = RandomModel(prices.ladder, seed=1)
    picks = {strategy.choose(ctx(f"question {i}")).model for i in range(50)}
    assert picks <= set(prices.ladder)


def test_random_actually_varies(prices: PriceTable) -> None:
    strategy = RandomModel(prices.ladder, seed=1)
    picks = {strategy.choose(ctx(f"question {i}")).model for i in range(50)}
    assert len(picks) > 1


# --- Keyword heuristic -----------------------------------------------------


def test_short_simple_prompts_go_cheap(prices: PriceTable) -> None:
    strategy = KeywordHeuristic(prices.ladder)
    assert strategy.choose(ctx("list the days")).model == prices.cheapest


def test_reasoning_words_escalate(prices: PriceTable) -> None:
    strategy = KeywordHeuristic(prices.ladder)
    cheap = strategy.choose(ctx("list the days")).model
    hard = strategy.choose(
        ctx(
            "Explain why this algorithm has that complexity and derive the "
            "worst case, then compare the trade-off against the alternative "
            "design and prove your reasoning step-by-step."
        )
    ).model
    assert prices.ladder.index(hard) > prices.ladder.index(cheap)


def test_decisions_stay_inside_the_ladder(prices: PriceTable) -> None:
    strategy = KeywordHeuristic(prices.ladder)
    for text in ["", "x", "why " * 500, "prove derive explain analyse debug"]:
        assert strategy.choose(ctx(text)).model in prices.ladder


def test_keyword_heuristic_cannot_tell_these_apart(prices: PriceTable) -> None:
    """The documented weakness, pinned as a test.

    Both contain 'why'; one is trivia and one needs real reasoning. Milestone 4
    exists to beat this.
    """
    strategy = KeywordHeuristic(prices.ladder)
    easy = strategy.choose(ctx("why is the sky blue")).model
    hard = strategy.choose(ctx("why does this code deadlock")).model
    assert easy == hard


def test_empty_ladder_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty ladder"):
        KeywordHeuristic([])


# --- Registry --------------------------------------------------------------


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_every_baseline_can_be_built(name: str, prices: PriceTable) -> None:
    strategy = build_strategy(name, prices)
    assert strategy.choose(ctx("hello")).model in prices.ladder


def test_always_prefix_builds_an_arbitrary_model(prices: PriceTable) -> None:
    assert build_strategy("always:qwen3:4b", prices).choose(ctx("x")).model == "qwen3:4b"


def test_unknown_strategy_names_are_rejected(prices: PriceTable) -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_strategy("magic", prices)


def test_decisions_explain_themselves(prices: PriceTable) -> None:
    """Without a reason, a wrong routing decision cannot be debugged."""
    decision = build_strategy("keyword", prices).choose(ctx("explain why"))
    assert decision.reason
    assert decision.strategy == "keyword"
