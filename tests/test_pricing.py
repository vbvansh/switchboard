"""Simulated cost arithmetic."""

from __future__ import annotations

import pytest

from switchboard.pricing import PriceTable


def test_cost_is_per_million_tokens(prices: PriceTable) -> None:
    # qwen2.5:7b = $3.00 input / $15.00 output per million tokens.
    assert prices.cost("qwen2.5:7b", 1_000_000, 0) == pytest.approx(3.00)
    assert prices.cost("qwen2.5:7b", 0, 1_000_000) == pytest.approx(15.00)
    assert prices.cost("qwen2.5:7b", 500_000, 100_000) == pytest.approx(3.00)


def test_zero_tokens_cost_nothing(prices: PriceTable) -> None:
    assert prices.cost("qwen2.5:7b", 0, 0) == 0.0


def test_cheap_tier_is_much_cheaper(prices: PriceTable) -> None:
    small = prices.cost("qwen2.5:1.5b", 100_000, 50_000)
    large = prices.cost("qwen2.5:7b", 100_000, 50_000)
    assert small < large / 20


def test_baseline_uses_the_top_tier(prices: PriceTable) -> None:
    assert prices.baseline_model == "qwen2.5:7b"
    assert prices.baseline_cost(1_000_000, 0) == prices.cost(
        "qwen2.5:7b", 1_000_000, 0
    )


def test_unknown_model_is_never_free(prices: PriceTable) -> None:
    """Silently pricing an unknown model at zero would overstate savings."""
    assert prices.cost("some-model-we-never-configured", 1_000_000, 0) > 0


def test_every_local_model_is_priced(prices: PriceTable) -> None:
    """The ladder from the README must all be present in prices.json."""
    for model in (
        "qwen2.5:1.5b",
        "qwen2.5:3b",
        "qwen3:4b",
        "qwen2.5:7b",
        "stable-code:3b-code-q4_0",
    ):
        assert model in prices.known_models()


def test_tiers_are_monotonically_priced(prices: PriceTable) -> None:
    """A bigger model must never be cheaper - that would break routing logic."""
    ladder = ["qwen2.5:1.5b", "qwen2.5:3b", "qwen3:4b", "qwen2.5:7b"]
    costs = [prices.cost(m, 100_000, 100_000) for m in ladder]
    assert costs == sorted(costs)
