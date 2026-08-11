"""Routing strategies and the registry that names them."""

from __future__ import annotations

from switchboard.pricing import PriceTable
from switchboard.routing.base import (
    RoutingContext,
    RoutingDecision,
    RoutingStrategy,
)
from switchboard.routing.baselines import (
    AlwaysModel,
    KeywordHeuristic,
    RandomModel,
)

#: Strategies compared in milestone 3. Milestone 4 adds the learned ones here.
BASELINE_NAMES = ("always-cheap", "always-expensive", "random", "keyword")


def build_strategy(name: str, prices: PriceTable, seed: int = 0) -> RoutingStrategy:
    """Construct a strategy by name.

    Names rather than classes so a strategy can be selected from a CLI flag, a
    config file, or a results file without importing anything.
    """
    if name == "always-cheap":
        return AlwaysModel(prices.cheapest, name="always-cheap")
    if name == "always-expensive":
        return AlwaysModel(prices.most_expensive, name="always-expensive")
    if name == "random":
        return RandomModel(prices.ladder, seed=seed)
    if name == "keyword":
        return KeywordHeuristic(prices.ladder)
    if name.startswith("always:"):
        return AlwaysModel(name.split(":", 1)[1])

    raise ValueError(
        f"Unknown strategy {name!r}. Available: {', '.join(BASELINE_NAMES)} "
        "or always:<model>."
    )


__all__ = [
    "BASELINE_NAMES",
    "AlwaysModel",
    "KeywordHeuristic",
    "RandomModel",
    "RoutingContext",
    "RoutingDecision",
    "RoutingStrategy",
    "build_strategy",
]
