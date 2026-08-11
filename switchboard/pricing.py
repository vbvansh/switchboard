"""Simulated cost model.

Local Ollama inference is free. To make budgets and "money saved" meaningful,
each local model wears the price tag of a commercial model of comparable
capability. Every number produced here is SIMULATED and is labelled as such
everywhere it surfaces. Real measured cost - latency, tokens/sec, model
switches - is recorded separately and is never mixed with these figures.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TOKENS_PER_PRICE_UNIT = 1_000_000

DEFAULT_PRICES_PATH = Path(__file__).parent / "prices.json"


@dataclass(frozen=True)
class ModelPrice:
    model: str
    tier: str
    stands_in_for: str
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_mtok
            + completion_tokens * self.output_per_mtok
        ) / TOKENS_PER_PRICE_UNIT


class PriceTable:
    """Model name -> simulated price, plus the baseline used for savings."""

    def __init__(
        self,
        prices: dict[str, ModelPrice],
        default: ModelPrice,
        baseline_model: str,
        ladder: list[str],
    ) -> None:
        self._prices = prices
        self._default = default
        self.baseline_model = baseline_model
        self._warned: set[str] = set()

        # Routable tiers, cheapest first. Sorted defensively rather than trusted
        # from the file: a mis-ordered ladder would silently corrupt every
        # routing decision that assumes index 0 is cheapest.
        self.ladder = sorted(ladder, key=lambda m: self.for_model(m).cost(1000, 1000))

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PRICES_PATH) -> PriceTable:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        def build(model: str, entry: dict) -> ModelPrice:
            return ModelPrice(
                model=model,
                tier=entry["tier"],
                stands_in_for=entry["stands_in_for"],
                input_per_mtok=float(entry["input_per_mtok"]),
                output_per_mtok=float(entry["output_per_mtok"]),
            )

        prices = {
            name: build(name, entry) for name, entry in raw["models"].items()
        }
        return cls(
            prices=prices,
            default=build("default", raw["default"]),
            baseline_model=raw["baseline_model"],
            ladder=list(raw["ladder"]),
        )

    @property
    def cheapest(self) -> str:
        return self.ladder[0]

    @property
    def most_expensive(self) -> str:
        return self.ladder[-1]

    def for_model(self, model: str) -> ModelPrice:
        """Price for a model, falling back to the default entry.

        An unknown model must never be silently free - that would understate
        spend and overstate savings. It falls back to mid-tier pricing and warns
        once. The ledger always records `served_model`, so any row priced this
        way stays traceable.
        """
        price = self._prices.get(model)
        if price is not None:
            return price

        if model not in self._warned:
            logger.warning(
                "No price entry for model %r; using default tier pricing. "
                "Add it to prices.json.",
                model,
            )
            self._warned.add(model)
        return self._default

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Simulated cost of serving this request on `model`."""
        return self.for_model(model).cost(prompt_tokens, completion_tokens)

    def baseline_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """What this request would have cost on the top tier.

        The comparison point for every savings figure: what a company that
        routed everything to its best model would have paid.
        """
        return self.cost(self.baseline_model, prompt_tokens, completion_tokens)

    def known_models(self) -> list[str]:
        return sorted(self._prices)
