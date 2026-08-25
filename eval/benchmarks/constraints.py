"""Routing under hard limits: latency SLAs, quality thresholds, budget caps.

Everything so far has optimised two things - cost and accuracy - and let the
result land wherever it lands. Real deployments have limits that are not
negotiable. A chat box needs an answer in two seconds whether or not a slower
model would have been more accurate; a per-request budget is a per-request
budget.

So the decision becomes two steps:

    1. ELIGIBILITY - drop every model that breaks a hard limit.
    2. CHOICE      - among what survives, take the cheapest one likely enough
                     to be right.

The order matters. Choosing first and checking afterwards would mean rejecting
a good answer you have already paid for.

Two honesty rules run through this file:

* The router knows only a model's TYPICAL latency, learned from training data.
  Any individual request can run slow, so picking a fast model is not the same
  as meeting the SLA. Violations are counted against what actually happened.

* When no model satisfies the limits, the request is not silently served as if
  nothing were wrong. It is served by the closest candidate and flagged, and
  the report says how often that happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eval.benchmarks.learned import SuccessPredictor
from eval.benchmarks.schema import Grid
from switchboard.routing.base import RoutingContext, RoutingDecision, RoutingStrategy

#: SLAs are conventionally stated as a percentile, not a maximum - one slow
#: request should not count as a broken promise.
SLA_PERCENTILE = 95


@dataclass(frozen=True)
class Constraints:
    """Hard limits a routing decision must respect.

    `None` means unconstrained on that axis.
    """

    max_latency_s: float | None = None
    max_cost_usd: float | None = None
    #: Minimum predicted probability that the chosen model answers correctly.
    min_quality: float | None = None

    def describe(self) -> str:
        parts = []
        if self.max_latency_s is not None:
            parts.append(f"latency<={self.max_latency_s:g}s")
        if self.max_cost_usd is not None:
            parts.append(f"cost<=${self.max_cost_usd:g}")
        if self.min_quality is not None:
            parts.append(f"quality>={self.min_quality:.2f}")
        return ", ".join(parts) or "unconstrained"

    @property
    def any_hard_limit(self) -> bool:
        return self.max_latency_s is not None or self.max_cost_usd is not None


@dataclass(frozen=True)
class ModelProfile:
    """What each model typically costs and how long it typically takes.

    Learned from the training split, never the test split. A router in
    production would build this from its own ledger - which Switchboard already
    records for exactly this reason.

    Latency uses the median rather than the mean: one pathological request must
    not make a normally-fast model look slow.
    """

    cost: pd.Series
    #: Median latency - what a typical request takes. For reporting.
    latency: pd.Series
    #: Latency at the SLA percentile - what the promise is actually about.
    #: Eligibility uses THIS, not the median.
    #:
    #: Selecting on the median was the first version of this and it was wrong:
    #: a model answering in 0.33s on a typical request can have a p95 of 11
    #: seconds. Restricting to "fast" models by median produced a set that was
    #: slower at the tail than routing with no SLA at all - the opposite of
    #: what the promise was for. To promise p95 <= B, require p95 <= B.
    latency_tail: pd.Series

    @classmethod
    def from_grid(cls, grid: Grid) -> ModelProfile:
        if grid.latency is None or not grid.latency.notna().any().any():
            empty = pd.Series(dtype=float)
            return cls(cost=grid.cost.mean(), latency=empty, latency_tail=empty)

        return cls(
            cost=grid.cost.mean(),
            latency=grid.latency.median(),
            latency_tail=grid.latency.quantile(SLA_PERCENTILE / 100.0),
        )

    @property
    def has_latency(self) -> bool:
        return bool(len(self.latency)) and bool(self.latency.notna().any())

    def eligible(self, models: list[str], limits: Constraints) -> list[str]:
        """Models that satisfy every hard limit, cheapest first."""
        survivors = []
        for model in sorted(models, key=lambda m: self.cost.get(m, float("inf"))):
            if (
                limits.max_cost_usd is not None
                and self.cost.get(model, float("inf")) > limits.max_cost_usd
            ):
                continue
            if limits.max_latency_s is not None and self.has_latency:
                # The TAIL, not the median - see `latency_tail`.
                tail = self.latency_tail.get(model, float("nan"))
                # An unknown latency is not evidence of being fast enough, so
                # NaN excludes the model rather than waving it through.
                if pd.isna(tail) or tail > limits.max_latency_s:
                    continue
            survivors.append(model)
        return survivors


class ConstrainedRouter(RoutingStrategy):
    """Eligibility first, then the cheapest model likely enough to be right."""

    def __init__(
        self,
        predictor: SuccessPredictor,
        profile: ModelProfile,
        limits: Constraints,
        name: str | None = None,
    ) -> None:
        self.predictor = predictor
        self.profile = profile
        self.limits = limits
        self.name = name or f"constrained[{limits.describe()}]"
        self._probabilities: dict[str, dict[str, float]] = {}
        #: Questions where no model satisfied the hard limits.
        self.unsatisfiable = 0

    def warm(self, texts: list[str]) -> None:
        unseen = [t for t in dict.fromkeys(texts) if t not in self._probabilities]
        if not unseen:
            return
        for text, row in zip(
            unseen, self.predictor.predict_batch(unseen), strict=True
        ):
            self._probabilities[text] = row

    def choose(self, context: RoutingContext) -> RoutingDecision:
        text = context.prompt_text
        probabilities = self._probabilities.get(text) or self.predictor.predict_one(
            text
        )

        candidates = self.profile.eligible(self.predictor.models, self.limits)

        if not candidates:
            # Nothing fits. Serve it with whatever comes closest rather than
            # dropping the request, and record that the promise was broken -
            # an SLA report that hides these is worthless.
            self.unsatisfiable += 1
            fallback = (
                self.profile.latency_tail.idxmin()
                if self.limits.max_latency_s is not None and self.profile.has_latency
                else self.profile.cost.idxmin()
            )
            return RoutingDecision(
                model=fallback,
                strategy=self.name,
                reason=f"no model satisfies {self.limits.describe()}; used closest",
                features={"unsatisfiable": True},
            )

        if self.limits.min_quality is not None:
            for model in candidates:  # already cheapest-first
                if probabilities.get(model, 0.0) >= self.limits.min_quality:
                    return RoutingDecision(
                        model=model,
                        strategy=self.name,
                        reason=(
                            f"cheapest eligible model clearing "
                            f"p>={self.limits.min_quality:.2f}"
                        ),
                        features={"eligible": len(candidates)},
                    )

        # Either no quality floor was set, or nothing eligible cleared it. Take
        # the most likely to succeed from what is allowed - going cheap here
        # would abandon the hard questions the limits were meant to protect.
        best = max(candidates, key=lambda m: probabilities.get(m, 0.0))
        return RoutingDecision(
            model=best,
            strategy=self.name,
            reason=(
                f"best of {len(candidates)} eligible "
                f"(p={probabilities.get(best, 0.0):.2f})"
            ),
            features={"eligible": len(candidates)},
        )


# --- Measuring what actually happened ---------------------------------------


def latency_report(
    grid: Grid, choices: pd.Series, budget_s: float | None
) -> dict[str, float]:
    """How the served requests actually behaved, against the promise.

    Measured on the recorded per-request latency, not on the model averages the
    router used to decide. Those are different numbers, and only the first one
    is what a user experienced.
    """
    if grid.latency is None or not grid.latency.notna().any().any():
        return {}

    aligned = choices.reindex(grid.correct.index)
    observed = np.array(
        [
            grid.latency.at[question, model]
            for question, model in aligned.items()
        ],
        dtype=float,
    )
    observed = observed[~np.isnan(observed)]
    if not len(observed):
        return {}

    report = {
        "mean_latency_s": float(observed.mean()),
        f"p{SLA_PERCENTILE}_latency_s": float(
            np.percentile(observed, SLA_PERCENTILE)
        ),
        "max_latency_s": float(observed.max()),
    }
    if budget_s is not None:
        report["sla_violation_rate"] = float((observed > budget_s).mean())
        report["sla_budget_s"] = budget_s
    return report
