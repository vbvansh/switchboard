"""Shadow mode: measure what routing WOULD have done, without doing it.

Nobody sensible points a new routing system at production traffic and hopes.
So shadow mode runs the router on every request, records the model it would
have chosen and what that would have cost - and then ignores the decision and
serves the request exactly as it would have been served anyway.

After a week of real traffic the operator has a report: "routing would have cut
this bill by 61%, and here are the requests where it would have used a weaker
model." That is a decision they can actually make, on their own workload,
having risked nothing.

It also solves the problem found in C.4. A router trained on public benchmarks
does not understand short chat prompts. The fix is to train on the traffic you
actually serve, and shadow mode is what collects that traffic with the router's
opinion attached.

TWO HONEST LIMITS, stated here because they are invisible in the numbers:

**The shadow cost is an estimate.** The shadow model was never called, so its
token count is unknown. The estimate reuses the tokens the real model actually
produced and prices them at the shadow model's rates. A more verbose model
would really have cost more than this says, and a terser one less. It is a
projection, not a measurement, and it is labelled that way everywhere.

**Shadow mode cannot tell you whether quality would have suffered.** It has no
answer from the shadow model to grade. What it can report is the router's own
predicted probability that the shadow model would have been right - which is a
forecast, not evidence. Anyone reading the savings figure needs to see that
number next to it.
"""

from __future__ import annotations

from dataclasses import dataclass

from switchboard.catalog import ModelCatalog


@dataclass(frozen=True)
class ShadowDecision:
    """What routing would have done with a request it did not get to route."""

    model: str
    reason: str
    #: Estimated cost, from the REAL request's tokens at the shadow model's
    #: prices. See the module docstring: a projection, not a measurement.
    estimated_cost_usd: float

    @property
    def is_estimate(self) -> bool:
        return True


def estimate_cost(
    catalog: ModelCatalog, model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """What `model` would have charged for this many tokens.

    Uses the token counts the served model actually produced. That is the only
    honest thing available - the shadow model was never called, so its own
    token count does not exist.
    """
    return catalog.cost(model, prompt_tokens, completion_tokens)


@dataclass
class ShadowSummary:
    """Aggregate over a period of shadowed traffic."""

    requests: int = 0
    actual_cost_usd: float = 0.0
    shadow_cost_usd: float = 0.0
    #: Requests where routing would have used a DIFFERENT model.
    changed: int = 0
    #: Of those, how many would have moved to a cheaper model.
    downgraded: int = 0
    upgraded: int = 0
    model_counts: dict[str, int] | None = None

    @property
    def projected_saving_usd(self) -> float:
        return self.actual_cost_usd - self.shadow_cost_usd

    @property
    def projected_saving_pct(self) -> float:
        if self.actual_cost_usd <= 0:
            return 0.0
        return 100.0 * self.projected_saving_usd / self.actual_cost_usd

    @property
    def changed_pct(self) -> float:
        return 100.0 * self.changed / self.requests if self.requests else 0.0

    def describe(self) -> str:
        if not self.requests:
            return "No shadowed requests recorded yet."
        direction = "save" if self.projected_saving_usd >= 0 else "COST"
        return (
            f"Over {self.requests:,} requests, routing would {direction} "
            f"${abs(self.projected_saving_usd):,.4f} "
            f"({abs(self.projected_saving_pct):.1f}%). It would have chosen a "
            f"different model {self.changed:,} times "
            f"({self.changed_pct:.0f}%)."
        )


def summarise(rows) -> ShadowSummary:
    """Aggregate ledger rows that carry a shadow decision.

    Rows without one are skipped rather than counted as "no change". A request
    served before shadow mode was switched on has no opinion attached, and
    treating that as agreement would quietly dilute the result.
    """
    summary = ShadowSummary(model_counts={})

    for row in rows:
        shadow_model = getattr(row, "shadow_model", None)
        if not shadow_model:
            continue

        summary.requests += 1
        summary.actual_cost_usd += row.simulated_cost_usd or 0.0
        summary.shadow_cost_usd += row.shadow_cost_usd or 0.0
        summary.model_counts[shadow_model] = (
            summary.model_counts.get(shadow_model, 0) + 1
        )

        if shadow_model != row.served_model:
            summary.changed += 1
            if (row.shadow_cost_usd or 0.0) < (row.simulated_cost_usd or 0.0):
                summary.downgraded += 1
            else:
                summary.upgraded += 1

    return summary
