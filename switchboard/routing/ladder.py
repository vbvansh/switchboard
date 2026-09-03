"""Routing with no trained model at all.

WHAT THIS IS FOR. Somebody has just installed Switchboard. They have no traffic,
no ratings, and no trained router. Something has to decide which model answers
their first request, and "nothing, fall back to a hardcoded default" is what the
project did until now.

WHY IT DOES NOT GUESS. The obvious temptation is a heuristic: long prompt means
hard, code means hard, send those to the expensive model. This project measured
exactly that. A hand-written keyword heuristic scored 77.8% on one benchmark and
**57.9% on another - worse than simply always using the cheapest model**. A rule
that helps on one workload and hurts on another is worse than no rule, because
you cannot tell in advance which one you are on.

Two further experiments confirmed there is nothing better available up front.
Predicting a question's difficulty from its text does not transfer between
domains: within a suite the ranking correlation was 0.077, which is zero.

So this router guesses NOTHING. It applies only facts:

    1. the cheapest model on the ladder, always
    2. unless the prompt does not fit in its context window - a hard limit, not
       an opinion
    3. unless the caller asked for something else, which always wins

Quality is not predicted here. It is handled AFTER the answer arrives, by
`switchboard/verification.py`: look at what came back, and escalate if it
obviously failed. Checking beats guessing, because checking needs only the
answer and guessing needs knowledge of the world.

WHAT THIS COSTS. Every request starts at the cheapest model. On the benchmarks,
always-cheapest scored 84.0% against 86.8% for always using the best model, at
1/20th of the price - before any escalation. Escalation recovers some of that
gap on the requests where failure is visible. It cannot recover the rest, and
nothing in this file pretends otherwise.
"""

from __future__ import annotations

import logging

from switchboard.catalog import ModelCatalog
from switchboard.routing.base import RoutingContext, RoutingDecision

logger = logging.getLogger(__name__)

#: Rough characters per token, for checking a prompt against a context window.
#: Deliberately pessimistic - assuming fewer characters per token means
#: assuming more tokens, so a borderline prompt is sent somewhere roomier
#: rather than being rejected by the provider.
CHARS_PER_TOKEN = 3.0

#: Fraction of a context window a prompt may occupy before the model is
#: considered too small. The rest is for the answer: a prompt filling 95% of
#: the window technically fits and leaves nowhere to reply.
PROMPT_BUDGET = 0.6


class LadderRouter:
    """Cheapest model that can physically hold the request.

    Implements the same surface as the trained router, so `api.py` does not
    care which one it has, and `/health` can report which is in use.
    """

    #: Always usable. Unlike the trained router there is no artifact to load,
    #: no models to map, and nothing that can be stale.
    enabled = True

    def __init__(self, catalog: ModelCatalog, available: list[str]) -> None:
        self.catalog = catalog
        # Ladder order is cheapest-first and validated when providers.yaml
        # loads, so the first model that fits is also the cheapest that fits.
        self.models = [m for m in catalog.ladder if m in set(available)]
        if not self.models:
            # Nothing on the ladder is servable. Fall back to whatever the pool
            # does have, sorted by price, rather than refusing to route at all.
            self.models = sorted(
                available, key=lambda m: catalog.models[m].output_per_mtok
            )

    @property
    def routable_models(self) -> list[str]:
        return list(self.models)

    @property
    def metadata(self):
        class _Metadata:
            models = self.models

            def describe(self_inner) -> str:
                return "untrained ladder policy - no prediction, no training"

        return _Metadata()

    def fits(self, model: str, prompt_chars: int) -> bool:
        spec = self.catalog.models.get(model)
        if spec is None:
            return False
        estimated_tokens = prompt_chars / CHARS_PER_TOKEN
        return estimated_tokens <= spec.context_window * PROMPT_BUDGET

    def choose(self, context: RoutingContext, limits=None) -> RoutingDecision:
        if not self.models:
            raise ValueError("LadderRouter has no models to choose between.")

        prompt_chars = len(context.prompt_text or "")
        allowed = self._within_limits(limits)

        for model in allowed:
            if self.fits(model, prompt_chars):
                return RoutingDecision(
                    model=model,
                    strategy="ladder",
                    reason=(
                        f"cheapest model that fits ({prompt_chars:,} chars); "
                        "no prediction was made - quality is checked after the "
                        "answer arrives"
                    ),
                )

        # Nothing on the ladder is big enough. The largest window is the best
        # remaining option, and saying so beats silently truncating somebody's
        # document.
        roomiest = max(
            allowed or self.models,
            key=lambda m: self.catalog.models[m].context_window,
        )
        return RoutingDecision(
            model=roomiest,
            strategy="ladder",
            reason=(
                f"prompt is {prompt_chars:,} characters and fits no model "
                "comfortably; used the largest context window available"
            ),
        )

    def next_model(self, current: str) -> str | None:
        """The next model up the ladder, for escalation. None at the top.

        Escalation walks this ladder one rung at a time rather than jumping
        straight to the most expensive model. A jump would turn every detected
        failure into the largest possible bill, and the benchmarks showed the
        cheapest-to-dearest spread is where the savings live.
        """
        if current not in self.models:
            return self.models[0] if self.models else None
        index = self.models.index(current)
        return self.models[index + 1] if index + 1 < len(self.models) else None

    def _within_limits(self, limits) -> list[str]:
        """Apply per-request caps sent as headers.

        A cost cap is a fact about what the caller will accept, not a guess, so
        it belongs here. If it excludes everything it is ignored rather than
        obeyed - failing a request because its budget was unsatisfiable is
        worse than serving it and recording the overrun.
        """
        if limits is None:
            return list(self.models)

        allowed = list(self.models)
        max_cost = getattr(limits, "max_cost_usd", None)
        if max_cost:
            affordable = [
                m
                for m in allowed
                # A nominal 1k in / 500 out request, purely to compare models
                # against a cap. The real cost is recorded from real tokens.
                if self.catalog.cost(m, 1000, 500) <= max_cost
            ]
            allowed = affordable or allowed

        max_latency = getattr(limits, "max_latency_s", None)
        if max_latency:
            quick = [
                m
                for m in allowed
                if (self.catalog.models[m].typical_latency_s or 0) <= max_latency
            ]
            allowed = quick or allowed

        return allowed


def build_ladder(catalog: ModelCatalog, available: list[str]) -> LadderRouter | None:
    """A ladder router, or None when there is nothing to route between."""
    if not available:
        return None
    router = LadderRouter(catalog, available)
    if len(router.models) < 2:
        logger.info(
            "Only %d model(s) available; ladder routing needs at least 2.",
            len(router.models),
        )
        return None
    return router
