"""The router that actually serves traffic.

Everything in `eval/` measures routing offline against recorded answers. This
is the same decision made for real, against a live catalog of models, on a
request that a person is waiting for.

Three things separate it from the evaluation-side router:

* **It maps names.** A router trained on public benchmarks knows
  `qwen2.5-7b-instruct`; the operator's catalog has `qwen2.5:7b`. Models
  declare `benchmark_alias` in providers.yaml to say which they stand in for.

* **It degrades instead of failing.** If the artifact is missing, or none of its
  models are in the catalog, requests must still be served - just not routed.
  A router that takes the service down when its model file is stale is worse
  than no router.

* **It takes limits per request.** A chat box asks for two seconds; a nightly
  batch job does not care. Those arrive as headers so the request body stays a
  valid OpenAI payload.

A LIMITATION worth stating plainly, because it is invisible in the offline
numbers. A router trained on public benchmarks learned from ~700-character
prompts carrying system instructions and multiple-choice options. Measured on
that distribution it discriminates well - predictions spread from 0.02 to 0.88.
Shown a 34-character chat message it has no idea, and returns roughly the same
probability for everything, so every request goes to the cheapest model.

That is distribution shift, not a defect in the model. The fix is to train on
the traffic you actually serve, which is why the ledger records prompts (behind
an explicit opt-in) and which model answered. Until then, a benchmark-trained
router is a reasonable default for benchmark-shaped work and close to useless
for short conversational prompts. `routing_reason` on every ledger row is what
makes that visible rather than mysterious.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from switchboard import paths
from switchboard.catalog import ModelCatalog
from switchboard.routing.artifact import RouterMetadata
from switchboard.routing.base import RoutingContext, RoutingDecision, RoutingStrategy

logger = logging.getLogger(__name__)


def shipped_router_path() -> Path:
    """A router bundled inside the package, if one was built.

    Written by `switchboard bench train-broad --save` and committed as package
    data. It knows public benchmark model names, so it only drives a catalog
    whose models declare a `benchmark_alias` - which is why `switchboard router
    info` reports what actually mapped.
    """
    return paths.PACKAGE_ROOT / "router.joblib"

#: Per-request limits, sent as headers so the body stays a valid OpenAI request.
HEADER_MAX_LATENCY = "x-switchboard-max-latency"
HEADER_MIN_QUALITY = "x-switchboard-min-quality"
HEADER_MAX_COST = "x-switchboard-max-cost"


@dataclass(frozen=True)
class RequestLimits:
    """What this particular caller asked for."""

    max_latency_s: float | None = None
    min_quality: float | None = None
    max_cost_usd: float | None = None

    @classmethod
    def from_headers(cls, headers) -> RequestLimits:
        """Parse routing hints from request headers.

        A malformed value is ignored rather than rejected. These are hints on
        an otherwise valid request, and failing someone's chat completion
        because they sent `max-latency: fast` would be a poor trade.
        """

        def number(name: str) -> float | None:
            raw = headers.get(name)
            if raw is None:
                return None
            try:
                value = float(raw)
            except (TypeError, ValueError):
                logger.warning("Ignoring unparseable %s: %r", name, raw)
                return None
            return value if value > 0 else None

        return cls(
            max_latency_s=number(HEADER_MAX_LATENCY),
            min_quality=number(HEADER_MIN_QUALITY),
            max_cost_usd=number(HEADER_MAX_COST),
        )

    @property
    def any(self) -> bool:
        return any(
            v is not None
            for v in (self.max_latency_s, self.min_quality, self.max_cost_usd)
        )

    def describe(self) -> str:
        parts = []
        if self.max_latency_s is not None:
            parts.append(f"latency<={self.max_latency_s:g}s")
        if self.min_quality is not None:
            parts.append(f"quality>={self.min_quality:.2f}")
        if self.max_cost_usd is not None:
            parts.append(f"cost<=${self.max_cost_usd:g}")
        return ", ".join(parts)


def build_model_map(
    catalog: ModelCatalog, trained_models: list[str], available: list[str]
) -> dict[str, str]:
    """Benchmark model name -> the catalog model that stands in for it.

    Matches on `benchmark_alias` first, then on an exact id match. Only models
    the provider pool can actually serve are included: routing to a model whose
    provider is disabled would produce a confident 503.
    """
    servable = set(available)
    mapping: dict[str, str] = {}

    for model_id, spec in catalog.models.items():
        if model_id not in servable:
            continue
        alias = getattr(spec, "benchmark_alias", "") or ""
        if alias and alias in trained_models:
            mapping[alias] = model_id
        elif model_id in trained_models:
            mapping[model_id] = model_id

    return mapping


class LiveRouter(RoutingStrategy):
    """Chooses a catalog model for a real request, or says why it cannot."""

    name = "learned"

    def __init__(
        self,
        predictor,
        metadata: RouterMetadata,
        model_map: dict[str, str],
        costs: dict[str, float],
        latencies: dict[str, float] | None = None,
        default_min_quality: float = 0.5,
        min_spread: float = 0.0,
    ) -> None:
        self.predictor = predictor
        self.metadata = metadata
        self.model_map = model_map
        self.costs = costs
        self.latencies = latencies or {}
        self.default_min_quality = default_min_quality
        # Below this spread the router says it has no opinion. See choose().
        self.min_spread = min_spread

    @property
    def enabled(self) -> bool:
        """Routing needs at least two models to choose between."""
        return len(self.model_map) >= 2

    @property
    def routable_models(self) -> list[str]:
        return sorted(self.model_map.values())

    def choose(
        self, context: RoutingContext, limits: RequestLimits | None = None
    ) -> RoutingDecision:
        limits = limits or RequestLimits()
        probabilities = self.predictor.predict_one(context.prompt_text)

        # THE HONESTY CHECK. If every model gets roughly the same score, the
        # router has not distinguished them - it has no opinion, and acting on
        # one would be inventing a decision.
        #
        # This is the failure C.4 found and could not see: on prompts unlike
        # its training data the router returned 0.67-0.87 for everything, sent
        # it all to the cheapest model, and wrote a reason implying judgement.
        # Saying "I do not know" makes that visible in every ledger row.
        spread = self._spread(probabilities)
        if spread < self.min_spread:
            cheapest = min(
                self.model_map.values(),
                key=lambda m: self.costs.get(m, float("inf")),
            )
            return RoutingDecision(
                model=cheapest,
                strategy=self.name,
                abstained=True,
                reason=(
                    f"predictions span only {spread:.3f} across "
                    f"{len(probabilities)} models - no usable discrimination "
                    "on this prompt, so no routing decision was made"
                ),
                features={"probabilities": probabilities, "spread": spread},
            )

        # Candidates are catalog models, cheapest first.
        candidates = [
            (catalog_id, probabilities.get(benchmark_name, 0.0))
            for benchmark_name, catalog_id in self.model_map.items()
        ]
        candidates.sort(key=lambda pair: self.costs.get(pair[0], float("inf")))

        eligible = [
            (model, p)
            for model, p in candidates
            if self._within_limits(model, limits)
        ]

        if not eligible:
            # Nothing satisfies the caller's limits. Serve it rather than
            # failing, and say so - a silent breach is worse than a late answer.
            fallback = candidates[0][0]
            return RoutingDecision(
                model=fallback,
                strategy=self.name,
                reason=(
                    f"no model satisfies {limits.describe()}; "
                    f"served by {fallback}"
                ),
            )

        floor = (
            limits.min_quality
            if limits.min_quality is not None
            else self.default_min_quality
        )
        for model, probability in eligible:
            if probability >= floor:
                return RoutingDecision(
                    model=model,
                    strategy=self.name,
                    reason=(
                        f"cheapest of {len(eligible)} eligible clearing "
                        f"p>={floor:.2f} (predicted {probability:.2f})"
                    ),
                )

        # Nothing cleared the bar: take the best chance available rather than
        # the cheapest, which would abandon exactly the hard requests.
        model, probability = max(eligible, key=lambda pair: pair[1])
        return RoutingDecision(
            model=model,
            strategy=self.name,
            reason=(
                f"no model reached p>={floor:.2f}; best of "
                f"{len(eligible)} eligible was {model} at {probability:.2f}"
            ),
        )

    @staticmethod
    def _spread(probabilities: dict[str, float]) -> float:
        """How far apart the best and worst predictions are.

        Deliberately the raw range rather than a variance or an entropy. The
        question being asked is "did this router separate the models at all",
        and the gap between the top and bottom answers it directly - which
        also makes the number in the ledger reason readable by a human.
        """
        if len(probabilities) < 2:
            return 0.0
        values = list(probabilities.values())
        return max(values) - min(values)

    def _within_limits(self, model: str, limits: RequestLimits) -> bool:
        if (
            limits.max_cost_usd is not None
            and self.costs.get(model, float("inf")) > limits.max_cost_usd
        ):
            return False
        if limits.max_latency_s is not None:
            typical = self.latencies.get(model)
            # No latency history is not evidence of being fast enough.
            if typical is None or typical > limits.max_latency_s:
                return False
        return True


def build_router(
    catalog: ModelCatalog, available: list[str], path: str | None = None
) -> LiveRouter | None:
    """Load the configured router, or return None with the reason logged.

    Never raises. A stale or missing artifact must degrade the service to
    "serves everything with the default model", not stop it from starting.
    """
    from switchboard.config import settings
    from switchboard.routing.artifact import ArtifactError, load

    # Best available wins, in this order:
    #
    #   1. the operator's own router, trained on their traffic and their models
    #   2. a broad router shipped inside the package, trained across 40 public
    #      benchmark suites - which is what makes a FRESH INSTALL route on its
    #      first request instead of waiting for a month of feedback
    #
    # A missing artifact at either level is normal, not an error. Whatever is
    # left - eventually the ladder policy - serves the request.
    attempts = [(path or settings.router_path, "your trained router")]
    if not path:
        attempts.append((str(shipped_router_path()), "the shipped router"))

    predictor = metadata = None
    for candidate, describe in attempts:
        if not candidate or not Path(candidate).exists():
            continue
        try:
            predictor, metadata = load(candidate)
            logger.info("Loaded %s from %s", describe, candidate)
            break
        except ArtifactError as exc:
            logger.warning("Could not load %s: %s", describe, exc)

    if predictor is None or metadata is None:
        logger.info(
            "No usable router; `auto` falls back to the ladder policy."
        )
        return None

    model_map = build_model_map(catalog, metadata.models, available)
    if len(model_map) < 2:
        logger.warning(
            "Routing disabled: the router knows %d model(s) this catalog can "
            "serve, and needs at least 2. Trained for: %s. Add "
            "`benchmark_alias` entries in providers.yaml to map them.",
            len(model_map),
            ", ".join(metadata.models[:6]) or "(none)",
        )

    costs, latencies = {}, {}
    for catalog_id in model_map.values():
        spec = catalog.models[catalog_id]
        # A representative request, so models are ranked on comparable terms.
        costs[catalog_id] = spec.cost(1000, 500)
        if spec.typical_latency_s is not None:
            latencies[catalog_id] = spec.typical_latency_s

    router = LiveRouter(
        predictor=predictor,
        metadata=metadata,
        model_map=model_map,
        costs=costs,
        latencies=latencies,
        default_min_quality=settings.router_min_quality,
        min_spread=settings.router_min_spread,
    )
    if router.enabled:
        logger.info(
            "Routing enabled over %s (%s)",
            ", ".join(router.routable_models),
            metadata.describe(),
        )
    return router
