"""The model catalog: everything Switchboard knows about providers and models.

Loaded from providers.yaml. This is the single source of truth that routing,
cost accounting, failover and the dashboard all read from - which is why it is
validated strictly on load. A silent mistake here would produce wrong numbers
everywhere downstream, and wrong numbers that look plausible are worse than a
crash.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

TOKENS_PER_PRICE_UNIT = 1_000_000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "providers.yaml"

#: Adapter types this build knows how to talk to.
KNOWN_PROVIDER_TYPES = frozenset({"openai-compatible"})

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"})


#: ${VAR} or ${VAR:-fallback} inside any string value in providers.yaml.
ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class CatalogError(ValueError):
    """providers.yaml is malformed or inconsistent."""


def expand_env(value: str) -> str:
    """Substitute ${VAR} and ${VAR:-fallback} from the environment.

    Exists so one catalog file works in every environment. A base_url of
    `http://localhost:11434/v1` is correct on a laptop and wrong inside a
    container, where the host is reached as host.docker.internal. Rather than
    shipping two files that drift apart, the file states the default and the
    deployment overrides it.

    An unset variable with no fallback expands to empty, which then fails
    validation loudly - better than silently pointing at a wrong address.
    """
    return ENV_PLACEHOLDER.sub(
        lambda m: os.environ.get(m.group(1), m.group(2) or ""), value
    )


def _expand_tree(node):
    """Apply env expansion to every string in a nested structure."""
    if isinstance(node, str):
        return expand_env(node)
    if isinstance(node, dict):
        return {key: _expand_tree(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_expand_tree(item) for item in node]
    return node


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider_id: str
    tier: str
    input_per_mtok: float
    output_per_mtok: float
    context_window: int = 8192
    stands_in_for: str = ""
    emits_thinking: bool = False
    simulated_pricing: bool = False

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_mtok
            + completion_tokens * self.output_per_mtok
        ) / TOKENS_PER_PRICE_UNIT


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    type: str
    base_url: str
    enabled: bool
    api_key_env: str | None = None
    timeout_seconds: float = 120.0
    simulated_pricing: bool = False
    model_ids: tuple[str, ...] = ()

    @property
    def is_local(self) -> bool:
        from urllib.parse import urlparse

        host = urlparse(self.base_url).hostname
        return host is not None and host.lower() in LOCAL_HOSTS

    @property
    def requires_key(self) -> bool:
        return self.api_key_env is not None

    def api_key(self) -> str | None:
        """Read the key from the environment. Never stored in the catalog.

        Returned as a value rather than cached on the object so a rotated key
        takes effect without a restart, and so a key is never accidentally
        serialised into a log line or an error message.
        """
        if self.api_key_env is None:
            return None
        return os.environ.get(self.api_key_env)

    @property
    def key_is_available(self) -> bool:
        return not self.requires_key or bool(self.api_key())


@dataclass(frozen=True)
class ModelCatalog:
    """Every model Switchboard can reach, and what each one costs."""

    providers: dict[str, ProviderSpec]
    models: dict[str, ModelSpec]
    baseline_model: str
    ladder: tuple[str, ...]
    default_pricing: ModelSpec
    _warned: set[str] = field(default_factory=set, repr=False, compare=False)

    # --- Loading -----------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CATALOG_PATH) -> ModelCatalog:
        path = Path(path)
        if not path.exists():
            raise CatalogError(
                f"No provider catalog at {path}. Copy providers.yaml from the "
                "repository root, or set SWITCHBOARD_PROVIDERS_FILE."
            )

        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise CatalogError(f"{path} is not valid YAML: {exc}") from exc

        return cls.from_dict(raw, source=str(path))

    @classmethod
    def from_dict(cls, raw: dict, source: str = "<dict>") -> ModelCatalog:
        raw = _expand_tree(raw)
        providers: dict[str, ProviderSpec] = {}
        models: dict[str, ModelSpec] = {}

        for entry in raw.get("providers") or []:
            provider, provider_models = cls._parse_provider(entry, source)
            if provider.id in providers:
                raise CatalogError(
                    f"{source}: duplicate provider id {provider.id!r}"
                )
            providers[provider.id] = provider

            for model in provider_models:
                if model.id in models:
                    raise CatalogError(
                        f"{source}: model {model.id!r} is declared by both "
                        f"{models[model.id].provider_id!r} and "
                        f"{model.provider_id!r}. Routing could not choose "
                        "between them."
                    )
                models[model.id] = model

        default = cls._parse_default_pricing(raw.get("default_pricing") or {})
        baseline = raw.get("baseline_model")
        if not baseline:
            raise CatalogError(f"{source}: baseline_model is required.")
        if baseline not in models:
            raise CatalogError(
                f"{source}: baseline_model {baseline!r} is not declared by any "
                "provider."
            )

        ladder = tuple(raw.get("ladder") or ())
        for model_id in ladder:
            if model_id not in models:
                raise CatalogError(
                    f"{source}: ladder entry {model_id!r} is not declared by "
                    "any provider."
                )

        catalog = cls(
            providers=providers,
            models=models,
            baseline_model=baseline,
            ladder=ladder,
            default_pricing=default,
        )
        catalog._validate_ladder_order(source)
        return catalog

    @staticmethod
    def _parse_provider(
        entry: dict, source: str
    ) -> tuple[ProviderSpec, list[ModelSpec]]:
        for required in ("id", "type", "base_url"):
            if not entry.get(required):
                raise CatalogError(
                    f"{source}: provider entry missing {required!r}: {entry}"
                )

        if entry["type"] not in KNOWN_PROVIDER_TYPES:
            raise CatalogError(
                f"{source}: provider {entry['id']!r} has unknown type "
                f"{entry['type']!r}. Supported: "
                f"{', '.join(sorted(KNOWN_PROVIDER_TYPES))}."
            )

        simulated = bool(entry.get("simulated_pricing", False))
        provider_models = [
            ModelSpec(
                id=model["id"],
                provider_id=entry["id"],
                tier=model.get("tier", "unknown"),
                input_per_mtok=float(model["input_per_mtok"]),
                output_per_mtok=float(model["output_per_mtok"]),
                context_window=int(model.get("context_window", 8192)),
                stands_in_for=model.get("stands_in_for", ""),
                emits_thinking=bool(model.get("emits_thinking", False)),
                simulated_pricing=simulated,
            )
            for model in entry.get("models") or []
        ]

        provider = ProviderSpec(
            id=entry["id"],
            type=entry["type"],
            base_url=str(entry["base_url"]).rstrip("/"),
            enabled=bool(entry.get("enabled", False)),
            api_key_env=entry.get("api_key_env"),
            timeout_seconds=float(entry.get("timeout_seconds", 120.0)),
            simulated_pricing=simulated,
            model_ids=tuple(m.id for m in provider_models),
        )
        return provider, provider_models

    @staticmethod
    def _parse_default_pricing(raw: dict) -> ModelSpec:
        return ModelSpec(
            id="<default>",
            provider_id="<none>",
            tier="unknown",
            input_per_mtok=float(raw.get("input_per_mtok", 1.0)),
            output_per_mtok=float(raw.get("output_per_mtok", 5.0)),
            stands_in_for="unrecognised model, priced at mid tier",
        )

    def _validate_ladder_order(self, source: str) -> None:
        """A mis-ordered ladder silently corrupts every routing decision.

        Strategies assume index 0 is cheapest. Rather than sorting quietly and
        hiding the mistake, fail loudly and point at the line to fix.
        """
        costs = [self.models[m].cost(1000, 1000) for m in self.ladder]
        if costs != sorted(costs):
            ordered = sorted(self.ladder, key=lambda m: self.models[m].cost(1000, 1000))
            raise CatalogError(
                f"{source}: `ladder` must run cheapest to most expensive.\n"
                f"  got:      {list(self.ladder)}\n"
                f"  expected: {ordered}"
            )

    # --- Lookups -----------------------------------------------------------

    def for_model(self, model: str) -> ModelSpec:
        """Spec for a model, falling back to default pricing.

        An unknown model must never be free - that would understate spend and
        overstate savings. It falls back to mid-tier pricing and warns once.
        """
        if (spec := self.models.get(model)) is not None:
            return spec

        if model not in self._warned:
            logger.warning(
                "Model %r is not in the catalog; using default pricing. "
                "Add it to providers.yaml.",
                model,
            )
            self._warned.add(model)
        return self.default_pricing

    def provider_for(self, model: str) -> ProviderSpec | None:
        spec = self.models.get(model)
        return self.providers.get(spec.provider_id) if spec else None

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        return self.for_model(model).cost(prompt_tokens, completion_tokens)

    def baseline_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return self.cost(self.baseline_model, prompt_tokens, completion_tokens)

    def known_models(self) -> list[str]:
        return sorted(self.models)

    def enabled_providers(self) -> list[ProviderSpec]:
        return [p for p in self.providers.values() if p.enabled]

    def routable_models(self) -> list[str]:
        """Ladder models whose provider is enabled and has its key available."""
        routable = []
        for model_id in self.ladder:
            provider = self.provider_for(model_id)
            if provider and provider.enabled and provider.key_is_available:
                routable.append(model_id)
        return routable

    @property
    def cheapest(self) -> str:
        if not self.ladder:
            raise CatalogError("The ladder is empty; nothing can be routed.")
        return self.ladder[0]

    @property
    def most_expensive(self) -> str:
        if not self.ladder:
            raise CatalogError("The ladder is empty; nothing can be routed.")
        return self.ladder[-1]

    @property
    def has_simulated_pricing(self) -> bool:
        """True if any enabled provider's prices are made up.

        Surfaced in the CLI and dashboard so a simulated saving is never
        mistaken for a real one.
        """
        return any(p.simulated_pricing for p in self.enabled_providers())
