"""Holds one client per enabled provider and answers "who serves this model?".

Clients are built once at startup rather than per request. An HTTP client owns
a connection pool; creating one per request would discard those connections and
pay a fresh TCP and TLS handshake every time - which on a remote provider costs
more than the inference for short prompts.
"""

from __future__ import annotations

import logging

from switchboard.catalog import ModelCatalog, ProviderSpec
from switchboard.providers.base import (
    Provider,
    ProviderNotConfigured,
    ProviderUnavailable,
)
from switchboard.providers.openai_compatible import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

#: provider type -> adapter class. Adding a provider type means adding a line.
ADAPTERS: dict[str, type[Provider]] = {
    "openai-compatible": OpenAICompatibleProvider,
}


class LocalOnlyViolation(RuntimeError):
    """A remote provider is enabled while local-only mode is on."""


class ProviderPool:
    def __init__(self, catalog: ModelCatalog, local_only: bool = False) -> None:
        self._catalog = catalog
        self._providers: dict[str, Provider] = {}
        self._unconfigured: dict[str, str] = {}

        for spec in catalog.enabled_providers():
            if local_only and not spec.is_local:
                raise LocalOnlyViolation(
                    f"Local-only mode is on, but provider {spec.id!r} points at "
                    f"{spec.base_url}, which is not local.\n"
                    f"  Either disable that provider in providers.yaml, or set "
                    f"SWITCHBOARD_LOCAL_ONLY=false."
                )
            self._build(spec)

        if not self._providers:
            logger.warning(
                "No providers are usable. Enable one in providers.yaml and make "
                "sure its API key environment variable is set."
            )

    def _build(self, spec: ProviderSpec) -> None:
        adapter = ADAPTERS.get(spec.type)
        if adapter is None:  # pragma: no cover - catalog validation blocks this
            self._unconfigured[spec.id] = f"unknown provider type {spec.type!r}"
            return

        try:
            self._providers[spec.id] = adapter(spec)  # type: ignore[call-arg]
        except ProviderNotConfigured as exc:
            # A missing key disables one provider; it must not stop the server.
            # Recorded so `switchboard providers` can explain the gap.
            logger.warning("Provider %r unavailable: %s", spec.id, exc)
            self._unconfigured[spec.id] = str(exc)

    # --- Lookups -----------------------------------------------------------

    def for_model(self, model: str) -> Provider:
        spec = self._catalog.provider_for(model)
        if spec is None:
            raise ProviderUnavailable(
                f"No provider declares model {model!r}. Add it to "
                "providers.yaml, or call a model from: "
                f"{', '.join(self.available_models()) or '(none available)'}"
            )

        provider = self._providers.get(spec.id)
        if provider is None:
            reason = self._unconfigured.get(spec.id, "provider is disabled")
            raise ProviderUnavailable(
                f"Model {model!r} is served by {spec.id!r}, which is not "
                f"usable: {reason}"
            )
        return provider

    def get(self, provider_id: str) -> Provider | None:
        return self._providers.get(provider_id)

    def available_models(self) -> list[str]:
        """Models whose provider is actually usable right now."""
        return sorted(
            model_id
            for model_id, spec in self._catalog.models.items()
            if spec.provider_id in self._providers
        )

    def unconfigured(self) -> dict[str, str]:
        return dict(self._unconfigured)

    @property
    def provider_ids(self) -> list[str]:
        return sorted(self._providers)

    async def health(self) -> dict[str, bool]:
        return {pid: await p.is_healthy() for pid, p in self._providers.items()}

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
        self._providers.clear()
