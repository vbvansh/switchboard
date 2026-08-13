"""Settings, and local-only mode.

Local-only mode replaced the hard localhost lock that existed while this was a
personal experiment. The constraint became a feature: an operator can make
Switchboard physically incapable of sending prompts off the host, which is what
some organisations require before letting a gateway near their data.
"""

from __future__ import annotations

import pytest

from switchboard.catalog import ModelCatalog
from switchboard.config import Settings
from switchboard.providers import LocalOnlyViolation, ProviderPool

MODELS = [{"id": "m", "tier": "T0", "input_per_mtok": 1, "output_per_mtok": 2}]


def catalog_with(base_url: str, enabled: bool = True) -> ModelCatalog:
    return ModelCatalog.from_dict(
        {
            "baseline_model": "m",
            "ladder": ["m"],
            "providers": [
                {
                    "id": "p",
                    "type": "openai-compatible",
                    "base_url": base_url,
                    "enabled": enabled,
                    "models": MODELS,
                }
            ],
        }
    )


# --- Defaults --------------------------------------------------------------


def test_local_only_is_off_by_default() -> None:
    """Talking to providers is the point of the product."""
    assert Settings().local_only is False


def test_defaults_are_safe_to_ship() -> None:
    settings = Settings()
    assert settings.store_prompts is False
    assert settings.host == "127.0.0.1"  # not 0.0.0.0 - do not expose by accident


def test_providers_file_defaults_to_the_shipped_catalog() -> None:
    assert Settings().providers_file.endswith("providers.yaml")


# --- Local-only enforcement ------------------------------------------------


def test_local_only_rejects_a_remote_provider() -> None:
    pool_args = catalog_with("https://api.openai.com/v1")
    with pytest.raises(LocalOnlyViolation, match="not local"):
        ProviderPool(pool_args, local_only=True)


def test_local_only_allows_a_local_provider() -> None:
    pool = ProviderPool(catalog_with("http://localhost:11434/v1"), local_only=True)
    assert pool.provider_ids == ["p"]


def test_local_only_ignores_disabled_remote_providers() -> None:
    """A declared-but-unused provider must not block startup."""
    catalog = catalog_with("https://api.openai.com/v1", enabled=False)
    pool = ProviderPool(catalog, local_only=True)
    assert pool.provider_ids == []


def test_remote_providers_work_when_local_only_is_off() -> None:
    pool = ProviderPool(catalog_with("https://api.openai.com/v1"), local_only=False)
    assert pool.provider_ids == ["p"]


def test_the_violation_explains_both_ways_out() -> None:
    """An error that does not say how to fix it wastes the operator's time."""
    with pytest.raises(LocalOnlyViolation) as excinfo:
        ProviderPool(catalog_with("https://api.openai.com/v1"), local_only=True)

    message = str(excinfo.value)
    assert "providers.yaml" in message
    assert "SWITCHBOARD_LOCAL_ONLY" in message
