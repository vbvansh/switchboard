"""The provider catalog.

This file is the single source of truth for routing, pricing, failover and the
dashboard, so it is validated strictly on load. Wrong numbers that look
plausible are worse than a crash - these tests exist to force the crash.
"""

from __future__ import annotations

import pytest

from switchboard.catalog import CatalogError, ModelCatalog

MINIMAL = {
    "baseline_model": "big",
    "ladder": ["small", "big"],
    "providers": [
        {
            "id": "local",
            "type": "openai-compatible",
            "base_url": "http://localhost:11434/v1",
            "enabled": True,
            "models": [
                {
                    "id": "small",
                    "tier": "T0",
                    "input_per_mtok": 1,
                    "output_per_mtok": 2,
                },
                {
                    "id": "big",
                    "tier": "T1",
                    "input_per_mtok": 10,
                    "output_per_mtok": 20,
                },
            ],
        }
    ],
}


def catalog(**overrides) -> ModelCatalog:
    return ModelCatalog.from_dict({**MINIMAL, **overrides})


# --- The shipped file ------------------------------------------------------


def test_shipped_catalog_loads(prices: ModelCatalog) -> None:
    """providers.yaml in the repo must always be valid."""
    assert prices.models
    assert prices.baseline_model in prices.models


def test_shipped_ladder_is_cheapest_first(prices: ModelCatalog) -> None:
    costs = [prices.cost(m, 1000, 1000) for m in prices.ladder]
    assert costs == sorted(costs)


def test_shipped_catalog_prices_every_local_model(prices: ModelCatalog) -> None:
    for model in ("qwen2.5:1.5b", "qwen2.5:3b", "qwen3:4b", "qwen2.5:7b"):
        assert model in prices.known_models()


# --- Validation ------------------------------------------------------------


def test_mis_ordered_ladder_is_rejected() -> None:
    """Strategies assume index 0 is cheapest; a wrong order corrupts routing.

    Sorting silently would hide the mistake. Failing points at the line to fix.
    """
    with pytest.raises(CatalogError, match="cheapest to most expensive"):
        catalog(ladder=["big", "small"])


def test_ladder_entry_must_exist() -> None:
    with pytest.raises(CatalogError, match="not declared by any provider"):
        catalog(ladder=["small", "imaginary"])


def test_baseline_model_must_exist() -> None:
    with pytest.raises(CatalogError, match="baseline_model"):
        catalog(baseline_model="imaginary")


def test_baseline_model_is_required() -> None:
    raw = {k: v for k, v in MINIMAL.items() if k != "baseline_model"}
    with pytest.raises(CatalogError, match="baseline_model is required"):
        ModelCatalog.from_dict(raw)


def test_unknown_provider_type_is_rejected() -> None:
    """Better to fail at load than to fail on the first request."""
    with pytest.raises(CatalogError, match="unknown type"):
        ModelCatalog.from_dict(
            {
                **MINIMAL,
                "providers": [
                    {
                        "id": "x",
                        "type": "carrier-pigeon",
                        "base_url": "http://localhost",
                        "models": [],
                    }
                ],
            }
        )


def test_duplicate_provider_ids_are_rejected() -> None:
    duplicate = dict(MINIMAL["providers"][0])
    with pytest.raises(CatalogError, match="duplicate provider"):
        ModelCatalog.from_dict(
            {**MINIMAL, "providers": [MINIMAL["providers"][0], duplicate]}
        )


def test_a_model_may_be_served_by_several_providers() -> None:
    """This used to be an error. It is now the failover feature.

    When two providers offer the same model, the first declaration wins for
    pricing and routing and the rest become backups, tried in file order when
    the primary is down.
    """
    second = {
        "id": "other",
        "type": "openai-compatible",
        "base_url": "http://localhost:9999/v1",
        "enabled": True,
        "models": [
            {"id": "small", "tier": "T0", "input_per_mtok": 99, "output_per_mtok": 99}
        ],
    }
    catalog = ModelCatalog.from_dict(
        {**MINIMAL, "providers": [MINIMAL["providers"][0], second]}
    )

    assert [p.id for p in catalog.providers_for("small")] == ["local", "other"]
    # The first declaration is the one priced and routed on.
    assert catalog.models["small"].provider_id == "local"
    assert catalog.cost("small", 1_000_000, 0) == pytest.approx(1.0)


def test_missing_required_provider_field_is_rejected() -> None:
    with pytest.raises(CatalogError, match="missing 'base_url'"):
        ModelCatalog.from_dict(
            {**MINIMAL, "providers": [{"id": "x", "type": "openai-compatible"}]}
        )


def test_a_missing_file_says_what_to_do() -> None:
    with pytest.raises(CatalogError, match="SWITCHBOARD_PROVIDERS_FILE"):
        ModelCatalog.load("does-not-exist.yaml")


# --- Pricing ---------------------------------------------------------------


def test_cost_is_per_million_tokens() -> None:
    cat = catalog()
    assert cat.cost("big", 1_000_000, 0) == pytest.approx(10.0)
    assert cat.cost("big", 0, 1_000_000) == pytest.approx(20.0)


def test_baseline_cost_uses_the_baseline_model() -> None:
    cat = catalog()
    assert cat.baseline_cost(1_000_000, 0) == cat.cost("big", 1_000_000, 0)


def test_unknown_models_are_never_free() -> None:
    """Silently pricing an unknown model at zero would overstate savings."""
    assert catalog().cost("never-configured", 1_000_000, 0) > 0


def test_zero_tokens_cost_nothing() -> None:
    assert catalog().cost("big", 0, 0) == 0.0


# --- Keys and secrets ------------------------------------------------------


def test_keys_come_from_the_environment_not_the_file(monkeypatch) -> None:
    """Config files get committed. Keys in config files leak."""
    cat = ModelCatalog.from_dict(
        {
            **MINIMAL,
            "providers": [
                {**MINIMAL["providers"][0], "api_key_env": "TEST_PROVIDER_KEY"}
            ],
        }
    )
    provider = cat.providers["local"]

    monkeypatch.delenv("TEST_PROVIDER_KEY", raising=False)
    assert provider.api_key() is None
    assert provider.key_is_available is False

    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-value")
    assert provider.api_key() == "secret-value"
    assert provider.key_is_available is True


def test_a_provider_needing_no_key_is_always_available() -> None:
    assert catalog().providers["local"].key_is_available is True


# --- Locality --------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:8000/v1", True),
        ("https://api.openai.com/v1", False),
        ("https://openrouter.ai/api/v1", False),
        ("http://192.168.1.50:11434/v1", False),
    ],
)
def test_locality_detection(url: str, local: bool) -> None:
    cat = ModelCatalog.from_dict(
        {**MINIMAL, "providers": [{**MINIMAL["providers"][0], "base_url": url}]}
    )
    assert cat.providers["local"].is_local is local


# --- Availability ----------------------------------------------------------


def test_disabled_providers_are_not_routable() -> None:
    cat = ModelCatalog.from_dict(
        {**MINIMAL, "providers": [{**MINIMAL["providers"][0], "enabled": False}]}
    )
    assert cat.routable_models() == []
    assert cat.enabled_providers() == []


def test_routable_models_need_their_key(monkeypatch) -> None:
    cat = ModelCatalog.from_dict(
        {
            **MINIMAL,
            "providers": [
                {**MINIMAL["providers"][0], "api_key_env": "TEST_PROVIDER_KEY"}
            ],
        }
    )
    monkeypatch.delenv("TEST_PROVIDER_KEY", raising=False)
    assert cat.routable_models() == []

    monkeypatch.setenv("TEST_PROVIDER_KEY", "x")
    assert cat.routable_models() == ["small", "big"]


def test_simulated_pricing_is_surfaced() -> None:
    """A simulated saving must never be mistaken for a real one."""
    assert catalog().has_simulated_pricing is False
    marked = ModelCatalog.from_dict(
        {
            **MINIMAL,
            "providers": [{**MINIMAL["providers"][0], "simulated_pricing": True}],
        }
    )
    assert marked.has_simulated_pricing is True


def test_the_shipped_catalog_is_marked_simulated(prices: ModelCatalog) -> None:
    """Local Ollama pricing is invented, and the product must say so."""
    assert prices.has_simulated_pricing is True
