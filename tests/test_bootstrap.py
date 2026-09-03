"""`switchboard init` — the file it writes.

The wizard's questions live in the CLI; everything tested here is the pure
function underneath, so the hard part — producing a catalog that actually
parses, prices correctly, and orders the ladder cheapest-first — can be checked
without simulating somebody typing.

The rule these tests exist to protect: **a model with no price is never written
as active.** Discovery does not invent prices and neither does this. A guessed
price flows straight into budget enforcement and every savings figure, and is
wrong in a way nobody can see from the outside.
"""

from __future__ import annotations

import pytest
import yaml

from switchboard.bootstrap import (
    KNOWN_PROVIDERS,
    PROVIDERS_BY_ID,
    ChosenModel,
    ChosenProvider,
    build_ladder,
    from_discovered,
    render_catalog,
    simulated_pricing,
    summarise,
)
from switchboard.discovery import DiscoveredModel


def local(*models: ChosenModel) -> ChosenProvider:
    return ChosenProvider("ollama-local", list(models))


def priced(model_id: str, out_price: float, **kwargs) -> ChosenModel:
    return ChosenModel(
        id=model_id,
        provider_id=kwargs.pop("provider_id", "ollama-local"),
        input_per_mtok=out_price / 4,
        output_per_mtok=out_price,
        **kwargs,
    )


# --- The catalog it writes --------------------------------------------------


def test_the_written_catalog_parses() -> None:
    text = render_catalog([local(priced("a", 0.4), priced("b", 15.0))])
    assert yaml.safe_load(text)


def test_the_written_catalog_actually_loads(tmp_path) -> None:
    """Parsing as YAML is not enough — the catalog validates strictly on load,
    and a wizard that writes a file Switchboard then rejects is worse than no
    wizard."""
    from switchboard.catalog import ModelCatalog

    path = tmp_path / "providers.yaml"
    path.write_text(
        render_catalog([local(priced("small", 0.4), priced("big", 15.0))]),
        encoding="utf-8",
    )
    catalog = ModelCatalog.load(path)
    assert catalog.ladder == ("small", "big")
    assert catalog.baseline_model == "big"


def test_model_ids_with_colons_and_slashes_survive() -> None:
    """`qwen2.5:7b` and `anthropic/claude-sonnet-4` both contain characters
    YAML reads as structure. Unquoted they load as something else entirely."""
    text = render_catalog(
        [local(priced("qwen2.5:7b", 15.0), priced("anthropic/claude-sonnet-4", 20.0))]
    )
    ids = [m["id"] for p in yaml.safe_load(text)["providers"] for m in p["models"]]
    assert "qwen2.5:7b" in ids
    assert "anthropic/claude-sonnet-4" in ids


def test_every_known_provider_is_written_even_when_unused() -> None:
    """Adding one later should be uncommenting, not remembering a URL and an
    adapter type."""
    doc = yaml.safe_load(render_catalog([local(priced("a", 1.0))]))
    written = {p["id"] for p in doc["providers"]}
    assert written == {spec["id"] for spec in KNOWN_PROVIDERS}


def test_unchosen_providers_are_written_disabled() -> None:
    doc = yaml.safe_load(render_catalog([local(priced("a", 1.0))]))
    enabled = {p["id"] for p in doc["providers"] if p["enabled"]}
    assert enabled == {"ollama-local"}


def test_api_keys_are_never_written_only_variable_names() -> None:
    """The file is meant to be committable. A key in it is public forever."""
    text = render_catalog([ChosenProvider("openai", [priced("gpt-5", 10.0)])])
    assert "OPENAI_API_KEY" in text
    for prefix in ("sk-", "sk-ant-", "sk-or-", "AIza", "gsk_"):
        assert prefix not in text


# --- The unpriced rule ------------------------------------------------------


def test_an_unpriced_model_is_commented_out_not_guessed() -> None:
    """THE rule. Written rather than dropped, because dropping it leaves
    somebody wondering where their model went."""
    text = render_catalog(
        [ChosenProvider("openai", [ChosenModel("gpt-5", "openai")])]
    )
    assert "NO PRICE SUPPLIED" in text
    assert "will not guess" in text

    doc = yaml.safe_load(text)
    openai = next(p for p in doc["providers"] if p["id"] == "openai")
    assert openai["models"] == [] or openai["models"] is None


def test_a_catalog_with_unpriced_models_still_loads(tmp_path) -> None:
    """Commented out means the file remains valid. Written live with a made-up
    price, it would load and quietly bill wrongly forever."""
    from switchboard.catalog import ModelCatalog

    path = tmp_path / "providers.yaml"
    path.write_text(
        render_catalog(
            [
                local(priced("small", 0.4), priced("big", 15.0)),
                ChosenProvider("openai", [ChosenModel("gpt-5", "openai")]),
            ]
        ),
        encoding="utf-8",
    )
    assert "gpt-5" not in ModelCatalog.load(path).known_models()


def test_unpriced_models_stay_off_the_ladder() -> None:
    """The ladder is what the router walks and what escalation climbs. A model
    in the wrong place on it sends requests to the wrong model, silently."""
    ladder = build_ladder(
        [
            local(priced("cheap", 0.4), priced("dear", 15.0)),
            ChosenProvider("openai", [ChosenModel("gpt-5", "openai")]),
        ]
    )
    assert ladder == ["cheap", "dear"]


# --- The ladder -------------------------------------------------------------


def test_the_ladder_is_cheapest_first() -> None:
    assert build_ladder(
        [local(priced("mid", 5.0), priced("dear", 15.0), priced("cheap", 0.4))]
    ) == ["cheap", "mid", "dear"]


def test_the_ladder_weights_output_price() -> None:
    """A typical request sends far fewer tokens than it receives, and output is
    priced several times higher almost everywhere."""
    cheap_out = ChosenModel("a", "p", input_per_mtok=10.0, output_per_mtok=1.0)
    dear_out = ChosenModel("b", "p", input_per_mtok=0.1, output_per_mtok=9.0)
    assert build_ladder([ChosenProvider("ollama-local", [dear_out, cheap_out])]) == [
        "a",
        "b",
    ]


def test_the_baseline_is_the_dearest_model() -> None:
    """Savings are measured against what a 'give everyone the best' setup would
    have cost, so the baseline must be the top of the ladder."""
    doc = yaml.safe_load(
        render_catalog([local(priced("cheap", 0.4), priced("dear", 15.0))])
    )
    assert doc["baseline_model"] == "dear"


def test_no_priced_models_does_not_produce_a_broken_file() -> None:
    text = render_catalog(
        [ChosenProvider("openai", [ChosenModel("gpt-5", "openai")])]
    )
    doc = yaml.safe_load(text)
    assert doc["ladder"] in ([], None)
    assert doc["baseline_model"] == ""


# --- Simulated pricing for local models -------------------------------------


def test_local_models_are_spread_across_the_tiers() -> None:
    """Local inference is free. Priced at zero, every budget and savings figure
    would be meaningless — so each model wears the price tag of the commercial
    model it stands in for."""
    tiers = [simulated_pricing(i, 4)[2] for i in range(4)]
    assert tiers == ["T0", "T1", "T2", "T3"]


def test_two_local_models_land_at_the_extremes() -> None:
    """The cheap/expensive gap is the thing a router exists to exploit. Two
    models bunched in the middle would leave nothing to route between."""
    assert [simulated_pricing(i, 2)[2] for i in range(2)] == ["T0", "T3"]


def test_a_single_local_model_gets_a_sensible_tier() -> None:
    assert simulated_pricing(0, 1)[2] == "T1"


def test_simulated_pricing_is_declared_in_the_file() -> None:
    text = render_catalog([local(priced("a", 0.4), priced("b", 15.0))])
    assert "simulated_pricing: true" in text
    assert "No real money" in text


# --- Carrying discovery across ----------------------------------------------


def test_prices_from_discovery_are_kept() -> None:
    discovered = [
        DiscoveredModel(
            id="anthropic/claude-sonnet-4",
            display_name="Claude Sonnet 4",
            context_window=200000,
            input_per_mtok=3.0,
            output_per_mtok=15.0,
        )
    ]
    chosen = from_discovered(discovered, "openrouter")
    assert chosen[0].priced
    assert chosen[0].output_per_mtok == pytest.approx(15.0)
    assert chosen[0].context_window == 200000


def test_a_model_discovery_could_not_price_stays_unpriced() -> None:
    chosen = from_discovered([DiscoveredModel(id="gpt-5")], "openai")
    assert not chosen[0].priced


# --- Small things -----------------------------------------------------------


def test_the_summary_flags_missing_prices() -> None:
    text = summarise(
        [
            local(priced("a", 1.0)),
            ChosenProvider("openai", [ChosenModel("gpt-5", "openai")]),
        ]
    )
    assert "need prices" in text


def test_the_summary_is_clean_when_everything_is_priced() -> None:
    assert "need prices" not in summarise([local(priced("a", 1.0))])


def test_nothing_configured_says_so() -> None:
    assert summarise([]) == "No models configured."


def test_every_known_provider_has_an_adapter() -> None:
    """A provider offered by the wizard that nothing can talk to would fail on
    the user's first request, after they had already pasted a key."""
    from switchboard.providers import ADAPTERS

    for spec in KNOWN_PROVIDERS:
        assert spec["type"] in ADAPTERS, spec["id"]


def test_known_providers_are_indexed_consistently() -> None:
    assert set(PROVIDERS_BY_ID) == {spec["id"] for spec in KNOWN_PROVIDERS}
