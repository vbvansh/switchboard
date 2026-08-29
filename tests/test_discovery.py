"""Asking a provider what models it has.

The behaviour these tests exist to lock down is a refusal: **discovery never
invents a price.** A guessed price would flow straight into budget enforcement
and savings reports and be wrong in a way nobody could see from the outside.
So an unpriced model comes back as unknown, and the YAML it produces will not
load until a human fills the number in.

All payloads here are recorded response shapes. No key, no network.
"""

from __future__ import annotations

import pytest
import yaml

from switchboard import discovery

# --- OpenAI-shaped lists ----------------------------------------------------

OPENAI_BODY = {
    "object": "list",
    "data": [
        {"id": "gpt-4.1", "object": "model", "owned_by": "openai"},
        {"id": "gpt-5", "object": "model", "owned_by": "openai"},
    ],
}


def test_an_openai_model_list_is_read() -> None:
    models = discovery.parse(
        "openai-compatible", "https://api.openai.com/v1", OPENAI_BODY
    )
    assert [m.id for m in models] == ["gpt-4.1", "gpt-5"]


def test_openai_publishes_no_prices_and_we_do_not_invent_them() -> None:
    """THE test. An unpriced model must stay unpriced."""
    models = discovery.parse(
        "openai-compatible", "https://api.openai.com/v1", OPENAI_BODY
    )
    assert all(not m.priced for m in models)
    assert all(m.input_per_mtok is None for m in models)


def test_a_body_that_is_not_a_model_list_is_an_error() -> None:
    with pytest.raises(discovery.DiscoveryError):
        discovery.parse("openai-compatible", "https://x/v1", {"oops": True})


def test_a_non_json_body_is_an_error() -> None:
    with pytest.raises(discovery.DiscoveryError):
        discovery.parse("openai-compatible", "https://x/v1", b"<html>502</html>")


# --- OpenRouter, the one that ships prices ----------------------------------

OPENROUTER_BODY = {
    "data": [
        {
            "id": "anthropic/claude-sonnet-4",
            "name": "Anthropic: Claude Sonnet 4",
            "context_length": 200000,
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
        {
            "id": "meta-llama/llama-3.1-8b-instruct:free",
            "name": "Llama 3.1 8B (free)",
            "context_length": 131072,
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ]
}


def test_openrouter_is_detected_by_its_url_not_its_type() -> None:
    """It IS openai-compatible. It just happens to publish more, so the parser
    is chosen by address rather than by declared type."""
    models = discovery.parse(
        "openai-compatible", "https://openrouter.ai/api/v1", OPENROUTER_BODY
    )
    assert models[0].priced


def test_per_token_prices_become_per_million() -> None:
    """OpenRouter quotes "0.000003" per token. Switchboard works in dollars per
    million tokens, so a factor of a million separates them - and getting it
    wrong would understate every bill by 1,000,000x."""
    models = discovery.parse(
        "openai-compatible", "https://openrouter.ai/api/v1", OPENROUTER_BODY
    )
    claude = next(m for m in models if "claude" in m.id)
    assert claude.input_per_mtok == pytest.approx(3.0)
    assert claude.output_per_mtok == pytest.approx(15.0)
    assert claude.context_window == 200000


def test_a_genuinely_free_model_is_zero_not_unknown() -> None:
    """"0" means free. That is a real price, and treating it as missing would
    make a free model look unconfigured."""
    models = discovery.parse(
        "openai-compatible", "https://openrouter.ai/api/v1", OPENROUTER_BODY
    )
    free = next(m for m in models if "free" in m.id)
    assert free.priced
    assert free.input_per_mtok == 0.0


def test_an_unparseable_price_is_unknown_not_zero() -> None:
    models = discovery.parse(
        "openai-compatible",
        "https://openrouter.ai/api/v1",
        {"data": [{"id": "x", "pricing": {"prompt": "n/a", "completion": "n/a"}}]},
    )
    assert not models[0].priced


# --- Anthropic and Gemini ---------------------------------------------------


def test_anthropic_model_list_is_read() -> None:
    models = discovery.parse(
        "anthropic",
        "https://api.anthropic.com/v1",
        {
            "data": [
                {"id": "claude-sonnet-4-20250514", "display_name": "Claude Sonnet 4"}
            ]
        },
    )
    assert models[0].id == "claude-sonnet-4-20250514"
    assert models[0].display_name == "Claude Sonnet 4"
    assert not models[0].priced


def test_gemini_names_are_stripped_of_their_prefix() -> None:
    """Google returns "models/gemini-2.5-flash". You do not put that in a
    request, so shipping it into the catalog would make every call fail."""
    models = discovery.parse(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta",
        {
            "models": [
                {
                    "name": "models/gemini-2.5-flash",
                    "displayName": "Gemini 2.5 Flash",
                    "inputTokenLimit": 1048576,
                    "supportedGenerationMethods": ["generateContent"],
                }
            ]
        },
    )
    assert models[0].id == "gemini-2.5-flash"
    assert models[0].context_window == 1048576


def test_models_that_cannot_generate_text_are_dropped() -> None:
    """An embedding model listed as callable produces a confusing failure the
    first time something routes to it."""
    models = discovery.parse(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta",
        {
            "models": [
                {
                    "name": "models/text-embedding-004",
                    "supportedGenerationMethods": ["embedContent"],
                },
                {
                    "name": "models/gemini-2.5-flash",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ]
        },
    )
    assert [m.id for m in models] == ["gemini-2.5-flash"]


def test_an_unknown_provider_type_says_so() -> None:
    with pytest.raises(discovery.DiscoveryError):
        discovery.parse("bedrock", "https://bedrock.aws", {"data": []})


# --- The YAML it prints -----------------------------------------------------


def test_priced_models_produce_loadable_yaml() -> None:
    block = discovery.to_yaml(
        discovery.parse(
            "openai-compatible", "https://openrouter.ai/api/v1", OPENROUTER_BODY
        )
    )
    parsed = yaml.safe_load(block)
    assert isinstance(parsed, list)
    entry = next(e for e in parsed if "claude" in e["id"])
    assert entry["input_per_mtok"] == 3.0


def test_model_ids_with_colons_and_slashes_survive_yaml() -> None:
    """`qwen2.5:7b` and `anthropic/claude-sonnet-4` both contain characters
    YAML reads as structure. Unquoted, they load as something else entirely."""
    block = discovery.to_yaml([discovery.DiscoveredModel(id="qwen2.5:7b")])
    assert yaml.safe_load(block)[0]["id"] == "qwen2.5:7b"


def test_unpriced_models_are_marked_rather_than_filled_in() -> None:
    """The YAML must be obviously incomplete. A confident wrong price is far
    worse than a blank that asks to be filled."""
    block = discovery.to_yaml([discovery.DiscoveredModel(id="gpt-5")])
    assert "PRICE UNKNOWN" in block
    assert "REPLACE ME" in block


def test_an_empty_result_still_produces_valid_yaml() -> None:
    assert yaml.safe_load(discovery.to_yaml([])) == []


def test_the_summary_states_how_many_prices_are_missing() -> None:
    models = [
        discovery.DiscoveredModel(id="a", input_per_mtok=1.0, output_per_mtok=2.0),
        discovery.DiscoveredModel(id="b"),
    ]
    text = discovery.summarise(models)
    assert "1 with published prices" in text
    assert "1 needing prices" in text


def test_all_priced_reads_cleanly() -> None:
    models = [
        discovery.DiscoveredModel(id="a", input_per_mtok=1.0, output_per_mtok=2.0)
    ]
    assert "all with published prices" in discovery.summarise(models)
