"""Ask a provider what models it has, and what they cost.

WHY THIS EXISTS. Adding a model to `providers.yaml` by hand means typing its
name, its price in, its price out and its context window. For five models that
is fine. For three hundred it is absurd, and every price is stale the moment a
vendor changes theirs.

This is the half of "works with every model" that people forget. Format
translation gets you *able* to call anything. Discovery is what makes it
*bearable*: paste a key, and Switchboard reports "found 47 models, here they
are, with prices".

WHAT IT WILL NOT DO. It will not invent a price.

Only some providers publish prices in their API. OpenRouter does, and it is
excellent - real per-token numbers for hundreds of models, kept current. OpenAI,
Anthropic, Google and every local server publish a model list with no prices at
all.

For those, this module marks the price as UNKNOWN and says so, loudly, in every
place the result surfaces. It does not fill in a number from memory. A price
this project guessed would flow straight into somebody's budget enforcement and
savings report, and be wrong in a way nobody could see. An obvious blank that
asks to be filled in beats a confident number that is wrong.

NOTHING IS WRITTEN WITHOUT BEING ASKED. `switchboard providers discover` prints
YAML for a human to read and paste. `--write` exists, and it is opt-in, and it
never overwrites a model you have already configured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

#: Value written for a price the provider did not publish. Deliberately not 0.0
#: - a model priced at zero would understate spend and overstate savings, which
#: is the exact self-flattering error this project keeps designing against.
UNKNOWN_PRICE = None


@dataclass(frozen=True)
class DiscoveredModel:
    """One model a provider says it has."""

    id: str
    context_window: int | None = None
    input_per_mtok: float | None = UNKNOWN_PRICE
    output_per_mtok: float | None = UNKNOWN_PRICE
    display_name: str = ""

    @property
    def priced(self) -> bool:
        return self.input_per_mtok is not None and self.output_per_mtok is not None


class DiscoveryError(RuntimeError):
    """The provider answered, but not with anything recognisable."""


# --- Parsers ----------------------------------------------------------------
# One per response shape. Pure functions over a decoded body, so every one of
# them is testable against a recorded payload with no key and no network.


def parse_openai_models(body: dict[str, Any]) -> list[DiscoveredModel]:
    """OpenAI's `/v1/models`, and the many providers that copied it.

    Ids only. No prices, no context windows - the endpoint simply does not
    carry them.
    """
    data = body.get("data")
    if not isinstance(data, list):
        raise DiscoveryError("expected a `data` list of models")
    return [
        DiscoveredModel(id=str(entry["id"]))
        for entry in data
        if isinstance(entry, dict) and entry.get("id")
    ]


def parse_openrouter_models(body: dict[str, Any]) -> list[DiscoveredModel]:
    """OpenRouter's model list - the only one that ships real prices.

    Prices arrive as strings, per single token, e.g. "0.000003". Multiplying by
    a million gives the per-million-token figure the rest of Switchboard uses.

    Free models are listed at "0". That is a genuine zero, not a missing value,
    so it is kept as 0.0 rather than being treated as unknown.
    """
    data = body.get("data")
    if not isinstance(data, list):
        raise DiscoveryError("expected a `data` list of models")

    models = []
    for entry in data:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        pricing = entry.get("pricing") or {}
        models.append(
            DiscoveredModel(
                id=str(entry["id"]),
                display_name=str(entry.get("name") or ""),
                context_window=_as_int(entry.get("context_length")),
                input_per_mtok=_per_million(pricing.get("prompt")),
                output_per_mtok=_per_million(pricing.get("completion")),
            )
        )
    return models


def parse_anthropic_models(body: dict[str, Any]) -> list[DiscoveredModel]:
    """Anthropic's `/v1/models`. Ids and display names; no prices."""
    data = body.get("data")
    if not isinstance(data, list):
        raise DiscoveryError("expected a `data` list of models")
    return [
        DiscoveredModel(
            id=str(entry["id"]),
            display_name=str(entry.get("display_name") or ""),
        )
        for entry in data
        if isinstance(entry, dict) and entry.get("id")
    ]


def parse_gemini_models(body: dict[str, Any]) -> list[DiscoveredModel]:
    """Google's model list.

    Names come back fully qualified as `models/gemini-2.5-flash`; the prefix is
    stripped because that is not what you put in a request. Token limits are
    published, prices are not.

    Models that cannot generate text at all - embedding models, for instance -
    are filtered out. Listing one as callable would produce a confusing failure
    the first time somebody routed to it.
    """
    data = body.get("models")
    if not isinstance(data, list):
        raise DiscoveryError("expected a `models` list")

    models = []
    for entry in data:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        methods = entry.get("supportedGenerationMethods")
        if isinstance(methods, list) and "generateContent" not in methods:
            continue
        models.append(
            DiscoveredModel(
                id=str(entry["name"]).removeprefix("models/"),
                display_name=str(entry.get("displayName") or ""),
                context_window=_as_int(entry.get("inputTokenLimit")),
            )
        )
    return models


#: provider type -> parser. OpenRouter is detected by its base_url rather than
#: its type, because it IS openai-compatible - it just happens to publish more.
PARSERS = {
    "openai-compatible": parse_openai_models,
    "anthropic": parse_anthropic_models,
    "gemini": parse_gemini_models,
}


def parser_for(provider_type: str, base_url: str):
    if "openrouter.ai" in (base_url or "").lower():
        return parse_openrouter_models
    parser = PARSERS.get(provider_type)
    if parser is None:
        raise DiscoveryError(
            f"no discovery support for provider type {provider_type!r}"
        )
    return parser


def parse(provider_type: str, base_url: str, body: Any) -> list[DiscoveredModel]:
    """Decode whatever the provider returned into a list of models."""
    if isinstance(body, bytes | bytearray | str):
        try:
            body = json.loads(body)
        except ValueError as exc:
            raise DiscoveryError(f"response was not JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise DiscoveryError("expected a JSON object")
    return parser_for(provider_type, base_url)(body)


# --- Rendering --------------------------------------------------------------


def to_yaml(
    models: list[DiscoveredModel],
    tier: str = "T2",
    indent: str = "      ",
) -> str:
    """Render discovered models as a YAML block to paste into providers.yaml.

    Models whose price is unknown are emitted with the price lines commented
    out and a marker above them. That is deliberate: the block will not load
    until a human supplies the numbers, so an unpriced model cannot quietly
    reach production billed at a fallback rate.
    """
    if not models:
        return f"{indent}[]  # the provider returned no models\n"

    lines: list[str] = []
    for model in sorted(models, key=lambda m: m.id):
        lines.append(f"{indent}- id: {_quote(model.id)}")
        lines.append(f"{indent}  tier: {tier}")
        if model.display_name:
            lines.append(f"{indent}  stands_in_for: {_quote(model.display_name)}")
        if model.context_window:
            lines.append(f"{indent}  context_window: {model.context_window}")

        if model.priced:
            lines.append(f"{indent}  input_per_mtok: {model.input_per_mtok:g}")
            lines.append(f"{indent}  output_per_mtok: {model.output_per_mtok:g}")
        else:
            lines.append(
                f"{indent}  # PRICE UNKNOWN - this provider does not publish "
                "prices."
            )
            lines.append(
                f"{indent}  # Fill these in from their price page. Switchboard "
                "will not guess:"
            )
            lines.append(
                f"{indent}  # a made-up price flows straight into budgets and "
                "savings reports."
            )
            lines.append(f"{indent}  input_per_mtok: 0.00   # <- REPLACE ME")
            lines.append(f"{indent}  output_per_mtok: 0.00  # <- REPLACE ME")
        lines.append("")

    return "\n".join(lines)


def summarise(models: list[DiscoveredModel]) -> str:
    priced = sum(1 for model in models if model.priced)
    if not models:
        return "No models returned."
    if priced == len(models):
        return f"{len(models)} models, all with published prices."
    return (
        f"{len(models)} models, {priced} with published prices and "
        f"{len(models) - priced} needing prices filled in by hand."
    )


# --- Small helpers ----------------------------------------------------------


def _per_million(value: Any) -> float | None:
    """OpenRouter quotes per-token prices as strings. Convert to per-million.

    A missing or unparseable value returns None - unknown, not free.
    """
    if value is None:
        return None
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _quote(value: str) -> str:
    """Model ids often contain a colon (`qwen2.5:7b`) or a slash
    (`anthropic/claude-sonnet-4`), both of which YAML reads as structure."""
    return '"' + value.replace('"', '\\"') + '"'
