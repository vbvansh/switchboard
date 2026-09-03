"""Getting from `pip install` to a working router without editing YAML by hand.

WHAT THIS REPLACES. Until now, setting Switchboard up meant cloning the
repository, opening providers.yaml, reading its comments, working out which
provider entries applied to you, finding each model's price page, and typing the
numbers in. That is a fine workflow for the person who wrote the file and a poor
one for everybody else.

`switchboard init` asks instead, checks each key actually works before writing
anything, and produces a catalog that loads.

WHY THE LOGIC LIVES HERE AND THE QUESTIONS LIVE IN THE CLI. Everything in this
module is a pure function over data: given some providers and models, produce
the YAML. Nothing here reads stdin, so the hard part - building a file that
parses, prices correctly, and orders the ladder cheapest-first - is testable
without simulating somebody typing.

THE ONE RULE IT WILL NOT BREAK. A model with no price is never written as
active. Discovery does not invent prices, and neither does this: a model whose
price nobody supplied is written commented out, with the price lines marked, so
the catalog still loads and the gap is visible. A guessed price would flow
straight into budget enforcement and every savings figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from switchboard.discovery import DiscoveredModel

#: Providers `init` knows how to set up. Anything else can still be added by
#: editing providers.yaml - this list is a convenience, not a limit.
#:
#: `type` selects the adapter: most of the industry copied OpenAI's request
#: format, and the two that did not get their own translating adapter.
KNOWN_PROVIDERS: tuple[dict, ...] = (
    {
        "id": "ollama-local",
        "label": "Ollama (local, free, no key)",
        "type": "openai-compatible",
        "base_url": "${OLLAMA_BASE_URL:-http://localhost:11434/v1}",
        "api_key_env": None,
        "simulated_pricing": True,
        "note": "runs on your machine; nothing leaves the host",
    },
    {
        "id": "openrouter",
        "label": "OpenRouter (one key, hundreds of models, publishes prices)",
        "type": "openai-compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "note": "the only provider whose API gives real prices - start here",
    },
    {
        "id": "openai",
        "label": "OpenAI",
        "type": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "note": "publishes no prices; you will be asked for them",
    },
    {
        "id": "anthropic",
        "label": "Anthropic (Claude)",
        "type": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "note": "native adapter; tool calls go via OpenRouter instead",
    },
    {
        "id": "gemini",
        "label": "Google Gemini",
        "type": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key_env": "GEMINI_API_KEY",
        "note": "native adapter; tool calls go via OpenRouter instead",
    },
    {
        "id": "groq",
        "label": "Groq (fast, has a free tier)",
        "type": "openai-compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "note": "cheapest way to verify a real remote provider end to end",
    },
)

PROVIDERS_BY_ID = {entry["id"]: entry for entry in KNOWN_PROVIDERS}

#: Used when a local model has no meaningful price of its own. Local inference
#: is free, and pricing it at zero would make every budget and savings figure
#: meaningless - so a local model wears the price tag of the commercial model
#: it stands in for, and every screen that shows money says "simulated".
SIMULATED_TIERS = (
    (0.10, 0.40, "T0", "budget tier"),
    (0.25, 1.25, "T1", "small general-purpose tier"),
    (1.00, 5.00, "T2", "mid reasoning tier"),
    (3.00, 15.00, "T3", "frontier tier"),
)


@dataclass
class ChosenModel:
    """One model the operator picked, with whatever price is known."""

    id: str
    provider_id: str
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    context_window: int | None = None
    tier: str = "T2"
    stands_in_for: str = ""
    benchmark_alias: str = ""

    @property
    def priced(self) -> bool:
        return self.input_per_mtok is not None and self.output_per_mtok is not None

    @property
    def blended_price(self) -> float:
        """One number for ordering the ladder cheapest-first.

        Weighted towards output because that is where the money goes: a typical
        request sends far fewer tokens than it receives, and output is priced
        several times higher almost everywhere.
        """
        if not self.priced:
            return float("inf")
        return self.input_per_mtok + 3.0 * self.output_per_mtok


@dataclass
class ChosenProvider:
    id: str
    models: list[ChosenModel] = field(default_factory=list)

    @property
    def spec(self) -> dict:
        return PROVIDERS_BY_ID[self.id]


def simulated_pricing(index: int, total: int) -> tuple[float, float, str, str]:
    """Spread local models across the simulated tiers, smallest first.

    With four local models they land on T0..T3; with two they land at the ends,
    so the cheap/expensive gap the router exists to exploit is still there.
    """
    if total <= 1:
        return SIMULATED_TIERS[1]
    position = index / (total - 1)
    slot = round(position * (len(SIMULATED_TIERS) - 1))
    return SIMULATED_TIERS[slot]


def from_discovered(
    models: list[DiscoveredModel], provider_id: str
) -> list[ChosenModel]:
    """Carry across whatever discovery managed to learn."""
    return [
        ChosenModel(
            id=model.id,
            provider_id=provider_id,
            input_per_mtok=model.input_per_mtok,
            output_per_mtok=model.output_per_mtok,
            context_window=model.context_window,
            stands_in_for=model.display_name,
        )
        for model in models
    ]


def build_ladder(providers: list[ChosenProvider]) -> list[str]:
    """Every priced model, cheapest first.

    Unpriced models are excluded rather than guessed at. The ladder is what the
    router walks and what escalation climbs, so a model in the wrong place on
    it would send requests to the wrong model - silently, and forever.
    """
    priced = [
        model
        for provider in providers
        for model in provider.models
        if model.priced
    ]
    return [m.id for m in sorted(priced, key=lambda m: m.blended_price)]


def _quote(value: str) -> str:
    """Model ids contain colons (`qwen2.5:7b`) and slashes
    (`anthropic/claude-sonnet-4`), both of which YAML reads as structure."""
    return '"' + str(value).replace('"', '\\"') + '"'


def _render_model(model: ChosenModel, indent: str = "      ") -> str:
    lines = [f"{indent}- id: {_quote(model.id)}", f"{indent}  tier: {model.tier}"]
    if model.stands_in_for:
        lines.append(f"{indent}  stands_in_for: {_quote(model.stands_in_for)}")
    if model.context_window:
        lines.append(f"{indent}  context_window: {model.context_window}")
    if model.benchmark_alias:
        lines.append(f"{indent}  benchmark_alias: {_quote(model.benchmark_alias)}")
    lines.append(f"{indent}  input_per_mtok: {model.input_per_mtok:g}")
    lines.append(f"{indent}  output_per_mtok: {model.output_per_mtok:g}")
    return "\n".join(lines)


def _render_unpriced(model: ChosenModel, indent: str = "      ") -> str:
    """Commented out, so the catalog still loads and the gap is obvious.

    Written rather than dropped because the alternative is worse both ways:
    inventing a price corrupts every budget silently, and dropping the model
    leaves somebody wondering where it went.
    """
    body = [
        f"{indent}# NO PRICE SUPPLIED - fill these in and uncomment to enable.",
        f"{indent}# Switchboard will not guess: a made-up price flows straight",
        f"{indent}# into budget enforcement and every savings figure.",
        f"{indent}# - id: {_quote(model.id)}",
        f"{indent}#   tier: {model.tier}",
    ]
    if model.context_window:
        body.append(f"{indent}#   context_window: {model.context_window}")
    body.append(f"{indent}#   input_per_mtok: 0.00")
    body.append(f"{indent}#   output_per_mtok: 0.00")
    return "\n".join(body)


def render_catalog(providers: list[ChosenProvider]) -> str:
    """The whole providers.yaml, ready to write.

    Every provider Switchboard knows about is written out, enabled or not, with
    its comments intact - so adding one later is uncommenting rather than
    remembering the URL and the adapter type.
    """
    chosen = {provider.id: provider for provider in providers}
    ladder = build_ladder(providers)
    # Built outside the f-string below: Python 3.11 forbids backslashes inside
    # an f-string expression, and `_quote` produces them.
    baseline = _quote(ladder[-1]) if ladder else '""'

    out: list[str] = [
        "# Switchboard provider registry",
        "# " + "=" * 75,
        "# Written by `switchboard init`. Edit it freely - this is a normal",
        "# config file and nothing regenerates it behind your back.",
        "#",
        "# API KEYS DO NOT GO IN THIS FILE. `api_key_env` names an environment",
        "# variable to read the key from, so this file is safe to commit.",
        "# " + "=" * 75,
        "",
        "# The model a 'give everyone the best' setup would use. Every request",
        "# records what it WOULD have cost here - that difference is the",
        "# savings figure on the dashboard.",
        f"baseline_model: {baseline}",
        "",
        "# Models the router may choose between, CHEAPEST FIRST. The order is",
        "# validated on load, and escalation climbs it one rung at a time.",
        "ladder:",
    ]
    out.extend(f"  - {_quote(model)}" for model in ladder)
    if not ladder:
        out.append("  []  # no priced models yet - see the entries below")

    out += [
        "",
        "# Fallback for a model not declared below. Never zero: an unpriced",
        "# model costing nothing would understate spend and overstate savings.",
        "default_pricing:",
        "  input_per_mtok: 1.0",
        "  output_per_mtok: 5.0",
        "",
        "providers:",
    ]

    for spec in KNOWN_PROVIDERS:
        provider = chosen.get(spec["id"])
        enabled = provider is not None and bool(provider.models)

        out.append("")
        out.append(f"  # {spec['label']} - {spec['note']}")
        out.append(f"  - id: {spec['id']}")
        out.append(f"    type: {spec['type']}")
        out.append(f"    base_url: {spec['base_url']}")
        if spec["api_key_env"]:
            out.append(f"    api_key_env: {spec['api_key_env']}")
        out.append(f"    enabled: {str(enabled).lower()}")
        if spec.get("simulated_pricing"):
            out.append("    # Local inference is free. Each model wears the price")
            out.append("    # tag of the commercial model it stands in for, so")
            out.append("    # budgets and savings mean something. No real money.")
            out.append("    simulated_pricing: true")
            out.append("    timeout_seconds: 600  # a cold load can be slow")

        if not enabled:
            out.append("    models: []")
            continue

        out.append("    models:")
        for model in provider.models:
            out.append(
                _render_model(model) if model.priced else _render_unpriced(model)
            )
            out.append("")

    return "\n".join(out).rstrip() + "\n"


def summarise(providers: list[ChosenProvider]) -> str:
    """One line for the end of the wizard."""
    models = [m for p in providers for m in p.models]
    priced = [m for m in models if m.priced]
    if not models:
        return "No models configured."
    if len(priced) == len(models):
        return f"{len(models)} models across {len(providers)} providers."
    return (
        f"{len(models)} models across {len(providers)} providers; "
        f"{len(models) - len(priced)} need prices before they can be used."
    )
