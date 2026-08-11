"""Objective grading. No LLM judge involved.

A judge model would need to be more capable than the models being judged, and
this project has no such model available. So every task carries a
mechanically-checkable answer instead: a number, an exact string, or required
substrings. Slower to author, but the resulting accuracy figures are facts
rather than opinions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Models are asked to end with this so answers can be extracted uniformly.
#: Applied identically to every model and strategy, so it cannot favour one.
ANSWER_MARKER = "ANSWER:"

SYSTEM_PROMPT = (
    "You are a concise assistant. Think briefly if you need to, then finish "
    "your reply with a final line in exactly this format:\n"
    f"{ANSWER_MARKER} <value>\n"
    "The value must be the bare answer - no units, no explanation, no "
    "punctuation. Always include that line, even when it is your whole reply. "
    "Write nothing after it."
)

# How the answer was recovered from the reply. Kept as three states rather than
# a pass/fail flag because they carry very different risk:
#
#   marker - the model followed instructions; extraction is exact.
#   bare   - no marker, but the whole reply was a short value. Harmless.
#   prose  - no marker and a long reply, so the answer was guessed out of the
#            text. These are the rows where a grading mistake is plausible, and
#            the only ones worth flagging in a report.
FORMAT_MARKER = "marker"
FORMAT_BARE = "bare"
FORMAT_PROSE = "prose"

#: A reply no longer than this, with no marker, counts as a bare answer.
BARE_ANSWER_MAX_CHARS = 60
BARE_ANSWER_MAX_LINES = 2

# Reasoning models (qwen3) wrap internal monologue in these. It must be removed
# before grading or the marker search finds text from the model's scratchpad.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)

NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")
PUNCTUATION = re.compile(r"[^\w\s]")


def strip_thinking(text: str) -> str:
    """Remove <think> blocks, including one left unclosed by truncation."""
    text = THINK_BLOCK.sub(" ", text)
    return UNCLOSED_THINK.sub(" ", text).strip()


def extract_answer(text: str) -> tuple[str, str]:
    """Pull the answer out of a reply.

    Returns (answer, format_kind) where format_kind is one of FORMAT_MARKER,
    FORMAT_BARE or FORMAT_PROSE - see the constants above for why this is three
    states and not a pass/fail flag.
    """
    cleaned = strip_thinking(text)

    marker_at = cleaned.upper().rfind(ANSWER_MARKER)
    if marker_at != -1:
        answer = cleaned[marker_at + len(ANSWER_MARKER) :]
        first_line = answer.splitlines()[0].strip() if answer.strip() else ""
        return first_line, FORMAT_MARKER

    # No marker. Fall back to the whole reply so a correct-but-unmarked answer
    # still counts - but distinguish a terse value from prose we had to mine.
    stripped = cleaned.strip()
    lines = [line for line in stripped.splitlines() if line.strip()]
    is_bare = (
        len(stripped) <= BARE_ANSWER_MAX_CHARS and len(lines) <= BARE_ANSWER_MAX_LINES
    )
    return stripped, FORMAT_BARE if is_bare else FORMAT_PROSE


def _to_float(value: str) -> float | None:
    try:
        return float(value.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _normalise(value: str) -> str:
    return PUNCTUATION.sub("", value).lower().strip()


@dataclass(frozen=True)
class Check:
    """How a task's answer is verified."""

    type: str
    value: str | float | None = None
    values: tuple[str, ...] = ()
    tolerance: float = 1e-6

    @classmethod
    def from_dict(cls, raw: dict) -> Check:
        return cls(
            type=raw["type"],
            value=raw.get("value"),
            values=tuple(raw.get("values", ())),
            tolerance=float(raw.get("tolerance", 1e-6)),
        )

    def grade(self, answer: str) -> bool:
        if self.type == "numeric":
            return self._grade_numeric(answer)
        if self.type == "exact":
            return _normalise(answer) == _normalise(str(self.value))
        if self.type == "contains":
            haystack = _normalise(answer)
            return all(_normalise(v) in haystack for v in self.values)
        raise ValueError(f"Unknown check type {self.type!r}")

    def _grade_numeric(self, answer: str) -> bool:
        expected = float(self.value)  # type: ignore[arg-type]

        if (direct := _to_float(answer)) is not None:
            return abs(direct - expected) <= self.tolerance

        # The model wrote prose around the number. Take the last number in the
        # text: conclusions come at the end, and intermediate working appears
        # before it.
        numbers = NUMBER_PATTERN.findall(answer)
        if not numbers:
            return False
        last = _to_float(numbers[-1])
        return last is not None and abs(last - expected) <= self.tolerance
