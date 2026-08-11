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
    "You are a concise assistant. Work through the problem briefly, then end "
    "your reply with your final answer on its own line, in exactly this "
    f"format:\n{ANSWER_MARKER} <answer>\n"
    "Give only the value after the marker - no units, no explanation, no "
    "punctuation. Write nothing after that line."
)

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


def extract_answer(text: str) -> tuple[str, bool]:
    """Pull the marked answer out of a reply.

    Returns (answer, followed_format). `followed_format` is recorded rather than
    thrown away: small models frequently ignore output instructions, and how
    often they do is itself a finding worth reporting.
    """
    cleaned = strip_thinking(text)

    marker_at = cleaned.upper().rfind(ANSWER_MARKER)
    if marker_at != -1:
        answer = cleaned[marker_at + len(ANSWER_MARKER) :]
        return answer.splitlines()[0].strip() if answer.strip() else "", True

    # No marker: fall back to the whole reply so a correct-but-unmarked answer
    # still has a chance. Flagged so these are distinguishable in the results.
    return cleaned.strip(), False


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
