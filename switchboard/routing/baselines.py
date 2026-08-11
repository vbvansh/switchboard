"""Baseline strategies.

These exist to make later results mean something. A learned router that beats
nothing is not a result; a learned router that beats "always cheap", "always
expensive", "pick at random" and "guess from keywords" is.

`Random` in particular is the honest floor. Any strategy that cannot beat a coin
flip on the cost/accuracy trade-off has learned nothing at all.
"""

from __future__ import annotations

import random
import re

from switchboard.routing.base import RoutingContext, RoutingDecision, RoutingStrategy


class AlwaysModel(RoutingStrategy):
    """Send everything to one model.

    Two instances matter: always-cheapest is the cost floor, always-most-
    expensive is the quality ceiling and the spend a company pays today. Every
    other strategy is judged on where it sits between them.
    """

    def __init__(self, model: str, name: str | None = None) -> None:
        self.model = model
        self.name = name or f"always:{model}"

    def choose(self, context: RoutingContext) -> RoutingDecision:
        return RoutingDecision(
            model=self.model, strategy=self.name, reason="fixed model"
        )


class RandomModel(RoutingStrategy):
    """Pick uniformly at random from the ladder.

    Seeded so a run can be reproduced exactly - an unseeded baseline would move
    between runs and make comparisons worthless.
    """

    name = "random"

    def __init__(self, ladder: list[str], seed: int = 0) -> None:
        self.ladder = list(ladder)
        self._random = random.Random(seed)

    def choose(self, context: RoutingContext) -> RoutingDecision:
        model = self._random.choice(self.ladder)
        return RoutingDecision(
            model=model, strategy=self.name, reason="uniform random choice"
        )


# Words that tend to appear in requests needing actual reasoning. This list is
# guesswork by design - it is the naive approach, included to show what naive
# costs.
HARD_HINTS = frozenset(
    {
        "prove", "derive", "explain", "why", "analyse", "analyze", "compare",
        "design", "optimise", "optimize", "debug", "refactor", "algorithm",
        "complexity", "architecture", "trade-off", "tradeoff", "step-by-step",
        "reason", "calculate", "solve",
    }
)

EASY_HINTS = frozenset(
    {
        "format", "capitalise", "capitalize", "lowercase", "uppercase", "spell",
        "list", "translate", "rename", "summarise", "summarize", "define",
    }
)

WORD_PATTERN = re.compile(r"[a-z][a-z\-]*")


class KeywordHeuristic(RoutingStrategy):
    """Route on prompt length and a hand-written keyword list.

    This is the approach most people reach for first, and it is included
    precisely so the project can show what it is worth. It cannot understand
    that "why is the sky blue" is easy while "why does this deadlock" is hard -
    both contain "why".
    """

    name = "keyword"

    def __init__(
        self,
        ladder: list[str],
        long_prompt_chars: int = 400,
        short_prompt_chars: int = 80,
    ) -> None:
        if not ladder:
            raise ValueError("KeywordHeuristic needs a non-empty ladder.")
        self.ladder = list(ladder)
        self.long_prompt_chars = long_prompt_chars
        self.short_prompt_chars = short_prompt_chars

    def choose(self, context: RoutingContext) -> RoutingDecision:
        text = context.prompt_text
        words = set(WORD_PATTERN.findall(text.lower()))

        hard_hits = len(words & HARD_HINTS)
        easy_hits = len(words & EASY_HINTS)
        length = len(text)

        score = 0
        if hard_hits:
            score += 1
        if hard_hits >= 3:
            score += 1
        if length > self.long_prompt_chars:
            score += 1
        if easy_hits and not hard_hits:
            score -= 1
        if length < self.short_prompt_chars:
            score -= 1

        index = max(0, min(len(self.ladder) - 1, score + 1))
        return RoutingDecision(
            model=self.ladder[index],
            strategy=self.name,
            reason=(
                f"score={score} (hard={hard_hits}, easy={easy_hits}, "
                f"chars={length}) -> tier {index}"
            ),
            features={
                "hard_hits": hard_hits,
                "easy_hits": easy_hits,
                "chars": length,
                "score": score,
            },
        )
