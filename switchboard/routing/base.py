"""The routing interface.

Every strategy - the trivial baselines here, and the learned router in
milestone 4 - implements `choose`. Keeping the interface this narrow is what
makes the comparison fair: the evaluation harness swaps one object and changes
nothing else about how a request is served.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoutingContext:
    """Everything a strategy is allowed to look at when deciding."""

    messages: list[dict]
    requested_model: str | None = None

    @property
    def prompt_text(self) -> str:
        """User-authored text only, flattened.

        System messages are excluded deliberately: they are usually boilerplate
        injected by the application, identical across requests, and would swamp
        any signal about how hard the actual question is.
        """
        parts = [
            message["content"]
            for message in self.messages
            if isinstance(message, dict)
            and message.get("role") != "system"
            and isinstance(message.get("content"), str)
        ]
        return "\n".join(parts)

    @property
    def last_user_message(self) -> str:
        for message in reversed(self.messages):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
        return ""


@dataclass(frozen=True)
class RoutingDecision:
    """Which model, and why.

    `reason` is not decoration. When a routing result looks wrong, the only way
    to debug it is to know what the strategy thought it was doing.
    """

    model: str
    strategy: str
    reason: str = ""
    features: dict = field(default_factory=dict)

    #: True when the strategy had no usable opinion and the caller should fall
    #: back to something simpler.
    #
    # This is the "I don't know" the project has been missing. A router that
    # always returns a confident answer is not confident - it is silent about
    # its own ignorance, which is exactly how the C.4 failure went unnoticed:
    # predictions clustered in a narrow band, everything went to the cheapest
    # model, and nothing in the logs said why.
    #
    # `model` is still populated when this is set, so a caller that ignores the
    # flag gets a sane answer rather than a crash.
    abstained: bool = False


class RoutingStrategy(ABC):
    """Chooses a model for a request."""

    #: Stable identifier used in configs, CLI flags, and result files.
    name: str = "unnamed"

    @abstractmethod
    def choose(self, context: RoutingContext) -> RoutingDecision:
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
