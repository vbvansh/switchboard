"""What every provider adapter must do.

Keeping this interface small is deliberate. Adding support for a new provider
should mean writing one class with three methods, not understanding the rest of
Switchboard.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx


class ProviderError(RuntimeError):
    """Base class for provider failures the API turns into responses."""


class ProviderUnavailable(ProviderError):
    """Could not reach the provider at all - down, refused, or unresolvable."""


class ProviderNotConfigured(ProviderError):
    """The provider is declared but unusable, typically a missing API key."""


class Provider(ABC):
    """One configured upstream that can serve chat completions."""

    #: Matches ProviderSpec.id, so log lines and ledger rows can name it.
    id: str

    @abstractmethod
    async def chat_completion(self, payload: dict[str, Any]) -> httpx.Response:
        ...

    @abstractmethod
    def stream_chat_completion(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        ...

    @abstractmethod
    async def is_healthy(self) -> bool:
        ...

    @abstractmethod
    async def aclose(self) -> None:
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.id!r}>"
