"""Thin async client for Ollama's OpenAI-compatible endpoint.

Ollama already speaks the OpenAI wire format at /v1, so this is a pass-through
rather than a translation layer. Keeping it a pass-through matters: it means
Switchboard stays compatible with OpenAI clients for free, including fields
this code has never heard of.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from switchboard.config import Settings


class ProviderUnavailable(RuntimeError):
    """Ollama could not be reached."""


class OllamaProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.openai_compat_url,
            timeout=httpx.Timeout(
                settings.read_timeout,
                connect=settings.connect_timeout,
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_completion(self, payload: dict[str, Any]) -> httpx.Response:
        """Non-streaming completion. Returns the raw upstream response."""
        try:
            return await self._client.post("/chat/completions", json=payload)
        except httpx.ConnectError as exc:
            raise ProviderUnavailable(str(exc)) from exc

    async def stream_chat_completion(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Streaming completion, forwarding upstream SSE bytes untouched."""
        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.ConnectError as exc:
            raise ProviderUnavailable(str(exc)) from exc

    async def list_models(self) -> httpx.Response:
        try:
            return await self._client.get("/models")
        except httpx.ConnectError as exc:
            raise ProviderUnavailable(str(exc)) from exc

    async def is_healthy(self) -> bool:
        try:
            response = await self._client.get("/models")
        except httpx.HTTPError:
            return False
        return response.status_code == httpx.codes.OK
