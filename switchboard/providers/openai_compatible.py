"""Adapter for anything speaking OpenAI's chat-completions format.

This one class covers most of the industry. OpenAI defined the format and it
became the de-facto standard, so Ollama, Groq, OpenRouter, Together, DeepSeek,
Mistral, vLLM, LM Studio, llama.cpp and TGI all accept it.

Anthropic's own API uses a different shape. Claude models are still reachable
here through OpenRouter, which exposes them in this format - so no separate
adapter is needed to use Claude. A native Anthropic adapter can be added later
by implementing Provider; nothing else has to change.

Requests are forwarded as-is. Not translating means client features this code
has never heard of keep working.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from switchboard.catalog import ProviderSpec
from switchboard.providers.base import (
    Provider,
    ProviderNotConfigured,
    ProviderUnavailable,
)


class OpenAICompatibleProvider(Provider):
    def __init__(self, spec: ProviderSpec) -> None:
        self.id = spec.id
        self.spec = spec

        if spec.requires_key and not spec.key_is_available:
            raise ProviderNotConfigured(
                f"Provider {spec.id!r} is enabled but its API key is missing. "
                f"Set the {spec.api_key_env} environment variable, or set "
                f"`enabled: false` for it in providers.yaml."
            )

        self._client = httpx.AsyncClient(
            base_url=spec.base_url,
            timeout=httpx.Timeout(spec.timeout_seconds, connect=10.0),
            headers=self._headers(spec),
        )

    @staticmethod
    def _headers(spec: ProviderSpec) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if (key := spec.api_key()) is not None:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_completion(self, payload: dict[str, Any]) -> httpx.Response:
        try:
            return await self._client.post("/chat/completions", json=payload)
        except httpx.ConnectError as exc:
            raise ProviderUnavailable(self._unreachable(exc)) from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailable(
                f"{self.id} did not respond within "
                f"{self.spec.timeout_seconds:.0f}s."
            ) from exc

    async def stream_chat_completion(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Forward upstream bytes untouched so streaming stays transparent."""
        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.ConnectError as exc:
            raise ProviderUnavailable(self._unreachable(exc)) from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailable(
                f"{self.id} did not respond within "
                f"{self.spec.timeout_seconds:.0f}s."
            ) from exc

    async def list_models(self) -> httpx.Response:
        try:
            return await self._client.get("/models")
        except httpx.ConnectError as exc:
            raise ProviderUnavailable(self._unreachable(exc)) from exc

    async def is_healthy(self) -> bool:
        try:
            response = await self._client.get("/models")
        except httpx.HTTPError:
            return False
        return response.status_code < 500

    def _unreachable(self, exc: Exception) -> str:
        hint = (
            " Start it with `ollama serve`."
            if self.spec.is_local
            else " Check the base_url and your network."
        )
        return f"Cannot reach {self.id} at {self.spec.base_url}.{hint} ({exc})"
