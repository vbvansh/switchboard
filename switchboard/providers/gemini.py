"""Native adapter for Google's Gemini API.

Same job as `anthropic.py`: translate OpenAI in, translate Google out, so
nothing else in Switchboard has to know Gemini is different.

Google's format diverges further from OpenAI's than Anthropic's does, and in
ways that are easy to get subtly wrong:

1. **The model name is in the URL, not the body.** Requests go to
   `/models/gemini-2.5-flash:generateContent`. Everywhere else in Switchboard a
   model is a field; here it changes the address.

2. **"assistant" is called "model".** Send `role: "assistant"` and the request
   is rejected.

3. **Messages are `contents`, and text is nested in `parts`.** One extra layer
   in each direction.

4. **The system prompt is `systemInstruction`**, a separate top-level field, as
   with Anthropic.

5. **Sampling options are renamed and moved** into `generationConfig`:
   `max_tokens` becomes `maxOutputTokens`, `top_p` becomes `topP`.

6. **Usage is `usageMetadata`**, with `promptTokenCount` and
   `candidatesTokenCount`. Miss this and every Gemini request records as free -
   an accounting error nobody would notice from the outside.

7. **The API key is a header**, `x-goog-api-key`. It can also go in the query
   string, which Google's own examples use; it is deliberately NOT done that way
   here, because query strings end up in server logs, proxy logs and browser
   history, and an API key does not belong in any of them.

TOOL CALLS ARE NOT TRANSLATED, for the same reason as the Anthropic adapter: a
half-working translation fails deep inside somebody's agent. Requests carrying
`tools` are refused with a message pointing at OpenRouter, which implements
them. Plain chat and streaming work fully.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from switchboard.catalog import ProviderSpec
from switchboard.providers.base import (
    Provider,
    ProviderError,
    ProviderNotConfigured,
    ProviderUnavailable,
)
from switchboard.providers.retry import RetryPolicy, with_retries
from switchboard.providers.sse import SSEDecoder

#: Google's finish reasons -> OpenAI's.
FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
}


class UnsupportedFeature(ProviderError):
    """A request uses something this adapter deliberately does not translate."""


# --- Translation ------------------------------------------------------------


def to_gemini_request(payload: dict[str, Any]) -> dict[str, Any]:
    """OpenAI chat-completions request -> Gemini generateContent request."""
    if payload.get("tools") or payload.get("functions"):
        raise UnsupportedFeature(
            "This adapter does not translate tool calls to Gemini's format. "
            "Reach Gemini through an `openai-compatible` provider such as "
            "OpenRouter, which implements them."
        )

    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        text = content if isinstance(content, str) else _flatten(content)

        if role == "system":
            system_parts.append(text)
            continue

        contents.append(
            {
                # Google's word for the assistant is "model".
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": text}],
            }
        )

    request: dict[str, Any] = {"contents": contents}
    if system_parts:
        request["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

    config: dict[str, Any] = {}
    if (value := payload.get("temperature")) is not None:
        config["temperature"] = value
    if (value := payload.get("top_p")) is not None:
        config["topP"] = value
    if (value := payload.get("max_tokens")) is not None:
        config["maxOutputTokens"] = value
    if (value := payload.get("stop")) is not None:
        config["stopSequences"] = [value] if isinstance(value, str) else value
    if config:
        request["generationConfig"] = config

    return request


def _flatten(content: Any) -> str:
    """OpenAI's multimodal shape is a list of typed parts; keep the text."""
    if not isinstance(content, list):
        return ""
    return "".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _candidate_text(candidate: dict[str, Any]) -> str:
    parts = (candidate.get("content") or {}).get("parts") or []
    return "".join(
        part.get("text", "") for part in parts if isinstance(part, dict)
    )


def from_gemini_response(body: dict[str, Any], model: str) -> dict[str, Any]:
    """Gemini generateContent reply -> OpenAI chat-completions reply."""
    candidates = body.get("candidates") or []
    text = _candidate_text(candidates[0]) if candidates else ""
    finish = str(candidates[0].get("finishReason")) if candidates else "STOP"

    usage = body.get("usageMetadata") or {}
    prompt_tokens = int(usage.get("promptTokenCount") or 0)
    completion_tokens = int(usage.get("candidatesTokenCount") or 0)

    return {
        "id": body.get("responseId", "chatcmpl-gemini"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": FINISH_REASONS.get(finish, "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _chunk(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


class StreamTranslator:
    """Turns Gemini's stream into OpenAI-shaped chunks.

    Gemini streams whole response objects rather than deltas: each event is a
    complete `candidates` structure holding only the newest text. Usage totals
    are repeated in every event and are cumulative, so the last one seen wins -
    which is why they are held and emitted once at the end, in the shape
    `streaming.py` already knows how to read.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._started = False

    def _envelope(self, delta: dict[str, Any], finish: str | None) -> dict:
        return {
            "id": "chatcmpl-gemini",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    def translate(self, data: str) -> list[bytes]:
        try:
            event = json.loads(data)
        except ValueError:
            return []
        if not isinstance(event, dict):
            return []

        if usage := event.get("usageMetadata"):
            self._prompt_tokens = int(usage.get("promptTokenCount") or 0)
            self._completion_tokens = int(usage.get("candidatesTokenCount") or 0)

        candidates = event.get("candidates") or []
        text = _candidate_text(candidates[0]) if candidates else ""
        if not text:
            return []

        chunks = []
        if not self._started:
            self._started = True
            chunks.append(_chunk(self._envelope({"role": "assistant"}, None)))
        chunks.append(_chunk(self._envelope({"content": text}, None)))
        return chunks

    def finish(self) -> list[bytes]:
        return [
            _chunk(self._envelope({}, "stop")),
            _chunk(
                {
                    "id": "chatcmpl-gemini",
                    "object": "chat.completion.chunk",
                    "model": self.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": self._prompt_tokens,
                        "completion_tokens": self._completion_tokens,
                        "total_tokens": self._prompt_tokens
                        + self._completion_tokens,
                    },
                }
            ),
            b"data: [DONE]\n\n",
        ]


# --- The adapter ------------------------------------------------------------


class GeminiProvider(Provider):
    def __init__(self, spec: ProviderSpec, retry: RetryPolicy | None = None) -> None:
        self.id = spec.id
        self.spec = spec
        self.retry = retry or RetryPolicy()

        if not spec.key_is_available:
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
        headers = {"content-type": "application/json"}
        if (key := spec.api_key()) is not None:
            # A header, never the query string. Query strings are written to
            # server logs, proxy logs and browser history.
            headers["x-goog-api-key"] = key
        return headers

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_completion(self, payload: dict[str, Any]) -> httpx.Response:
        model = str(payload.get("model") or "")
        request = to_gemini_request(payload)

        async def send() -> httpx.Response:
            return await self._client.post(
                f"/models/{model}:generateContent", json=request
            )

        try:
            upstream = await with_retries(send, self.retry, f"{self.id} completion")
        except httpx.ConnectError as exc:
            raise ProviderUnavailable(self._unreachable(exc)) from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailable(
                f"{self.id} did not respond within "
                f"{self.spec.timeout_seconds:.0f}s."
            ) from exc

        if upstream.status_code >= 400:
            return upstream

        return httpx.Response(
            status_code=upstream.status_code,
            content=json.dumps(
                from_gemini_response(upstream.json(), model)
            ).encode(),
            headers={"content-type": "application/json"},
        )

    async def stream_chat_completion(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        model = str(payload.get("model") or "")
        request = to_gemini_request(payload)
        translator = StreamTranslator(model)
        decoder = SSEDecoder()

        try:
            async with self._client.stream(
                "POST",
                f"/models/{model}:streamGenerateContent",
                params={"alt": "sse"},
                json=request,
            ) as response:
                response.raise_for_status()
                async for raw in response.aiter_bytes():
                    for event in decoder.feed(raw):
                        for chunk in translator.translate(event.data):
                            yield chunk
                for chunk in translator.finish():
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
        return (
            f"Cannot reach {self.id} at {self.spec.base_url}. "
            f"Check the base_url and your network. ({exc})"
        )
