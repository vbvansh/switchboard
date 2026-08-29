"""Native adapter for Anthropic's Messages API (Claude).

WHY THIS EXISTS. Almost every provider copied OpenAI's request format, so one
adapter covers most of the industry. Anthropic did not. Their API is a different
shape, and until now the only way to reach Claude through Switchboard was via a
reseller like OpenRouter, which works but adds a middleman, their markup, and
their outage.

WHAT AN ADAPTER ACTUALLY DOES. Switchboard speaks OpenAI to your application, in
both directions, always. This file translates:

    OpenAI request  ->  Anthropic request      (on the way out)
    Anthropic reply ->  OpenAI reply           (on the way back)

Your application never learns that Claude is different, the ledger records
tokens the same way, and the router treats it like any other model.

THE FOUR DIFFERENCES THAT MATTER, and each one is a real bug if missed:

1. **The system prompt is not a message.** OpenAI puts it in the messages list
   with `role: "system"`. Anthropic has a separate top-level `system` field.
   Leaving it in the list is rejected outright.

2. **`max_tokens` is required.** OpenAI treats it as optional and defaults to
   the model's limit. Anthropic refuses a request without it, so a default is
   supplied here - and it is generous, because silently truncating somebody's
   answer is worse than a slightly larger bill.

3. **Content is a list, not a string.** Anthropic answers with
   `[{"type": "text", "text": "..."}]`, allowing several blocks. They are joined
   back into the single string OpenAI clients expect.

4. **Usage has different names.** `input_tokens` / `output_tokens` rather than
   `prompt_tokens` / `completion_tokens`. Getting this wrong would not crash
   anything - it would silently record every Claude request as costing nothing,
   which is exactly the kind of quiet accounting error this project keeps
   guarding against.

TOOL CALLS ARE NOT TRANSLATED. Anthropic's tool format differs from OpenAI's in
more than naming, and a half-working translation is worse than an honest gap: it
would fail deep inside somebody's agent with a confusing error. A request
carrying `tools` is refused here with a message saying to use OpenRouter, which
does implement it. Plain chat and streaming work fully.
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

#: Anthropic pins its API shape to a date. Sending it is required.
API_VERSION = "2023-06-01"

#: Used when the caller did not set max_tokens, which OpenAI clients often
#: do not. High on purpose: truncating an answer to save a few cents produces a
#: broken response that the user then pays to ask for again.
DEFAULT_MAX_TOKENS = 4096

#: Anthropic's stop reasons -> OpenAI's finish reasons.
FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


class UnsupportedFeature(ProviderError):
    """A request uses something this adapter deliberately does not translate."""


# --- Translation ------------------------------------------------------------
# Plain functions, no network, no state - so they can be tested exhaustively
# against recorded payloads without an API key.


def to_anthropic_request(payload: dict[str, Any]) -> dict[str, Any]:
    """OpenAI chat-completions request -> Anthropic messages request."""
    if payload.get("tools") or payload.get("functions"):
        raise UnsupportedFeature(
            "This adapter does not translate tool calls to Anthropic's format. "
            "Reach Claude through an `openai-compatible` provider such as "
            "OpenRouter, which implements them."
        )

    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []

    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            # Lifted out, not dropped. A system prompt left in the list is
            # rejected by Anthropic, and dropping it would silently change the
            # model's behaviour - the worst of the three options.
            if isinstance(content, str):
                system_parts.append(content)
            continue

        # Anthropic knows "user" and "assistant" only. Anything else is mapped
        # to user rather than refused: a strange role should not be fatal.
        messages.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": content if content is not None else "",
            }
        )

    request: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": messages,
        "max_tokens": payload.get("max_tokens") or DEFAULT_MAX_TOKENS,
    }
    if system_parts:
        request["system"] = "\n\n".join(system_parts)

    for openai_name, anthropic_name in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop", "stop_sequences"),
        ("stream", "stream"),
    ):
        if (value := payload.get(openai_name)) is not None:
            if anthropic_name == "stop_sequences" and isinstance(value, str):
                value = [value]
            request[anthropic_name] = value

    return request


def from_anthropic_response(body: dict[str, Any]) -> dict[str, Any]:
    """Anthropic messages reply -> OpenAI chat-completions reply."""
    text = "".join(
        block.get("text", "")
        for block in body.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    )
    usage = body.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)

    return {
        "id": body.get("id", "chatcmpl-anthropic"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": FINISH_REASONS.get(
                    str(body.get("stop_reason")), "stop"
                ),
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
    """Turns Anthropic's stream of events into OpenAI-shaped chunks.

    Anthropic sends several event types; only three carry anything Switchboard
    needs. The rest are ignored rather than guessed at.

        message_start          -> the id, and the input token count
        content_block_delta    -> a piece of text
        message_delta          -> the output token count, at the end

    Token counts arrive in two different events, so both are held until the
    stream finishes and emitted as one final usage chunk. That is the shape
    `streaming.py` already knows how to read, so the ledger needs no special
    case for Claude.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._id = "chatcmpl-anthropic"
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def _envelope(self, delta: dict[str, Any], finish: str | None) -> dict:
        return {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    def translate(self, name: str, data: str) -> list[bytes]:
        try:
            event = json.loads(data)
        except ValueError:
            return []
        if not isinstance(event, dict):
            return []

        kind = name or str(event.get("type", ""))

        if kind == "message_start":
            message = event.get("message") or {}
            self._id = message.get("id", self._id)
            usage = message.get("usage") or {}
            self._prompt_tokens = int(usage.get("input_tokens") or 0)
            return [_chunk(self._envelope({"role": "assistant"}, None))]

        if kind == "content_block_delta":
            text = (event.get("delta") or {}).get("text")
            if not isinstance(text, str) or not text:
                return []
            return [_chunk(self._envelope({"content": text}, None))]

        if kind == "message_delta":
            usage = event.get("usage") or {}
            self._completion_tokens = int(
                usage.get("output_tokens") or self._completion_tokens
            )
            return []

        return []

    def finish(self) -> list[bytes]:
        """The closing chunks: a finish reason, the usage totals, then DONE."""
        return [
            _chunk(self._envelope({}, "stop")),
            _chunk(
                {
                    "id": self._id,
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


class AnthropicProvider(Provider):
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
        # Anthropic uses its own header, not `Authorization: Bearer`.
        headers = {
            "content-type": "application/json",
            "anthropic-version": API_VERSION,
        }
        if (key := spec.api_key()) is not None:
            headers["x-api-key"] = key
        return headers

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_completion(self, payload: dict[str, Any]) -> httpx.Response:
        request = to_anthropic_request(payload)

        async def send() -> httpx.Response:
            return await self._client.post("/messages", json=request)

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
            # Errors are passed through untranslated. The failover and retry
            # logic keys off the status code, and Anthropic's error bodies are
            # already clear; rewriting them would only lose detail.
            return upstream

        return httpx.Response(
            status_code=upstream.status_code,
            content=json.dumps(from_anthropic_response(upstream.json())).encode(),
            headers={"content-type": "application/json"},
        )

    async def stream_chat_completion(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        request = to_anthropic_request(payload) | {"stream": True}
        translator = StreamTranslator(str(payload.get("model") or ""))
        decoder = SSEDecoder()

        try:
            async with self._client.stream(
                "POST", "/messages", json=request
            ) as response:
                response.raise_for_status()
                async for raw in response.aiter_bytes():
                    for event in decoder.feed(raw):
                        for chunk in translator.translate(event.name, event.data):
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
        # 401 means reachable but misconfigured, which is a configuration
        # problem rather than an outage - and failing readiness for it would
        # hide the real cause behind a generic "not ready".
        return response.status_code < 500

    def _unreachable(self, exc: Exception) -> str:
        return (
            f"Cannot reach {self.id} at {self.spec.base_url}. "
            f"Check the base_url and your network. ({exc})"
        )
