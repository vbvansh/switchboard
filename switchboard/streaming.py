"""Reading token usage out of a streaming response without disturbing it.

A streamed answer arrives as Server-Sent Events: many small `data: {...}` lines.
Two problems for the ledger:

1. Usage totals only appear in the final event, and only if the client asked for
   them (`stream_options.include_usage`). Switchboard asks on the client's
   behalf.
2. Network chunks do not align with lines - one chunk may hold half an event.
   So bytes are buffered and only complete lines are parsed.

The bytes forwarded to the client are never modified. This only observes.
"""

from __future__ import annotations

import json

DATA_PREFIX = b"data:"
DONE_SENTINEL = b"[DONE]"


class UsageSniffer:
    """Watches an SSE byte stream and extracts usage totals if present."""

    def __init__(self) -> None:
        self._buffer = b""
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.text_length = 0

    @property
    def found_usage(self) -> bool:
        return self.prompt_tokens is not None and self.completion_tokens is not None

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            self._consume_line(line.strip())

    def _consume_line(self, line: bytes) -> None:
        if not line.startswith(DATA_PREFIX):
            return

        payload = line[len(DATA_PREFIX) :].strip()
        if not payload or payload == DONE_SENTINEL:
            return

        try:
            event = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            # A malformed or partial event must never break the response the
            # client is reading. Accounting degrades to estimation instead.
            return

        if isinstance(usage := event.get("usage"), dict):
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            if isinstance(prompt, int):
                self.prompt_tokens = prompt
            if isinstance(completion, int):
                self.completion_tokens = completion

        for choice in event.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if isinstance(content, str):
                self.text_length += len(content)


def request_usage_in_stream(payload: dict) -> dict:
    """Ask the provider to include usage totals in the final event.

    Mutates a copy, not the caller's dict. If the client already set
    `stream_options`, their choice is respected - overriding it could break a
    client that is parsing events strictly.
    """
    if not payload.get("stream"):
        return payload
    if "stream_options" in payload:
        return payload
    return payload | {"stream_options": {"include_usage": True}}
