"""Reading Server-Sent Events out of a byte stream.

Every streaming AI provider uses the same transport - a long HTTP response made
of small text blocks like this:

    event: content_block_delta
    data: {"delta": {"text": "hello"}}

    data: {"delta": {"text": " world"}}

The catch is that network chunks do not line up with those blocks. One chunk can
hold half an event, or three events and a fragment of a fourth. So bytes are
buffered here and only complete events are handed on.

This is a *decoder*, not a translator. It says what arrived; deciding what an
event means belongs to the adapter for that provider.

`switchboard/streaming.py` does something related but different: it watches the
already-OpenAI-shaped stream on its way to the client and never alters it. This
module is used earlier, by adapters that have to rebuild the stream.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SSEEvent:
    """One complete event. `name` is empty when the stream does not use them."""

    name: str
    data: str


class SSEDecoder:
    """Feed it bytes, get back whole events."""

    def __init__(self) -> None:
        self._buffer = ""
        self._name = ""
        self._data: list[str] = []

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        # errors="replace" rather than strict: a stream cut mid-character must
        # not raise and kill a response the client is already reading.
        self._buffer += chunk.decode("utf-8", errors="replace")
        events: list[SSEEvent] = []

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")

            if not line:
                # A blank line ends an event. That is the only terminator SSE
                # has, which is why a stream that stops mid-event simply
                # produces nothing rather than producing something wrong.
                if self._data:
                    events.append(SSEEvent(self._name, "\n".join(self._data)))
                self._name, self._data = "", []
                continue

            if line.startswith(":"):
                continue  # a comment, usually a keep-alive ping

            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value

            if field == "event":
                self._name = value
            elif field == "data":
                self._data.append(value)

        return events

    def flush(self) -> list[SSEEvent]:
        """Whatever is left when the connection closes without a blank line."""
        if not self._data:
            return []
        event = SSEEvent(self._name, "\n".join(self._data))
        self._name, self._data = "", []
        return [event]
