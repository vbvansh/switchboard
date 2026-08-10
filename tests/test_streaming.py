"""Reading usage out of a streamed response."""

from __future__ import annotations

from switchboard.streaming import UsageSniffer, request_usage_in_stream


def test_usage_is_captured_from_the_final_event() -> None:
    sniffer = UsageSniffer()
    sniffer.feed(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
    sniffer.feed(
        b'data: {"choices":[],"usage":'
        b'{"prompt_tokens":12,"completion_tokens":34}}\n\n'
    )
    sniffer.feed(b"data: [DONE]\n\n")

    assert sniffer.found_usage
    assert (sniffer.prompt_tokens, sniffer.completion_tokens) == (12, 34)


def test_events_split_across_chunks_are_reassembled() -> None:
    """Network chunks do not respect line boundaries."""
    sniffer = UsageSniffer()
    payload = (
        b'data: {"choices":[],"usage":'
        b'{"prompt_tokens":7,"completion_tokens":8}}\n'
    )
    for i in range(0, len(payload), 5):
        sniffer.feed(payload[i : i + 5])

    assert sniffer.found_usage
    assert (sniffer.prompt_tokens, sniffer.completion_tokens) == (7, 8)


def test_missing_usage_is_reported_as_missing() -> None:
    sniffer = UsageSniffer()
    sniffer.feed(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
    sniffer.feed(b"data: [DONE]\n\n")
    assert not sniffer.found_usage


def test_content_length_accumulates_for_fallback_estimation() -> None:
    sniffer = UsageSniffer()
    sniffer.feed(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
    sniffer.feed(b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n')
    assert sniffer.text_length == len("hello world")


def test_malformed_events_are_ignored() -> None:
    """A broken event must never break the response the client is reading."""
    sniffer = UsageSniffer()
    sniffer.feed(b"data: {not valid json\n\n")
    sniffer.feed(b": this is an SSE comment\n\n")
    sniffer.feed(
        b'data: {"choices":[],"usage":'
        b'{"prompt_tokens":1,"completion_tokens":2}}\n\n'
    )
    assert sniffer.found_usage


def test_usage_is_requested_for_streams() -> None:
    result = request_usage_in_stream({"stream": True, "messages": []})
    assert result["stream_options"] == {"include_usage": True}


def test_non_streaming_payloads_are_untouched() -> None:
    payload = {"messages": []}
    assert request_usage_in_stream(payload) == payload


def test_client_stream_options_are_respected() -> None:
    """Overriding an explicit client choice could break a strict parser."""
    payload = {"stream": True, "stream_options": {"include_usage": False}}
    assert request_usage_in_stream(payload)["stream_options"] == {
        "include_usage": False
    }
