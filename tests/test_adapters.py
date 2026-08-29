"""Native adapters for Anthropic and Gemini.

These providers do not speak OpenAI's format, so the adapters translate in both
directions. Everything here runs against RECORDED payloads - real response
shapes copied from each vendor's documentation - so the whole file passes with
no API key, no network and no spend.

That is also the honest limit of these tests, and it is stated in the README
too: they prove the translation is correct against the documented shape. They
cannot prove the shape is still current. Only a call with a real key does that.

The bug this file exists to prevent is not a crash. It is the quiet one: get
the usage field names wrong and every Claude request records as costing $0.00,
the savings figures look wonderful, and nothing ever errors.
"""

from __future__ import annotations

import json

import pytest

from switchboard.providers import anthropic, gemini
from switchboard.providers.sse import SSEDecoder

# --- The SSE reader ---------------------------------------------------------


def test_events_are_assembled_from_split_chunks() -> None:
    """Network chunks do not line up with events. One event can arrive in
    three pieces, and three events can arrive in one piece."""
    decoder = SSEDecoder()
    assert decoder.feed(b'data: {"a"') == []
    assert decoder.feed(b': 1}\n') == []
    events = decoder.feed(b"\n")
    assert [e.data for e in events] == ['{"a": 1}']


def test_event_names_are_read() -> None:
    decoder = SSEDecoder()
    events = decoder.feed(b"event: message_start\ndata: {}\n\n")
    assert events[0].name == "message_start"


def test_comments_and_keepalives_are_ignored() -> None:
    decoder = SSEDecoder()
    assert decoder.feed(b": keep-alive\n\n") == []


def test_several_events_in_one_chunk() -> None:
    decoder = SSEDecoder()
    events = decoder.feed(b'data: {"a":1}\n\ndata: {"b":2}\n\n')
    assert len(events) == 2


def test_a_stream_cut_mid_character_does_not_raise() -> None:
    """A truncated UTF-8 sequence must not kill a response the client is
    already reading."""
    decoder = SSEDecoder()
    decoder.feed(b"data: \xe2\x82\n\n")  # half a euro sign


# --- Anthropic: requests ----------------------------------------------------


def test_the_system_prompt_is_lifted_out_of_the_messages() -> None:
    """Anthropic rejects a system message in the list. Dropping it instead
    would silently change how the model behaves - the worst of the options."""
    request = anthropic.to_anthropic_request(
        {
            "model": "claude-sonnet-4",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hi"},
            ],
        }
    )
    assert request["system"] == "Be terse."
    assert [m["role"] for m in request["messages"]] == ["user"]


def test_several_system_messages_are_joined() -> None:
    request = anthropic.to_anthropic_request(
        {
            "messages": [
                {"role": "system", "content": "One."},
                {"role": "system", "content": "Two."},
                {"role": "user", "content": "hi"},
            ]
        }
    )
    assert request["system"] == "One.\n\nTwo."


def test_max_tokens_is_always_sent() -> None:
    """Anthropic refuses a request without it; OpenAI clients rarely set it."""
    request = anthropic.to_anthropic_request({"messages": []})
    assert request["max_tokens"] == anthropic.DEFAULT_MAX_TOKENS


def test_an_explicit_max_tokens_wins() -> None:
    request = anthropic.to_anthropic_request({"messages": [], "max_tokens": 50})
    assert request["max_tokens"] == 50


def test_sampling_options_are_renamed() -> None:
    request = anthropic.to_anthropic_request(
        {"messages": [], "temperature": 0.2, "stop": "END"}
    )
    assert request["temperature"] == 0.2
    assert request["stop_sequences"] == ["END"]


def test_unset_options_are_not_sent() -> None:
    """Sending temperature: null is not the same as not sending temperature."""
    request = anthropic.to_anthropic_request({"messages": []})
    assert "temperature" not in request


def test_tool_calls_are_refused_rather_than_half_translated() -> None:
    """An honest gap beats a translation that fails deep inside an agent."""
    with pytest.raises(anthropic.UnsupportedFeature) as caught:
        anthropic.to_anthropic_request({"messages": [], "tools": [{"x": 1}]})
    assert "OpenRouter" in str(caught.value)


# --- Anthropic: responses ---------------------------------------------------

CLAUDE_REPLY = {
    "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4",
    "content": [{"type": "text", "text": "Hello there."}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 12, "output_tokens": 6},
}


def test_the_reply_comes_back_in_openai_shape() -> None:
    reply = anthropic.from_anthropic_response(CLAUDE_REPLY)
    assert reply["object"] == "chat.completion"
    assert reply["choices"][0]["message"]["content"] == "Hello there."
    assert reply["choices"][0]["message"]["role"] == "assistant"


def test_token_counts_are_renamed_not_lost() -> None:
    """THE test in this file. Miss this and every Claude request is recorded as
    free: no error, no crash, and a savings report that is pure fiction."""
    usage = anthropic.from_anthropic_response(CLAUDE_REPLY)["usage"]
    assert usage["prompt_tokens"] == 12
    assert usage["completion_tokens"] == 6
    assert usage["total_tokens"] == 18


def test_several_content_blocks_are_joined() -> None:
    reply = anthropic.from_anthropic_response(
        {
            "content": [
                {"type": "text", "text": "one "},
                {"type": "text", "text": "two"},
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    assert reply["choices"][0]["message"]["content"] == "one two"


@pytest.mark.parametrize(
    "stop_reason,expected",
    [("end_turn", "stop"), ("max_tokens", "length"), ("tool_use", "tool_calls")],
)
def test_stop_reasons_are_mapped(stop_reason: str, expected: str) -> None:
    reply = anthropic.from_anthropic_response(
        {"content": [], "stop_reason": stop_reason}
    )
    assert reply["choices"][0]["finish_reason"] == expected


def test_a_missing_usage_block_does_not_crash() -> None:
    assert anthropic.from_anthropic_response({"content": []})["usage"][
        "total_tokens"
    ] == 0


# --- Anthropic: streaming ---------------------------------------------------


def _decode(chunks: list[bytes]) -> list[dict]:
    events = []
    for chunk in chunks:
        body = chunk.removeprefix(b"data: ").strip()
        if body != b"[DONE]":
            events.append(json.loads(body))
    return events


def test_a_claude_stream_becomes_an_openai_stream() -> None:
    translator = anthropic.StreamTranslator("claude-sonnet-4")
    chunks: list[bytes] = []
    chunks += translator.translate(
        "message_start",
        json.dumps({"message": {"id": "msg_1", "usage": {"input_tokens": 9}}}),
    )
    chunks += translator.translate(
        "content_block_delta", json.dumps({"delta": {"text": "Hel"}})
    )
    chunks += translator.translate(
        "content_block_delta", json.dumps({"delta": {"text": "lo"}})
    )
    chunks += translator.translate(
        "message_delta", json.dumps({"usage": {"output_tokens": 4}})
    )
    chunks += translator.finish()

    text = "".join(
        (event.get("choices") or [{}])[0].get("delta", {}).get("content", "")
        for event in _decode(chunks)
        if event.get("choices")
    )
    assert text == "Hello"
    assert chunks[-1] == b"data: [DONE]\n\n"


def test_the_stream_ends_with_usage_the_ledger_can_read() -> None:
    """Token counts arrive in two different Anthropic events. They are held and
    emitted once, in the shape switchboard/streaming.py already parses - so the
    ledger needs no special case for Claude."""
    from switchboard.streaming import UsageSniffer

    translator = anthropic.StreamTranslator("claude-sonnet-4")
    sniffer = UsageSniffer()
    for chunk in (
        translator.translate(
            "message_start", json.dumps({"message": {"usage": {"input_tokens": 9}}})
        )
        + translator.translate(
            "message_delta", json.dumps({"usage": {"output_tokens": 4}})
        )
        + translator.finish()
    ):
        sniffer.feed(chunk)

    assert sniffer.found_usage
    assert (sniffer.prompt_tokens, sniffer.completion_tokens) == (9, 4)


def test_a_malformed_stream_event_is_skipped_not_fatal() -> None:
    translator = anthropic.StreamTranslator("claude-sonnet-4")
    assert translator.translate("content_block_delta", "{not json") == []


# --- Gemini: requests -------------------------------------------------------


def test_assistant_is_renamed_to_model() -> None:
    """Google's word for the assistant is "model". Send "assistant" and the
    request is rejected."""
    request = gemini.to_gemini_request(
        {"messages": [{"role": "assistant", "content": "hi"}]}
    )
    assert request["contents"][0]["role"] == "model"


def test_text_is_nested_in_parts() -> None:
    request = gemini.to_gemini_request(
        {"messages": [{"role": "user", "content": "hello"}]}
    )
    assert request["contents"][0]["parts"] == [{"text": "hello"}]


def test_the_system_prompt_becomes_a_system_instruction() -> None:
    request = gemini.to_gemini_request(
        {
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hi"},
            ]
        }
    )
    assert request["systemInstruction"]["parts"][0]["text"] == "Be terse."
    assert len(request["contents"]) == 1


def test_sampling_options_move_into_generation_config() -> None:
    request = gemini.to_gemini_request(
        {"messages": [], "temperature": 0.3, "max_tokens": 100, "top_p": 0.9}
    )
    config = request["generationConfig"]
    assert config["temperature"] == 0.3
    assert config["maxOutputTokens"] == 100
    assert config["topP"] == 0.9


def test_no_generation_config_when_nothing_was_set() -> None:
    assert "generationConfig" not in gemini.to_gemini_request({"messages": []})


def test_multimodal_text_parts_are_flattened() -> None:
    request = gemini.to_gemini_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": "..."}},
                    ],
                }
            ]
        }
    )
    assert request["contents"][0]["parts"][0]["text"] == "look"


def test_gemini_tool_calls_are_refused() -> None:
    with pytest.raises(gemini.UnsupportedFeature):
        gemini.to_gemini_request({"messages": [], "tools": [{"x": 1}]})


# --- Gemini: responses ------------------------------------------------------

GEMINI_REPLY = {
    "candidates": [
        {
            "content": {"parts": [{"text": "Hello there."}], "role": "model"},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 11,
        "candidatesTokenCount": 5,
        "totalTokenCount": 16,
    },
}


def test_a_gemini_reply_comes_back_in_openai_shape() -> None:
    reply = gemini.from_gemini_response(GEMINI_REPLY, "gemini-2.5-flash")
    assert reply["choices"][0]["message"]["content"] == "Hello there."
    assert reply["model"] == "gemini-2.5-flash"


def test_gemini_token_counts_are_renamed_not_lost() -> None:
    usage = gemini.from_gemini_response(GEMINI_REPLY, "gemini-2.5-flash")["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (11, 5)


def test_a_safety_block_is_reported_as_a_content_filter() -> None:
    reply = gemini.from_gemini_response(
        {"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]},
        "gemini-2.5-flash",
    )
    assert reply["choices"][0]["finish_reason"] == "content_filter"


def test_an_empty_candidate_list_does_not_crash() -> None:
    reply = gemini.from_gemini_response({}, "gemini-2.5-flash")
    assert reply["choices"][0]["message"]["content"] == ""


def test_a_gemini_stream_becomes_an_openai_stream() -> None:
    translator = gemini.StreamTranslator("gemini-2.5-flash")
    chunks: list[bytes] = []
    for piece in ("Hel", "lo"):
        chunks += translator.translate(
            json.dumps(
                {
                    "candidates": [{"content": {"parts": [{"text": piece}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 7,
                        "candidatesTokenCount": 2,
                    },
                }
            )
        )
    chunks += translator.finish()

    from switchboard.streaming import UsageSniffer

    sniffer = UsageSniffer()
    for chunk in chunks:
        sniffer.feed(chunk)
    assert sniffer.text_length == 5
    assert (sniffer.prompt_tokens, sniffer.completion_tokens) == (7, 2)


# --- Wiring -----------------------------------------------------------------


def test_both_adapters_are_registered() -> None:
    """A translation nothing can select is a translation nobody can use."""
    from switchboard.providers import ADAPTERS

    assert ADAPTERS["anthropic"] is anthropic.AnthropicProvider
    assert ADAPTERS["gemini"] is gemini.GeminiProvider


def test_the_catalog_accepts_the_new_provider_types() -> None:
    from switchboard.catalog import KNOWN_PROVIDER_TYPES

    assert {"anthropic", "gemini"} <= KNOWN_PROVIDER_TYPES


def test_anthropic_uses_its_own_auth_header(monkeypatch) -> None:
    """Anthropic does not use `Authorization: Bearer`. Sending the wrong header
    fails as a 401, which reads like a bad key rather than a bad adapter."""
    from switchboard.catalog import ProviderSpec

    monkeypatch.setenv("FAKE_KEY", "secret-value")
    spec = ProviderSpec(
        id="anthropic",
        type="anthropic",
        base_url="https://api.anthropic.com/v1",
        enabled=True,
        api_key_env="FAKE_KEY",
    )
    headers = anthropic.AnthropicProvider._headers(spec)
    assert headers["x-api-key"] == "secret-value"
    assert headers["anthropic-version"] == anthropic.API_VERSION
    assert "Authorization" not in headers


def test_gemini_sends_its_key_as_a_header_not_a_query_string(monkeypatch) -> None:
    """Google's own examples put the key in the URL. Query strings end up in
    server logs, proxy logs and browser history."""
    from switchboard.catalog import ProviderSpec

    monkeypatch.setenv("FAKE_KEY", "secret-value")
    spec = ProviderSpec(
        id="gemini",
        type="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        enabled=True,
        api_key_env="FAKE_KEY",
    )
    assert gemini.GeminiProvider._headers(spec)["x-goog-api-key"] == "secret-value"


def test_a_missing_key_is_reported_clearly(monkeypatch) -> None:
    from switchboard.catalog import ProviderSpec
    from switchboard.providers.base import ProviderNotConfigured

    monkeypatch.delenv("ABSENT_KEY", raising=False)
    spec = ProviderSpec(
        id="anthropic",
        type="anthropic",
        base_url="https://api.anthropic.com/v1",
        enabled=True,
        api_key_env="ABSENT_KEY",
    )
    with pytest.raises(ProviderNotConfigured) as caught:
        anthropic.AnthropicProvider(spec)
    assert "ABSENT_KEY" in str(caught.value)
