"""Checking an answer after it arrives, and escalating when that would help.

This is the cold-start answer. Two experiments showed a prompt cannot be judged
before it is answered — difficulty only shows in the text when it comes from how
much work is required, and most real traffic is not like that. So the judgement
moved to after the call, where the answer is sitting in memory and costs nothing
to look at.

Four properties are load-bearing:

1. **A check may only escalate if escalating would fix the problem.** Three of
   the five deliberately do not.
2. **A refusal is never escalated.** Retrying until a model complies is shopping
   for a yes.
3. **An escalated request is charged for BOTH calls.** Charging only for the
   final model is the easiest way to make this feature look free.
4. **Nothing escalates unless the operator switched it on.** The default checks
   and records; it does not double anybody's bill by surprise.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from switchboard import verification as v
from switchboard.catalog import ModelCatalog
from switchboard.ledger.models import RequestLog
from switchboard.routing.base import RoutingContext
from switchboard.routing.ladder import LadderRouter, build_ladder


def body(content: str, finish: str = "stop") -> dict:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content},
             "finish_reason": finish}
        ]
    }


# --- The checks -------------------------------------------------------------


def test_a_good_answer_passes() -> None:
    assert not v.inspect(body("Use enumerate() to get the index.")).failed


@pytest.mark.parametrize("content", ["", " ", "\n", "\t "])
def test_an_empty_answer_is_caught(content: str) -> None:
    """Some providers return a stray space rather than nothing on a failed
    generation, so this cannot just test for the empty string."""
    found = v.inspect(body(content))
    assert found.failed
    assert found.names == "empty_response"


def test_an_empty_answer_escalates() -> None:
    """A different model may well produce content, so retrying helps."""
    assert v.inspect(body("")).should_escalate


def test_nothing_else_is_reported_about_an_empty_answer() -> None:
    """"empty AND not valid JSON" is noise, not two findings."""
    found = v.inspect(body(""), {"response_format": {"type": "json_object"}})
    assert found.names == "empty_response"


def test_truncation_is_caught() -> None:
    found = v.inspect(body("The answer is", finish="length"))
    assert "truncated" in found.names


def test_truncation_does_NOT_escalate() -> None:
    """THE rule this file exists for. A stronger model hits the same
    max_tokens, so retrying reproduces the cut-off answer and bills twice. The
    fix is to raise max_tokens, and the report says so."""
    assert not v.inspect(body("cut off", finish="length")).should_escalate
    assert "raise max_tokens" in v.CHECKS["truncated"].explanation


@pytest.mark.parametrize(
    "content",
    [
        "I cannot help with that.",
        "I'm sorry, I can't assist with this request.",
        "Sorry, but I am unable to do that.",
        "As an AI, I cannot provide that.",
    ],
)
def test_a_refusal_is_caught(content: str) -> None:
    assert "refused" in v.inspect(body(content)).names


def test_a_refusal_is_NEVER_escalated() -> None:
    """The safety rule. If a model declines, sending the request up the ladder
    until one complies is shopping for a yes. Switchboard records the refusal
    and passes it through."""
    found = v.inspect(body("I'm sorry, I can't help with that."))
    assert found.failed
    assert not found.should_escalate
    assert not v.CHECKS["refused"].escalates


def test_i_cannot_deep_inside_a_long_answer_is_not_a_refusal() -> None:
    """Only the opening is scanned. A model that is going to decline does it
    immediately; the same words in paragraph four are prose."""
    prose = "Here is how to do it. " * 40 + "I cannot stress this enough."
    assert "refused" not in v.inspect(body(prose)).names


def test_hedging_is_flagged_but_never_escalated() -> None:
    """"I don't know" may be the correct answer. Escalating on uncertainty
    punishes a model for being honest and would rank a confident wrong answer
    above a cautious right one."""
    found = v.inspect(body("I'm not sure, the documentation is ambiguous here."))
    assert "hedged" in found.names
    assert not found.should_escalate


def test_a_refusal_is_not_double_counted_as_hedging() -> None:
    found = v.inspect(body("I'm sorry, I'm not sure I can help with that."))
    assert "refused" in found.names
    assert "hedged" not in found.names


# --- JSON -------------------------------------------------------------------

WANTS_JSON = {"response_format": {"type": "json_object"}}


def test_invalid_json_is_caught_when_json_was_requested() -> None:
    found = v.inspect(body("Sure! Here you go: name = Alice"), WANTS_JSON)
    assert "invalid_json" in found.names
    assert found.should_escalate


def test_valid_json_passes() -> None:
    assert not v.inspect(body('{"name": "Alice"}'), WANTS_JSON).failed


def test_json_wrapped_in_a_code_fence_still_passes() -> None:
    """Models wrap JSON in ```json even when told not to. That is a formatting
    quirk, not a failure to produce JSON, and counting it as one would escalate
    a large share of perfectly good answers."""
    fenced = '```json\n{"name": "Alice"}\n```'
    assert not v.inspect(body(fenced), WANTS_JSON).failed


def test_prose_is_not_checked_for_json_when_none_was_asked_for() -> None:
    """Otherwise every ordinary English answer ever given is 'invalid JSON'."""
    assert not v.inspect(body("The capital of France is Paris.")).failed


def test_a_malformed_response_body_does_not_raise() -> None:
    for junk in (None, {}, {"choices": []}, {"choices": [{}]}, "not a dict"):
        assert v.inspect(junk).names == "empty_response"


# --- The ladder -------------------------------------------------------------


@pytest.fixture
def ladder(prices: ModelCatalog) -> LadderRouter:
    return build_ladder(prices, prices.known_models())


def test_the_ladder_picks_the_cheapest_model(ladder, prices) -> None:
    decision = ladder.choose(
        RoutingContext(messages=[{"role": "user", "content": "fix this bug"}])
    )
    assert decision.model == prices.ladder[0]


def test_the_ladder_admits_it_is_not_predicting(ladder) -> None:
    """The keyword heuristic scored 77.8% on one benchmark and 57.9% on
    another - worse than always-cheapest. This router therefore guesses
    nothing, and its reason says so rather than implying a judgement."""
    decision = ladder.choose(
        RoutingContext(messages=[{"role": "user", "content": "prove P != NP"}])
    )
    assert "no prediction" in decision.reason


def test_a_hard_looking_prompt_gets_the_same_model_as_an_easy_one(ladder) -> None:
    """Deliberate. Guessing from wording is the thing that was measured and
    found to be worse than useless."""
    easy = ladder.choose(RoutingContext(messages=[{"role": "user", "content": "hi"}]))
    hard = ladder.choose(
        RoutingContext(
            messages=[{"role": "user", "content": "derive the Euler-Lagrange equation"}]
        )
    )
    assert easy.model == hard.model


def test_a_prompt_too_big_for_a_model_skips_it(prices) -> None:
    """A context window is a hard limit, not an opinion - which is why this is
    the one thing the ladder is allowed to act on."""
    router = build_ladder(prices, prices.known_models())
    small = router.models[0]
    assert router.fits(small, 100)
    assert not router.fits(small, 10_000_000)


def test_an_enormous_prompt_gets_the_roomiest_model_rather_than_an_error(
    ladder,
) -> None:
    decision = ladder.choose(
        RoutingContext(messages=[{"role": "user", "content": "x" * 5_000_000}])
    )
    assert decision.model
    assert "largest context window" in decision.reason


def test_escalation_walks_one_rung_at_a_time(ladder) -> None:
    """Jumping to the most expensive model turns every detected failure into
    the largest possible bill."""
    first = ladder.models[0]
    second = ladder.next_model(first)
    assert second == ladder.models[1]
    assert ladder.next_model(ladder.models[-1]) is None


def test_a_cost_cap_narrows_the_choice(ladder) -> None:
    from switchboard.routing.live import RequestLimits

    decision = ladder.choose(
        RoutingContext(messages=[{"role": "user", "content": "hello"}]),
        RequestLimits(max_cost_usd=0.001),
    )
    assert decision.model in ladder.models


def test_an_impossible_cap_is_ignored_rather_than_obeyed(ladder) -> None:
    """Failing a request because its budget was unsatisfiable is worse than
    serving it and recording the overrun."""
    from switchboard.routing.live import RequestLimits

    decision = ladder.choose(
        RoutingContext(messages=[{"role": "user", "content": "hello"}]),
        RequestLimits(max_cost_usd=0.0000001),
    )
    assert decision.model


def test_one_model_is_not_a_ladder(prices) -> None:
    assert build_ladder(prices, [prices.ladder[0]]) is None
    assert build_ladder(prices, []) is None


# --- Through the API --------------------------------------------------------


def _chat(text: str = "hello", **extra) -> dict:
    return {
        "model": "auto",
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        **extra,
    }


@pytest.fixture
def flagging(client, monkeypatch):
    from switchboard.config import settings

    monkeypatch.setattr(settings, "verify_mode", "flag")
    return client


@pytest.fixture
def escalating(client, monkeypatch):
    from switchboard.config import settings

    monkeypatch.setattr(settings, "verify_mode", "escalate")
    return client


def test_a_clean_answer_is_recorded_as_examined(flagging, auth, database) -> None:
    """"" means checked and fine; NULL means never checked. Different facts."""
    flagging.post("/v1/chat/completions", json=_chat(), headers=auth)
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.verification == ""
    assert row.attempts == 1


def test_with_verification_off_nothing_is_recorded(client, auth, database, monkeypatch):
    from switchboard.config import settings

    monkeypatch.setattr(settings, "verify_mode", "off")
    client.post("/v1/chat/completions", json=_chat(), headers=auth)
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.verification is None


def test_flag_mode_records_a_failure_without_retrying(
    flagging, auth, provider, database
) -> None:
    """THE default. It measures how often checks fire on your traffic without
    spending a penny extra, so you can decide whether to switch escalation on."""
    provider.content = ""
    flagging.post("/v1/chat/completions", json=_chat(), headers=auth)

    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.verification == "empty_response"
    assert row.attempts == 1
    assert row.escalated_from is None


def test_escalate_mode_retries_on_the_next_model_up(
    escalating, auth, provider, database
) -> None:
    provider.content = ""
    escalating.post("/v1/chat/completions", json=_chat(), headers=auth)

    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.attempts == 2
    assert row.escalated_from
    assert row.escalated_from != row.served_model


def test_an_escalated_request_is_charged_for_both_calls(
    escalating, auth, provider, database, prices
) -> None:
    """THE accounting rule. Charging only for the model that produced the final
    answer is the easiest way to make escalation look free, and it is the same
    error the cascade scoring was built to avoid."""
    provider.content = ""
    escalating.post("/v1/chat/completions", json=_chat(), headers=auth)

    with database.session() as session:
        row = session.scalar(select(RequestLog))

    second_only = prices.cost(
        row.served_model, row.prompt_tokens, row.completion_tokens
    )
    assert row.simulated_cost_usd > second_only


def test_the_reason_explains_the_escalation(escalating, auth, provider, database):
    provider.content = ""
    escalating.post("/v1/chat/completions", json=_chat(), headers=auth)
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert "empty_response" in row.routing_reason
    assert "retried on" in row.routing_reason


def test_a_refusal_is_not_escalated_through_the_api(
    escalating, auth, provider, database
) -> None:
    """The safety rule, end to end. A refused request must be recorded and
    passed through, never retried on a stronger model."""
    provider.content = "I'm sorry, I can't help with that."
    escalating.post("/v1/chat/completions", json=_chat(), headers=auth)

    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert "refused" in row.verification
    assert row.attempts == 1
    assert row.escalated_from is None


def test_nothing_escalates_from_the_top_of_the_ladder(
    escalating, auth, provider, database, prices
) -> None:
    """A normal outcome, not an error: the answer is returned as it is."""
    provider.content = ""
    escalating.post(
        "/v1/chat/completions",
        json=_chat() | {"model": prices.ladder[-1]},
        headers=auth,
    )
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.attempts == 1


def test_escalation_is_counted_in_metrics(escalating, auth, provider) -> None:
    provider.content = ""
    escalating.post("/v1/chat/completions", json=_chat(), headers=auth)
    body_text = escalating.get("/metrics").text
    assert "switchboard_escalations_total" in body_text
    assert "switchboard_verification_total" in body_text


def test_verification_never_records_the_answer_text(
    flagging, auth, provider, database
) -> None:
    """Only which checks fired. The verification column must not become a
    second place answers get stored."""
    provider.content = "I'm sorry, I cannot discuss Aunt Mildred's medication."
    flagging.post("/v1/chat/completions", json=_chat(), headers=auth)
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert "Mildred" not in (row.verification or "")


def test_a_json_request_that_comes_back_as_prose_escalates(
    escalating, auth, provider, database
) -> None:
    provider.content = "Sure! The name is Alice."
    escalating.post(
        "/v1/chat/completions",
        json=_chat(response_format={"type": "json_object"}),
        headers=auth,
    )
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert "invalid_json" in row.verification
    assert row.attempts == 2


def test_streaming_is_left_alone(escalating, auth, database) -> None:
    """A streamed answer is already on its way to the client before it could
    be checked, so verification does not apply. Recorded as never examined
    rather than pretending it passed."""
    escalating.post(
        "/v1/chat/completions", json=_chat() | {"stream": True}, headers=auth
    )
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.verification is None


def test_health_reports_the_ladder(client) -> None:
    routing = client.get("/health").json()["routing"]
    assert routing is not None


def test_the_response_is_still_a_normal_openai_body(escalating, auth, provider):
    provider.content = ""
    response = escalating.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert response.status_code == 200
    assert "choices" in json.loads(response.content)
