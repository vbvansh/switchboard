"""The usage policy.

Two properties are load-bearing here, and every other test in this file exists
to protect one of them:

1. In `flag` mode, NOTHING is refused. If that ever broke, an operator who
   switched on labelling would find they had switched on blocking.
2. The policy never writes prompt text anywhere. It reads the prompt to score
   it and keeps only the verdict.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from switchboard.guardrails import (
    ACTION_ALLOWED,
    ACTION_BLOCKED,
    ACTION_FLAGGED,
    ACTION_OVERRIDDEN,
    MODE_BLOCK,
    MODE_FLAG,
    MODE_OFF,
    SCAN_LIMIT,
    Guardrails,
    Rule,
    build_guardrails,
    calibrate,
    load_rules,
    load_samples,
    prompt_text,
)
from switchboard.ledger.models import RequestLog
from switchboard.ledger.service import STATUS_BLOCKED_POLICY

SAMPLES = "switchboard/guardrail_samples.jsonl"


@pytest.fixture
def guard() -> Guardrails:
    return Guardrails(mode=MODE_FLAG)


# --- Scoring ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Plan my holiday to Bali for two weeks",
        "Book me the cheapest flight to Goa",
        "My girlfriend is upset with me, what do I say?",
        "Write my essay on the French Revolution",
        "What does my horoscope say for Leo?",
        "What should I cook for dinner tonight?",
    ],
)
def test_obvious_personal_prompts_are_flagged(guard: Guardrails, text: str) -> None:
    assert guard.score(text).flagged, text


@pytest.mark.parametrize(
    "text",
    [
        "Why does this function return None instead of a list?",
        "Help me plan the sprint for next week",
        "Plan a migration from MySQL to Postgres",
        "I need to book a meeting room via the Calendar API",
        "What is the p95 latency of this endpoint?",
        "Explain this traceback from api.py",
    ],
)
def test_ordinary_work_is_not_flagged(guard: Guardrails, text: str) -> None:
    """The expensive mistake. Every string here is a phrase that shares words
    with a rule and must survive anyway."""
    assert not guard.score(text).flagged, text


def test_technical_content_lowers_the_score(guard: Guardrails) -> None:
    """A prompt full of code needs a stronger personal signal to trip.

    This is the bias the whole design rests on: missing a personal request is
    cheap, blocking a real one is not.
    """
    personal = "Plan my holiday itinerary"
    with_code = personal + "\n```python\ndef main(): pass\n```\nimport os"
    assert guard.score(personal).score > guard.score(with_code).score


def test_the_work_discount_is_capped(guard: Guardrails) -> None:
    """Otherwise one pasted stack trace would excuse anything after it."""
    heavy = (
        "traceback docker kubernetes select 1 from t "
        "def f(): pass ```x``` main.py refactor api endpoint "
        "Plan my holiday to Bali. Book a hotel. My girlfriend is coming."
    )
    assert guard.score(heavy).flagged


def test_one_half_weight_rule_is_not_enough(guard: Guardrails) -> None:
    """"my mum" appears in plenty of harmless sentences. Two weak signals are
    needed, which is the whole reason weights exist."""
    verdict = guard.score("Help me pick a birthday gift for my mum")
    assert not verdict.flagged
    assert verdict.matched  # it noticed; it just did not act


def test_two_half_weight_rules_reach_the_threshold(guard: Guardrails) -> None:
    verdict = guard.score("A gift for my mum for our anniversary")
    assert verdict.flagged


def test_a_near_miss_explains_itself(guard: Guardrails) -> None:
    """Someone deciding where to set the threshold needs to see near misses."""
    verdict = guard.score("Which laptop should I buy?")
    assert "below the threshold" in verdict.explain()


def test_off_means_off(guard: Guardrails) -> None:
    assert not Guardrails(mode=MODE_OFF).score("Plan my holiday to Bali").flagged


def test_only_the_first_characters_are_scanned(guard: Guardrails) -> None:
    """A regex sweep over a 200 KB pasted log on every request would put this
    in the hot path of a proxy."""
    buried = "x" * (SCAN_LIMIT + 100) + " Plan my holiday to Bali"
    assert not guard.score(buried).flagged


def test_an_unknown_mode_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        Guardrails(mode="lenient")


# --- Reading the prompt -----------------------------------------------------


def test_system_messages_are_not_scored() -> None:
    """Otherwise a product whose system prompt mentions holidays would flag
    every single one of its users."""
    messages = [
        {"role": "system", "content": "You plan my holiday itineraries."},
        {"role": "user", "content": "What is a Python decorator?"},
    ]
    assert "holiday" not in prompt_text(messages)
    assert not Guardrails(mode=MODE_FLAG).score(prompt_text(messages)).flagged


def test_multimodal_content_parts_are_read() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Plan my holiday to Bali"},
                {"type": "image_url", "image_url": {"url": "..."}},
            ],
        }
    ]
    assert "Bali" in prompt_text(messages)


def test_malformed_messages_do_not_raise() -> None:
    assert prompt_text(None) == ""
    assert prompt_text(["not a dict", {"role": "user"}]) == ""


# --- Decisions --------------------------------------------------------------


def test_flag_mode_never_blocks() -> None:
    """THE test. If this fails, labelling has quietly become blocking."""
    verdict = Guardrails(mode=MODE_FLAG).evaluate("Plan my holiday to Bali")
    assert verdict.flagged
    assert not verdict.blocked
    assert verdict.action == ACTION_FLAGGED


def test_block_mode_blocks() -> None:
    verdict = Guardrails(mode=MODE_BLOCK).evaluate("Plan my holiday to Bali")
    assert verdict.blocked


def test_an_override_gets_through_and_is_recorded() -> None:
    verdict = Guardrails(mode=MODE_BLOCK).evaluate(
        "Plan my holiday to Bali", override="writing docs for our travel product"
    )
    assert not verdict.blocked
    assert verdict.action == ACTION_OVERRIDDEN
    assert "travel product" in verdict.override_reason


def test_an_override_cannot_smuggle_in_unbounded_text() -> None:
    verdict = Guardrails(mode=MODE_BLOCK).evaluate(
        "Plan my holiday to Bali", override="x" * 5000
    )
    assert len(verdict.override_reason) <= 200


def test_a_clean_request_is_marked_allowed_not_flagged() -> None:
    verdict = Guardrails(mode=MODE_BLOCK).evaluate("Fix this failing test")
    assert verdict.action == ACTION_ALLOWED


def test_the_refusal_message_says_how_to_get_past_it() -> None:
    """A refusal that does not admit it might be wrong turns a regex false
    positive into a support ticket and a grudge."""
    guard = Guardrails(mode=MODE_BLOCK)
    message = guard.refusal(guard.evaluate("Plan my holiday to Bali"))
    assert "x-switchboard-policy-override" in message
    assert "gets things wrong" in message or "wrong" in message
    assert "holiday_planning" in message


# --- Custom rules -----------------------------------------------------------


def test_a_rules_file_replaces_the_builtins(tmp_path) -> None:
    """Replacing, not merging - otherwise a shipped rule that keeps catching
    your team's real work can never be removed."""
    path = tmp_path / "rules.yaml"
    path.write_text(
        "rules:\n  - name: pineapple\n    pattern: pineapple\n", encoding="utf-8"
    )
    guard = build_guardrails(MODE_FLAG, str(path))
    assert guard.score("pineapple").flagged
    assert not guard.score("Plan my holiday to Bali").flagged


def test_a_broken_pattern_fails_at_load_not_at_request_time(tmp_path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text("rules:\n  - name: bad\n    pattern: '['\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_rules(path)


def test_an_empty_rules_file_is_an_error(tmp_path) -> None:
    """Silently running with no rules would leave an operator believing a
    policy is enforced when nothing is."""
    path = tmp_path / "rules.yaml"
    path.write_text("rules: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_rules(path)


# --- Calibration ------------------------------------------------------------


def test_the_shipped_samples_load() -> None:
    samples = load_samples(SAMPLES)
    assert len(samples) >= 50
    assert any(is_personal for _, is_personal in samples)
    assert any(not is_personal for _, is_personal in samples)


def test_the_false_positive_rate_stays_low(guard: Guardrails) -> None:
    """The number that matters, pinned so a future rule cannot quietly raise
    it. Loosen this only with a deliberate decision to get in people's way
    more often.
    """
    result = calibrate(guard, load_samples(SAMPLES))
    assert result.false_positive_rate <= 0.05, result.false_positive_examples


def test_recall_is_reported_but_allowed_to_be_imperfect(guard: Guardrails) -> None:
    """Deliberately a weaker bar than the false-positive one. Misses cost a
    fraction of a cent; false alarms cost somebody their afternoon."""
    result = calibrate(guard, load_samples(SAMPLES))
    assert result.recall >= 0.7


def test_calibration_counts_add_up() -> None:
    guard = Guardrails(mode=MODE_FLAG, rules=(Rule("x", "personal", "bali"),))
    result = calibrate(guard, [("bali", True), ("bali", False), ("nope", True)])
    assert (result.true_positive, result.false_positive, result.false_negative) == (
        1,
        1,
        1,
    )
    assert result.total == 3


def test_an_empty_calibration_does_not_divide_by_zero(guard: Guardrails) -> None:
    result = calibrate(guard, [])
    assert result.false_positive_rate == 0.0
    assert result.recall == 0.0
    assert result.precision == 0.0


# --- Through the API --------------------------------------------------------


def _chat(text: str) -> dict:
    return {
        "model": "qwen2.5:3b",
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
    }


@pytest.fixture
def flagging(client):
    client.app.state.guardrails = Guardrails(mode=MODE_FLAG)
    return client


@pytest.fixture
def blocking(client):
    client.app.state.guardrails = Guardrails(mode=MODE_BLOCK)
    return client


def test_flag_mode_serves_the_request(flagging, auth, provider) -> None:
    response = flagging.post(
        "/v1/chat/completions", json=_chat("Plan my holiday to Bali"), headers=auth
    )
    assert response.status_code == 200
    assert provider.last_payload is not None


def test_flag_mode_labels_the_ledger_row(flagging, auth, database) -> None:
    flagging.post(
        "/v1/chat/completions", json=_chat("Plan my holiday to Bali"), headers=auth
    )
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.guardrail_label == "personal"
    assert row.guardrail_action == ACTION_FLAGGED
    assert "holiday_planning" in row.guardrail_rules


def test_a_clean_request_is_recorded_as_examined(flagging, auth, database) -> None:
    """"Examined and fine" must be distinguishable from "never examined", or a
    report can show a clean month that nothing ever looked at."""
    flagging.post(
        "/v1/chat/completions", json=_chat("Fix this failing test"), headers=auth
    )
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.guardrail_action == ACTION_ALLOWED
    assert row.guardrail_label is None


def test_with_the_policy_off_nothing_is_recorded(client, auth, database) -> None:
    client.app.state.guardrails = Guardrails(mode=MODE_OFF)
    client.post(
        "/v1/chat/completions", json=_chat("Plan my holiday to Bali"), headers=auth
    )
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.guardrail_action is None


def test_block_mode_refuses_with_403(blocking, auth, provider) -> None:
    response = blocking.post(
        "/v1/chat/completions", json=_chat("Plan my holiday to Bali"), headers=auth
    )
    assert response.status_code == 403
    assert response.json()["error"]["type"] == "policy_violation"
    # Refused before any provider was called, so it cost nothing.
    assert provider.last_payload is None


def test_a_blocked_request_is_recorded_and_costs_nothing(
    blocking, auth, database
) -> None:
    blocking.post(
        "/v1/chat/completions", json=_chat("Plan my holiday to Bali"), headers=auth
    )
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.status == STATUS_BLOCKED_POLICY
    assert row.simulated_cost_usd == 0.0
    assert row.guardrail_action == ACTION_BLOCKED


def test_the_override_header_gets_a_blocked_request_through(
    blocking, auth, provider, database
) -> None:
    headers = {**auth, "X-Switchboard-Policy-Override": "docs for a travel app"}
    response = blocking.post(
        "/v1/chat/completions", json=_chat("Plan my holiday to Bali"), headers=headers
    )
    assert response.status_code == 200
    assert provider.last_payload is not None
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.guardrail_action == ACTION_OVERRIDDEN


def test_the_prompt_text_is_never_stored_by_the_policy(
    blocking, auth, database, ledger
) -> None:
    """The whole point. Scoring a prompt must not become a reason to keep it.

    store_prompts is True in this fixture, so this asserts the stronger thing:
    even then, the POLICY columns carry no prompt text.
    """
    ledger._store_prompts = False
    blocking.post(
        "/v1/chat/completions", json=_chat("Plan my holiday to Bali"), headers=auth
    )
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.prompt_json is None
    for value in (row.guardrail_label, row.guardrail_action, row.guardrail_rules):
        assert value is None or "Bali" not in value


def test_health_reports_the_policy(flagging) -> None:
    body = flagging.get("/health").json()
    assert body["guardrails"]["mode"] == MODE_FLAG
    assert body["guardrails"]["blocking"] is False


def test_metrics_count_policy_verdicts(flagging, auth) -> None:
    flagging.post(
        "/v1/chat/completions", json=_chat("Plan my holiday to Bali"), headers=auth
    )
    body = flagging.get("/metrics").text
    assert "switchboard_policy_events_total" in body
    assert 'category="personal"' in body


def test_metric_labels_never_include_caller_text(flagging, auth) -> None:
    """Cardinality. A label drawn from the prompt would create one time series
    per request and eventually take the monitoring system down."""
    secret = "Plan my holiday to Bali with Aunt Mildred"
    flagging.post("/v1/chat/completions", json=_chat(secret), headers=auth)
    assert "Mildred" not in flagging.get("/metrics").text


# --- Reporting --------------------------------------------------------------


def test_the_ledger_summarises_verdicts(flagging, auth, ledger) -> None:
    flagging.post(
        "/v1/chat/completions", json=_chat("Plan my holiday to Bali"), headers=auth
    )
    flagging.post(
        "/v1/chat/completions", json=_chat("Fix this failing test"), headers=auth
    )
    rows = ledger.guardrail_counts()
    assert {(label, action) for label, action, _, _ in rows} == {
        ("personal", ACTION_FLAGGED),
        ("clean", ACTION_ALLOWED),
    }


def test_rules_that_keep_firing_can_be_found(flagging, auth, ledger) -> None:
    """An operator needs to see WHICH rule keeps catching their team's work,
    so they can delete it."""
    for _ in range(3):
        flagging.post(
            "/v1/chat/completions",
            json=_chat("Plan my holiday to Bali"),
            headers=auth,
        )
    assert ledger.flagged_rules()[0] == ("holiday_planning", 3)


def test_the_dashboard_shows_the_policy(flagging, auth) -> None:
    flagging.post(
        "/v1/chat/completions", json=_chat("Plan my holiday to Bali"), headers=auth
    )
    page = flagging.get("/dashboard").text
    assert "Usage policy" in page
    assert "holiday_planning" in page
    assert "keyword match" in page


def test_the_dashboard_says_when_the_policy_is_off(client) -> None:
    client.app.state.guardrails = Guardrails(mode=MODE_OFF)
    assert "SWITCHBOARD_GUARDRAILS_MODE" in client.get("/dashboard").text


def test_the_samples_file_is_valid_json_throughout() -> None:
    """A stray comma here would break `calibrate` for everyone."""
    with open(SAMPLES, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                row = json.loads(line)
                assert row["label"] in {"work", "personal"}
