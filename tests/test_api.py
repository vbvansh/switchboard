"""API behaviour: identity, budgets, accounting, pass-through."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from switchboard.config import settings
from switchboard.ledger import LedgerService
from switchboard.ledger.keys import generate_api_key


def _chat(model: str | None = "auto", **extra) -> dict:
    body: dict = {"messages": [{"role": "user", "content": "hello"}], **extra}
    if model is not None:
        body["model"] = model
    return body


# --- Health ----------------------------------------------------------------


def test_health_needs_no_credentials(client: TestClient) -> None:
    """A health check that requires a key is useless to a monitoring system."""
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["provider_reachable"] is True


# --- Identity --------------------------------------------------------------


def test_request_without_a_key_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json=_chat())
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_request_with_an_unknown_key_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=_chat(),
        headers={"Authorization": f"Bearer {generate_api_key()}"},
    )
    assert response.status_code == 401


def test_valid_key_is_accepted(client: TestClient, auth: dict) -> None:
    response = client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert response.status_code == 200


def test_deactivated_user_is_rejected(
    client: TestClient, auth: dict, ledger: LedgerService
) -> None:
    ledger.set_active("alice", False)
    response = client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert response.status_code == 401


def test_models_endpoint_requires_a_key(client: TestClient, auth: dict) -> None:
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=auth).status_code == 200


# --- Model selection -------------------------------------------------------


def test_auto_resolves_to_the_default_model(
    client: TestClient, auth: dict, provider
) -> None:
    client.post("/v1/chat/completions", json=_chat("auto"), headers=auth)
    assert provider.last_payload["model"] == settings.default_model


def test_missing_model_resolves_to_the_default(
    client: TestClient, auth: dict, provider
) -> None:
    client.post("/v1/chat/completions", json=_chat(None), headers=auth)
    assert provider.last_payload["model"] == settings.default_model


def test_explicit_model_is_honoured(client: TestClient, auth: dict, provider) -> None:
    client.post("/v1/chat/completions", json=_chat("qwen2.5:1.5b"), headers=auth)
    assert provider.last_payload["model"] == "qwen2.5:1.5b"


def test_unknown_fields_pass_through(client: TestClient, auth: dict, provider) -> None:
    client.post(
        "/v1/chat/completions",
        json=_chat("auto", temperature=0.2, some_future_field=True),
        headers=auth,
    )
    assert provider.last_payload["temperature"] == 0.2
    assert provider.last_payload["some_future_field"] is True


# --- Accounting ------------------------------------------------------------


def test_successful_request_is_recorded(
    client: TestClient, auth: dict, ledger: LedgerService
) -> None:
    client.post("/v1/chat/completions", json=_chat(), headers=auth)

    (row,) = ledger.usage()
    assert row.requests == 1
    assert row.spent_usd > 0
    assert row.baseline_usd > row.spent_usd  # default tier is cheaper than baseline


def test_recorded_row_captures_the_details(
    client: TestClient, auth: dict, ledger: LedgerService, database
) -> None:
    from sqlalchemy import select

    from switchboard.ledger.models import RequestLog

    client.post("/v1/chat/completions", json=_chat("auto"), headers=auth)

    with database.session() as session:
        row = session.scalar(select(RequestLog))

    assert row.requested_model == "auto"
    assert row.served_model == settings.default_model
    assert (row.prompt_tokens, row.completion_tokens) == (1000, 500)
    assert row.tokens_estimated is False
    assert row.status == "ok"
    assert "hello" in row.prompt_json


def test_tokens_are_estimated_when_provider_omits_usage(
    client: TestClient, auth: dict, provider, database
) -> None:
    from sqlalchemy import select

    from switchboard.ledger.models import RequestLog

    provider.include_usage = False
    client.post("/v1/chat/completions", json=_chat(), headers=auth)

    with database.session() as session:
        row = session.scalar(select(RequestLog))

    assert row.tokens_estimated is True
    assert row.prompt_tokens > 0


def test_spend_accumulates_across_requests(
    client: TestClient, auth: dict, ledger: LedgerService
) -> None:
    for _ in range(3):
        client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert ledger.usage()[0].requests == 3


# --- Budgets ---------------------------------------------------------------


def test_exhausted_budget_returns_402(
    client: TestClient, auth: dict, ledger: LedgerService, provider
) -> None:
    ledger.set_budget("alice", 0.01)
    provider.prompt_tokens = 1_000_000  # blows the budget in one request

    first = client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert first.status_code == 200  # allowed: spend was zero beforehand

    second = client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert second.status_code == 402
    assert second.json()["error"]["type"] == "insufficient_quota"


def test_blocked_attempts_are_visible_but_not_charged(
    client: TestClient, auth: dict, ledger: LedgerService, provider, database
) -> None:
    from sqlalchemy import select

    from switchboard.ledger.models import RequestLog

    ledger.set_budget("alice", 0.01)
    provider.prompt_tokens = 1_000_000
    client.post("/v1/chat/completions", json=_chat(), headers=auth)
    spend_after_first = ledger.month_to_date_spend(1)

    client.post("/v1/chat/completions", json=_chat(), headers=auth)

    with database.session() as session:
        blocked = session.scalars(
            select(RequestLog).where(RequestLog.status == "blocked_budget")
        ).all()

    assert len(blocked) == 1
    assert blocked[0].simulated_cost_usd == 0.0
    assert ledger.month_to_date_spend(1) == pytest.approx(spend_after_first)


def test_raising_the_budget_restores_service(
    client: TestClient, auth: dict, ledger: LedgerService, provider
) -> None:
    def attempt() -> int:
        return client.post(
            "/v1/chat/completions", json=_chat(), headers=auth
        ).status_code

    ledger.set_budget("alice", 0.01)
    provider.prompt_tokens = 1_000_000
    attempt()
    assert attempt() == 402

    ledger.set_budget("alice", 1000.0)
    assert attempt() == 200


# --- Streaming -------------------------------------------------------------


def test_streaming_returns_sse(client: TestClient, auth: dict) -> None:
    response = client.post(
        "/v1/chat/completions", json=_chat("auto", stream=True), headers=auth
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert b"[DONE]" in response.content


def test_streaming_requests_usage_from_the_provider(
    client: TestClient, auth: dict, provider
) -> None:
    client.post("/v1/chat/completions", json=_chat("auto", stream=True), headers=auth)
    assert provider.last_payload["stream_options"] == {"include_usage": True}


def test_streaming_is_recorded_with_real_token_counts(
    client: TestClient, auth: dict, database
) -> None:
    from sqlalchemy import select

    from switchboard.ledger.models import RequestLog

    client.post("/v1/chat/completions", json=_chat("auto", stream=True), headers=auth)

    with database.session() as session:
        row = session.scalar(select(RequestLog))

    assert (row.prompt_tokens, row.completion_tokens) == (1000, 500)
    assert row.tokens_estimated is False


def test_streaming_falls_back_to_estimates(
    client: TestClient, auth: dict, provider, database
) -> None:
    from sqlalchemy import select

    from switchboard.ledger.models import RequestLog

    provider.include_usage = False
    client.post("/v1/chat/completions", json=_chat("auto", stream=True), headers=auth)

    with database.session() as session:
        row = session.scalar(select(RequestLog))

    assert row.tokens_estimated is True
    assert row.completion_tokens > 0


# --- Failures --------------------------------------------------------------


def test_invalid_json_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400


def test_dead_provider_returns_503(client: TestClient, auth: dict, provider) -> None:
    provider.healthy = False
    response = client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "provider_unavailable"


def test_dead_provider_returns_503_when_streaming(
    client: TestClient, auth: dict, provider
) -> None:
    provider.healthy = False
    response = client.post(
        "/v1/chat/completions", json=_chat("auto", stream=True), headers=auth
    )
    assert response.status_code == 503


def test_provider_failure_is_recorded_but_not_charged(
    client: TestClient, auth: dict, provider, ledger: LedgerService, database
) -> None:
    from sqlalchemy import select

    from switchboard.ledger.models import RequestLog

    provider.healthy = False
    client.post("/v1/chat/completions", json=_chat(), headers=auth)

    with database.session() as session:
        row = session.scalar(select(RequestLog))

    assert row.status == "provider_error"
    assert row.simulated_cost_usd == 0.0
    assert ledger.month_to_date_spend(1) == 0.0
