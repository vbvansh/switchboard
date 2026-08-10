"""API tests with a stubbed provider - no Ollama required to run these."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from switchboard import api
from switchboard.config import settings
from switchboard.providers.ollama import ProviderUnavailable


class StubProvider:
    """Records what the proxy forwarded upstream."""

    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy
        self.last_payload: dict | None = None

    async def aclose(self) -> None:  # pragma: no cover - nothing to release
        pass

    async def is_healthy(self) -> bool:
        return self.healthy

    async def chat_completion(self, payload: dict) -> httpx.Response:
        self.last_payload = payload
        if not self.healthy:
            raise ProviderUnavailable("stub is down")
        return httpx.Response(
            200,
            content=json.dumps(
                {"id": "chatcmpl-stub", "choices": [{"message": {"content": "hi"}}]}
            ).encode(),
        )

    async def stream_chat_completion(self, payload: dict):
        self.last_payload = payload
        if not self.healthy:
            raise ProviderUnavailable("stub is down")
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def list_models(self) -> httpx.Response:
        if not self.healthy:
            raise ProviderUnavailable("stub is down")
        return httpx.Response(
            200, content=json.dumps({"data": [{"id": "qwen2.5:3b"}]}).encode()
        )


@pytest.fixture
def stub() -> StubProvider:
    return StubProvider()


@pytest.fixture
def client(stub: StubProvider) -> TestClient:
    # Deliberately not entering TestClient as a context manager: that would run
    # the lifespan and build a real OllamaProvider. These tests must pass with
    # Ollama stopped.
    api.app.state.provider = stub
    return TestClient(api.app)


def _chat(model: str | None) -> dict:
    body: dict = {"messages": [{"role": "user", "content": "hello"}]}
    if model is not None:
        body["model"] = model
    return body


def test_health_reports_provider_state(client: TestClient) -> None:
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["provider_reachable"] is True


def test_missing_model_falls_back_to_default(
    client: TestClient, stub: StubProvider
) -> None:
    client.post("/v1/chat/completions", json=_chat(None))
    assert stub.last_payload["model"] == settings.default_model


def test_auto_model_resolves_to_default(
    client: TestClient, stub: StubProvider
) -> None:
    client.post("/v1/chat/completions", json=_chat("auto"))
    assert stub.last_payload["model"] == settings.default_model


def test_explicit_model_is_honoured(client: TestClient, stub: StubProvider) -> None:
    client.post("/v1/chat/completions", json=_chat("qwen2.5:1.5b"))
    assert stub.last_payload["model"] == "qwen2.5:1.5b"


def test_unknown_fields_pass_through(client: TestClient, stub: StubProvider) -> None:
    body = _chat("auto") | {"temperature": 0.2, "some_future_field": True}
    client.post("/v1/chat/completions", json=body)
    assert stub.last_payload["temperature"] == 0.2
    assert stub.last_payload["some_future_field"] is True


def test_streaming_returns_sse(client: TestClient) -> None:
    body = _chat("auto") | {"stream": True}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert b"[DONE]" in response.content


def test_invalid_json_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400


def test_dead_provider_returns_503(client: TestClient, stub: StubProvider) -> None:
    stub.healthy = False
    response = client.post("/v1/chat/completions", json=_chat("auto"))
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "provider_unavailable"


def test_dead_provider_returns_503_when_streaming(
    client: TestClient, stub: StubProvider
) -> None:
    stub.healthy = False
    body = _chat("auto") | {"stream": True}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 503
