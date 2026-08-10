"""Shared fixtures.

Every test runs against an in-memory SQLite database and a stubbed provider, so
the suite needs neither Ollama running nor a file on disk.
"""

from __future__ import annotations

import json

import httpx
import pytest

from switchboard.ledger import Database, LedgerService
from switchboard.ledger.service import STATUS_OK
from switchboard.pricing import PriceTable
from switchboard.providers.ollama import ProviderUnavailable


@pytest.fixture
def prices() -> PriceTable:
    return PriceTable.load()


@pytest.fixture
def database() -> Database:
    db = Database("sqlite://")
    db.create_all()
    yield db
    db.dispose()


@pytest.fixture
def ledger(database: Database, prices: PriceTable) -> LedgerService:
    return LedgerService(database, prices, store_prompts=True)


class StubProvider:
    """Stands in for Ollama. Records what the proxy forwarded upstream."""

    def __init__(self) -> None:
        self.healthy = True
        self.last_payload: dict | None = None
        self.prompt_tokens = 1000
        self.completion_tokens = 500
        self.include_usage = True

    async def aclose(self) -> None:
        pass

    async def is_healthy(self) -> bool:
        return self.healthy

    def _usage(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    async def chat_completion(self, payload: dict) -> httpx.Response:
        self.last_payload = payload
        if not self.healthy:
            raise ProviderUnavailable("stub is down")
        body: dict = {
            "id": "chatcmpl-stub",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        }
        if self.include_usage:
            body["usage"] = self._usage()
        return httpx.Response(200, content=json.dumps(body).encode())

    async def stream_chat_completion(self, payload: dict):
        self.last_payload = payload
        if not self.healthy:
            raise ProviderUnavailable("stub is down")
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        if self.include_usage:
            yield (
                b'data: {"choices":[],"usage":'
                + json.dumps(self._usage()).encode()
                + b"}\n\n"
            )
        yield b"data: [DONE]\n\n"

    async def list_models(self) -> httpx.Response:
        if not self.healthy:
            raise ProviderUnavailable("stub is down")
        return httpx.Response(
            200, content=json.dumps({"data": [{"id": "qwen2.5:3b"}]}).encode()
        )


@pytest.fixture
def provider() -> StubProvider:
    return StubProvider()


@pytest.fixture
def client(database: Database, ledger: LedgerService, provider: StubProvider):
    """TestClient with lifespan bypassed.

    Entering TestClient as a context manager would run the real lifespan and
    build a live Ollama client plus an on-disk database. These tests must pass
    with Ollama stopped and must not touch the filesystem.
    """
    from fastapi.testclient import TestClient

    from switchboard import api

    api.app.state.provider = provider
    api.app.state.database = database
    api.app.state.ledger = ledger
    return TestClient(api.app)


@pytest.fixture
def alice(ledger: LedgerService):
    """A user with a generous budget, plus their raw API key."""
    return ledger.create_user("alice", monthly_budget_usd=50.0)


@pytest.fixture
def auth(alice):
    return {"Authorization": f"Bearer {alice.api_key}"}


__all__ = ["STATUS_OK", "StubProvider"]
