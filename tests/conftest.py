"""Shared fixtures.

Every test runs against an in-memory SQLite database and stubbed providers, so
the suite needs neither Ollama running nor a file on disk.
"""

from __future__ import annotations

import json

import httpx
import pytest

from switchboard.catalog import ModelCatalog
from switchboard.ledger import Database, LedgerService
from switchboard.ledger.service import STATUS_OK
from switchboard.providers import ProviderUnavailable


@pytest.fixture
def prices() -> ModelCatalog:
    """The real providers.yaml, so tests fail if the shipped catalog breaks."""
    return ModelCatalog.load()


@pytest.fixture
def catalog(prices: ModelCatalog) -> ModelCatalog:
    return prices


@pytest.fixture
def database() -> Database:
    db = Database("sqlite://")
    db.create_all()
    yield db
    db.dispose()


@pytest.fixture
def ledger(database: Database, prices: ModelCatalog) -> LedgerService:
    return LedgerService(database, prices, store_prompts=True)


class StubProvider:
    """Stands in for an upstream. Records what the proxy forwarded."""

    def __init__(self, provider_id: str = "stub") -> None:
        self.id = provider_id
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


class StubPool:
    """A ProviderPool that hands out one stub for every model."""

    def __init__(self, provider: StubProvider, catalog: ModelCatalog) -> None:
        from switchboard.providers.breaker import CircuitBreaker

        self.provider = provider
        self._catalog = catalog
        self.breaker = CircuitBreaker()

    def providers_for(self, model: str) -> list:
        return [] if self._catalog.provider_for(model) is None else [self.provider]

    def for_model(self, model: str):
        if self._catalog.provider_for(model) is None:
            raise ProviderUnavailable(f"No provider declares model {model!r}.")
        return self.provider

    def available_models(self) -> list[str]:
        return self._catalog.known_models()

    def unconfigured(self) -> dict[str, str]:
        return {}

    async def health(self) -> dict[str, bool]:
        return {self.provider.id: self.provider.healthy}

    async def aclose(self) -> None:
        await self.provider.aclose()


@pytest.fixture
def provider() -> StubProvider:
    return StubProvider()


@pytest.fixture
def pool(provider: StubProvider, prices: ModelCatalog) -> StubPool:
    return StubPool(provider, prices)


@pytest.fixture
def client(
    database: Database,
    ledger: LedgerService,
    pool: StubPool,
    prices: ModelCatalog,
):
    """TestClient with lifespan bypassed.

    Entering TestClient as a context manager would run the real lifespan and
    build live HTTP clients plus an on-disk database. These tests must pass
    with no provider running and must not touch the filesystem.
    """
    from fastapi.testclient import TestClient

    from switchboard import api
    from switchboard.cache import ResponseCache
    from switchboard.metrics import build_registry
    from switchboard.ratelimit import RateLimiter

    api.app.state.pool = pool
    api.app.state.catalog = prices
    # A fresh cache per test: entries leaking between tests would make results
    # depend on execution order.
    api.app.state.cache = ResponseCache()
    api.app.state.router = None
    # Generous default so tests are not accidentally rate limited; the
    # rate-limit tests set their own.
    api.app.state.limiter = RateLimiter(default_limit=10_000)
    api.app.state.metrics = build_registry()
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


__all__ = ["STATUS_OK", "StubPool", "StubProvider"]
