"""Failover, circuit breaking, rate limiting and metrics."""

from __future__ import annotations

import json

import httpx
import pytest

from switchboard.catalog import ModelCatalog
from switchboard.metrics import LATENCY_BUCKETS, REQUESTS, Metrics, build_registry
from switchboard.providers.breaker import BreakerState, CircuitBreaker
from switchboard.ratelimit import RateLimiter

# --- Circuit breaker --------------------------------------------------------


def test_a_healthy_provider_is_allowed() -> None:
    assert CircuitBreaker().allows("p") is True


def test_occasional_failures_do_not_trip_it() -> None:
    """A provider that fails once an hour is working."""
    breaker = CircuitBreaker(failure_threshold=3)
    for _ in range(10):
        breaker.record_failure("p")
        breaker.record_success("p")
    assert breaker.allows("p") is True
    assert breaker.state_of("p") is BreakerState.CLOSED


def test_consecutive_failures_trip_it() -> None:
    breaker = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        breaker.record_failure("p")
    assert breaker.state_of("p") is BreakerState.OPEN
    assert breaker.allows("p") is False


def test_a_tripped_provider_recovers_after_the_cooldown() -> None:
    """One trial request gets through to find out whether it is back."""
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=10.0)
    breaker.record_failure("p", now=100.0)

    assert breaker.allows("p", now=105.0) is False   # still cooling down
    assert breaker.allows("p", now=111.0) is True    # trial allowed
    assert breaker.state_of("p") is BreakerState.HALF_OPEN


def test_a_successful_trial_restores_normal_service() -> None:
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=10.0)
    breaker.record_failure("p", now=100.0)
    breaker.allows("p", now=111.0)
    breaker.record_success("p")
    assert breaker.state_of("p") is BreakerState.CLOSED


def test_a_failed_trial_reopens_immediately() -> None:
    """It must not need another five failures to work out it is still down."""
    breaker = CircuitBreaker(failure_threshold=5, cooldown_s=10.0)
    for _ in range(5):
        breaker.record_failure("p", now=100.0)
    breaker.allows("p", now=111.0)                 # half-open
    breaker.record_failure("p", now=112.0)
    assert breaker.state_of("p") is BreakerState.OPEN
    assert breaker.allows("p", now=113.0) is False


def test_providers_are_tracked_separately() -> None:
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure("broken")
    assert breaker.allows("broken") is False
    assert breaker.allows("healthy") is True


def test_the_breaker_can_be_disabled() -> None:
    breaker = CircuitBreaker(failure_threshold=0)
    for _ in range(100):
        breaker.record_failure("p")
    assert breaker.allows("p") is True


def test_snapshot_reports_trips() -> None:
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure("p")
    assert breaker.snapshot()["p"]["trips"] == 1


# --- Failover ordering ------------------------------------------------------


TWO_PROVIDERS = {
    "baseline_model": "shared",
    "ladder": ["shared"],
    "providers": [
        {
            "id": "primary",
            "type": "openai-compatible",
            "base_url": "http://localhost:1111/v1",
            "enabled": True,
            "models": [
                {
                    "id": "shared",
                    "tier": "T0",
                    "input_per_mtok": 1,
                    "output_per_mtok": 1,
                }
            ],
        },
        {
            "id": "backup",
            "type": "openai-compatible",
            "base_url": "http://localhost:2222/v1",
            "enabled": True,
            "models": [
                {
                    "id": "shared",
                    "tier": "T0",
                    "input_per_mtok": 2,
                    "output_per_mtok": 2,
                }
            ],
        },
    ],
}


def test_two_providers_may_serve_the_same_model() -> None:
    """That is what makes failover possible. The catalog used to refuse it."""
    catalog = ModelCatalog.from_dict(TWO_PROVIDERS)
    assert [p.id for p in catalog.providers_for("shared")] == ["primary", "backup"]


def test_the_first_declaration_wins_for_pricing() -> None:
    catalog = ModelCatalog.from_dict(TWO_PROVIDERS)
    assert catalog.models["shared"].provider_id == "primary"
    assert catalog.cost("shared", 1_000_000, 0) == pytest.approx(1.0)


def test_a_tripped_provider_moves_to_the_back_not_out() -> None:
    """If everything is failing, trying a dead provider beats having nowhere
    to send the request."""
    from switchboard.providers import ProviderPool

    catalog = ModelCatalog.from_dict(TWO_PROVIDERS)
    pool = ProviderPool(catalog, breaker=CircuitBreaker(failure_threshold=1))

    assert [p.id for p in pool.providers_for("shared")] == ["primary", "backup"]
    pool.breaker.record_failure("primary")
    assert [p.id for p in pool.providers_for("shared")] == ["backup", "primary"]


# --- Failover in the request path -------------------------------------------


class FlakyProvider:
    """Fails a set number of times, then answers."""

    def __init__(self, provider_id: str, failures: int = 0, status: int = 200) -> None:
        self.id = provider_id
        self.failures = failures
        self.status = status
        self.calls = 0

    async def aclose(self) -> None:
        pass

    async def is_healthy(self) -> bool:
        return True

    async def chat_completion(self, payload: dict) -> httpx.Response:
        from switchboard.providers import ProviderUnavailable

        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderUnavailable(f"{self.id} is down")
        body = {
            "choices": [{"message": {"role": "assistant", "content": self.id}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        return httpx.Response(self.status, content=json.dumps(body).encode())


class FailoverPool:
    def __init__(self, providers: list) -> None:
        self.providers = providers
        self.breaker = CircuitBreaker(failure_threshold=2)

    def providers_for(self, model: str) -> list:
        return list(self.providers)

    def for_model(self, model: str):
        return self.providers[0]

    def available_models(self) -> list[str]:
        return ["qwen2.5:3b"]

    def unconfigured(self) -> dict:
        return {}

    async def health(self) -> dict:
        return {p.id: True for p in self.providers}

    async def aclose(self) -> None:
        pass


def _chat() -> dict:
    return {
        "model": "qwen2.5:3b",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }


def test_a_dead_primary_falls_over_to_the_backup(client, auth) -> None:
    primary = FlakyProvider("primary", failures=99)
    backup = FlakyProvider("backup")
    client.app.state.pool = FailoverPool([primary, backup])

    response = client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "backup"


def test_a_server_error_also_fails_over(client, auth) -> None:
    primary = FlakyProvider("primary", status=503)
    backup = FlakyProvider("backup")
    client.app.state.pool = FailoverPool([primary, backup])

    response = client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert response.json()["choices"][0]["message"]["content"] == "backup"


def test_a_client_error_does_not_fail_over(client, auth) -> None:
    """A 400 is wrong everywhere. Retrying it elsewhere multiplies the cost of
    one bad request by the number of providers configured."""
    primary = FlakyProvider("primary", status=400)
    backup = FlakyProvider("backup")
    client.app.state.pool = FailoverPool([primary, backup])

    client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert backup.calls == 0


def test_every_provider_failing_returns_503(client, auth) -> None:
    pool = FailoverPool(
        [FlakyProvider("a", failures=99), FlakyProvider("b", failures=99)]
    )
    client.app.state.pool = pool

    response = client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert response.status_code == 503


def test_failures_are_recorded_against_the_breaker(client, auth) -> None:
    pool = FailoverPool(
        [FlakyProvider("primary", failures=99), FlakyProvider("backup")]
    )
    client.app.state.pool = pool

    client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert pool.breaker.snapshot()["primary"]["consecutive_failures"] == 1
    assert pool.breaker.state_of("backup") is BreakerState.CLOSED


# --- Rate limiting ----------------------------------------------------------


def test_requests_under_the_limit_are_allowed() -> None:
    limiter = RateLimiter(default_limit=3)
    assert all(limiter.check(1).allowed for _ in range(3))


def test_the_limit_is_enforced() -> None:
    limiter = RateLimiter(default_limit=3)
    for _ in range(3):
        limiter.check(1)
    verdict = limiter.check(1)
    assert verdict.allowed is False
    assert verdict.retry_after_s > 0


def test_users_are_counted_separately() -> None:
    limiter = RateLimiter(default_limit=1)
    assert limiter.check(1).allowed
    assert limiter.check(2).allowed


def test_the_window_slides() -> None:
    """A fixed window would let someone send a full allowance either side of
    the reset - twice the intended rate, within the rules."""
    limiter = RateLimiter(default_limit=2, window_s=60.0)
    limiter.check(1, now=0.0)
    limiter.check(1, now=1.0)
    assert limiter.check(1, now=2.0).allowed is False
    # The first request ages out just after t=60.
    assert limiter.check(1, now=61.0).allowed is True


def test_a_per_user_limit_overrides_the_default() -> None:
    limiter = RateLimiter(default_limit=1)
    assert limiter.check(1, limit=5).allowed
    assert limiter.check(1, limit=5).allowed


def test_rate_limiting_can_be_disabled() -> None:
    limiter = RateLimiter(default_limit=0)
    assert all(limiter.check(1).allowed for _ in range(100))
    assert limiter.enabled is False


def test_headers_tell_a_client_how_to_back_off() -> None:
    limiter = RateLimiter(default_limit=1)
    limiter.check(1)
    headers = limiter.check(1).headers()
    assert headers["X-RateLimit-Limit"] == "1"
    assert headers["X-RateLimit-Remaining"] == "0"
    assert int(headers["Retry-After"]) >= 1  # never 0, which invites a retry storm


def test_a_rate_limited_request_gets_429(client, auth) -> None:
    client.app.state.limiter = RateLimiter(default_limit=2)

    for _ in range(2):
        response = client.post(
            "/v1/chat/completions", json=_chat(), headers=auth
        )
        assert response.status_code == 200

    refused = client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert refused.status_code == 429
    assert refused.json()["error"]["type"] == "rate_limit_exceeded"
    assert "Retry-After" in refused.headers


def test_a_rate_limited_request_never_reaches_the_provider(
    client, auth, provider
) -> None:
    """A limit applied after the call would not protect the provider at all."""
    client.app.state.limiter = RateLimiter(default_limit=1)
    client.post("/v1/chat/completions", json=_chat(), headers=auth)
    before = provider.last_payload

    client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert provider.last_payload is before


# --- Metrics ----------------------------------------------------------------


def test_counters_add_up() -> None:
    metrics = Metrics()
    metrics.increment("m", status="ok")
    metrics.increment("m", status="ok")
    metrics.increment("m", status="error")
    snapshot = metrics.snapshot()["counters"]
    assert snapshot['m{status="ok"}'] == 2
    assert snapshot['m{status="error"}'] == 1


def test_histogram_buckets_are_cumulative() -> None:
    """Prometheus histograms count "at most this", not "exactly this"."""
    metrics = Metrics()
    for value in (0.02, 0.3, 45.0):
        metrics.observe("d", value)
    rendered = metrics.render()
    assert 'd_bucket{le="0.05"} 1' in rendered
    assert 'd_bucket{le="0.5"} 2' in rendered
    assert 'd_bucket{le="+Inf"} 3' in rendered
    assert "d_count 3" in rendered


def test_a_value_beyond_every_bucket_lands_in_infinity() -> None:
    metrics = Metrics()
    metrics.observe("d", LATENCY_BUCKETS[-1] * 10)
    assert 'd_bucket{le="+Inf"} 1' in metrics.render()


def test_rendered_output_declares_types() -> None:
    metrics = build_registry()
    metrics.increment(REQUESTS, status="ok")
    rendered = metrics.render()
    assert f"# TYPE {REQUESTS} counter" in rendered
    assert f"# HELP {REQUESTS}" in rendered


def test_label_values_are_escaped() -> None:
    """An unescaped quote would produce output the scraper cannot parse."""
    metrics = Metrics()
    metrics.increment("m", model='weird"name')
    assert r'model="weird\"name"' in metrics.render()


def test_the_metrics_endpoint_serves_prometheus(client, auth) -> None:
    client.post("/v1/chat/completions", json=_chat(), headers=auth)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert REQUESTS in response.text


def test_metrics_need_no_credentials(client) -> None:
    """A scrape endpoint that needs a key is one nobody configures."""
    assert client.get("/metrics").status_code == 200
