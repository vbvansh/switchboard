"""Retrying provider calls - and, more importantly, NOT retrying the wrong ones."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from switchboard.providers.retry import (
    MAX_RETRY_AFTER_S,
    RetryPolicy,
    is_transient,
    parse_retry_after,
    should_retry_response,
    with_retries,
)


def response(status: int, **headers) -> httpx.Response:
    return httpx.Response(status, headers=headers)


# --- What deserves a second attempt -----------------------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_transient_failures_are_retried(status: int) -> None:
    assert should_retry_response(response(status))


@pytest.mark.parametrize("status", [200, 201, 400, 401, 403, 404, 422])
def test_permanent_outcomes_are_not_retried(status: int) -> None:
    """A malformed request will be malformed the second time too.

    Retrying it wastes time and, on a paid provider, money.
    """
    assert not should_retry_response(response(status))


def test_network_errors_are_transient() -> None:
    request = httpx.Request("POST", "http://example")
    assert is_transient(httpx.ConnectError("refused", request=request))
    assert is_transient(httpx.ReadTimeout("slow", request=request))


def test_programming_errors_are_not_transient() -> None:
    """Retrying a bug just runs the bug again."""
    assert not is_transient(ValueError("bad code"))


# --- How long to wait -------------------------------------------------------


def test_the_wait_grows_after_each_failure() -> None:
    """Retrying a rate limit immediately makes the overload worse."""
    policy = RetryPolicy(base_delay_s=1.0, jitter=0.0)
    assert policy.delay_for(1) == pytest.approx(1.0)
    assert policy.delay_for(2) == pytest.approx(2.0)
    assert policy.delay_for(3) == pytest.approx(4.0)


def test_the_wait_has_a_ceiling() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=3.0, jitter=0.0)
    assert policy.delay_for(10) == pytest.approx(3.0)


def test_jitter_spreads_retries_out() -> None:
    """Without it, every client that failed together retries together and
    knocks the provider over again."""
    policy = RetryPolicy(base_delay_s=1.0, jitter=0.5)
    delays = {policy.delay_for(1) for _ in range(20)}
    assert len(delays) > 1


def test_a_delay_is_never_negative() -> None:
    policy = RetryPolicy(base_delay_s=0.01, jitter=5.0)
    assert all(policy.delay_for(1) >= 0 for _ in range(50))


def test_the_provider_knows_best() -> None:
    """An explicit Retry-After beats our own guess."""
    policy = RetryPolicy(base_delay_s=1.0, jitter=0.0)
    assert policy.delay_for(1, retry_after=7.0) == pytest.approx(7.0)


def test_an_absurd_retry_after_is_clamped() -> None:
    """A provider asking for ten minutes must not hold a user's request open."""
    policy = RetryPolicy()
    assert policy.delay_for(1, retry_after=600.0) == MAX_RETRY_AFTER_S


def test_retry_after_is_parsed_when_numeric() -> None:
    assert parse_retry_after(response(429, **{"retry-after": "2.5"})) == 2.5


def test_a_date_style_retry_after_falls_back_to_our_own_backoff() -> None:
    """Parsing it wrong would be worse than not parsing it."""
    header = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
    assert parse_retry_after(response(429, **header)) is None


def test_a_missing_retry_after_is_none() -> None:
    assert parse_retry_after(response(429)) is None


# --- The retry loop ---------------------------------------------------------


def run(coro):
    return asyncio.run(coro)


def test_a_successful_call_is_not_retried() -> None:
    calls = []

    async def call():
        calls.append(1)
        return response(200)

    result = run(with_retries(call, RetryPolicy(base_delay_s=0)))
    assert result.status_code == 200
    assert len(calls) == 1


def test_a_transient_failure_is_retried_until_it_works() -> None:
    calls = []

    async def call():
        calls.append(1)
        return response(200 if len(calls) >= 3 else 503)

    result = run(with_retries(call, RetryPolicy(attempts=5, base_delay_s=0)))
    assert result.status_code == 200
    assert len(calls) == 3


def test_a_permanent_failure_is_returned_immediately() -> None:
    calls = []

    async def call():
        calls.append(1)
        return response(400)

    result = run(with_retries(call, RetryPolicy(attempts=5, base_delay_s=0)))
    assert result.status_code == 400
    assert len(calls) == 1


def test_the_last_failure_is_returned_not_swallowed() -> None:
    """The caller must see the real error, not a wrapper."""

    async def call():
        return response(503)

    result = run(with_retries(call, RetryPolicy(attempts=2, base_delay_s=0)))
    assert result.status_code == 503


def test_a_network_error_is_retried_then_raised() -> None:
    calls = []

    async def call():
        calls.append(1)
        raise httpx.ConnectError("refused", request=httpx.Request("POST", "http://x"))

    with pytest.raises(httpx.ConnectError):
        run(with_retries(call, RetryPolicy(attempts=3, base_delay_s=0)))
    assert len(calls) == 3


def test_a_non_transient_exception_is_raised_at_once() -> None:
    calls = []

    async def call():
        calls.append(1)
        raise ValueError("a bug")

    with pytest.raises(ValueError):
        run(with_retries(call, RetryPolicy(attempts=5, base_delay_s=0)))
    assert len(calls) == 1


def test_retries_can_be_switched_off() -> None:
    calls = []

    async def call():
        calls.append(1)
        return response(503)

    run(with_retries(call, RetryPolicy(attempts=1, base_delay_s=0)))
    assert len(calls) == 1
    assert RetryPolicy(attempts=1).enabled is False
