"""Retrying provider calls that failed for a reason worth retrying.

Network calls fail. A connection resets, a provider returns 503 while it
restarts, a rate limit trips. Without retries every one of those is a failed
request for a user, when waiting half a second would have worked.

The care is in deciding WHAT to retry. Retrying the wrong thing is worse than
not retrying at all:

* A 400 "your request is malformed" will be malformed the second time too.
  Retrying wastes time and, on a paid provider, money.
* A 401 will still be unauthorised.
* A 429 "slow down" retried immediately makes the overload worse. That is why
  the wait grows after each attempt, and why the provider's own Retry-After
  header wins over our guess.

So only transient failures are retried, the wait grows each time, and a random
jitter is added so that a hundred clients recovering from the same outage do
not all retry in the same instant and knock the provider over again.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

#: Status codes worth trying again. Everything else is the caller's problem or
#: a genuine refusal, and repeating it would only waste time.
RETRYABLE_STATUS = frozenset(
    {
        408,  # request timeout
        409,  # conflict, often transient on a busy provider
        429,  # rate limited
        500,  # internal error
        502,  # bad gateway
        503,  # service unavailable
        504,  # gateway timeout
    }
)

#: Cap on how long to honour a provider's Retry-After. A provider asking us to
#: wait ten minutes should not hold a user's request open that long.
MAX_RETRY_AFTER_S = 30.0


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to try, and how long to wait between attempts."""

    attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    #: Fraction of the delay to randomise. Without jitter, every client that
    #: failed at the same moment retries at the same moment.
    jitter: float = 0.25

    @property
    def enabled(self) -> bool:
        return self.attempts > 1

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait before attempt number `attempt` (1-based).

        The provider knows better than we do, so an explicit Retry-After wins -
        clamped, so a very long one cannot hold the request open indefinitely.
        """
        if retry_after is not None:
            return min(max(retry_after, 0.0), MAX_RETRY_AFTER_S)

        # Exponential: each failure doubles the wait, up to a ceiling.
        delay = min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
        spread = delay * self.jitter
        return max(0.0, delay + random.uniform(-spread, spread))


def parse_retry_after(response: httpx.Response) -> float | None:
    """Seconds from a Retry-After header, when it is given as a number.

    The header may also carry an HTTP date. That form is rare from LLM
    providers and parsing it wrong would be worse than falling back to our own
    backoff, so only the numeric form is honoured.
    """
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return None


def should_retry_response(response: httpx.Response) -> bool:
    return response.status_code in RETRYABLE_STATUS


def is_transient(exc: Exception) -> bool:
    """Network-level failures that a second attempt might survive."""
    return isinstance(
        exc, httpx.ConnectError | httpx.ReadTimeout | httpx.WriteTimeout
        | httpx.ConnectTimeout | httpx.RemoteProtocolError | httpx.PoolTimeout
    )


async def with_retries(
    call,
    policy: RetryPolicy,
    describe: str = "provider call",
):
    """Run `call`, retrying transient failures according to `policy`.

    Returns whatever `call` returns. The final failure is raised or returned
    unchanged, so the caller sees the real error rather than a wrapper.
    """
    last_error: Exception | None = None

    for attempt in range(1, policy.attempts + 1):
        try:
            response = await call()
        except Exception as exc:  # noqa: BLE001 - re-raised below if not transient
            if not is_transient(exc) or attempt == policy.attempts:
                raise
            last_error = exc
            delay = policy.delay_for(attempt)
            logger.warning(
                "%s failed (%s); retrying in %.2fs [attempt %d/%d]",
                describe, exc, delay, attempt, policy.attempts,
            )
            await asyncio.sleep(delay)
            continue

        if not should_retry_response(response) or attempt == policy.attempts:
            return response

        delay = policy.delay_for(attempt, parse_retry_after(response))
        logger.warning(
            "%s returned %d; retrying in %.2fs [attempt %d/%d]",
            describe, response.status_code, delay, attempt, policy.attempts,
        )
        await asyncio.sleep(delay)

    # Unreachable: the loop either returns or raises on its last attempt.
    raise last_error or RuntimeError(f"{describe} exhausted retries")
