"""Stop hammering a provider that is clearly down.

Retries handle a blip. A circuit breaker handles an outage.

Without one, every request to a dead provider waits for its full timeout
before failing over. If the timeout is 60 seconds and the provider has been
down for an hour, every single request in that hour pays 60 seconds of waiting
for an answer that was never coming. The user sees a slow service, the provider
sees a flood of pointless traffic, and neither gets better.

A breaker remembers. After a few consecutive failures it stops trying for a
while and fails over immediately. Once the cooldown passes it lets ONE request
through as a test. If that works, normal service resumes; if not, the cooldown
starts again.

The three states are the standard ones:

    closed     normal. Requests flow. Failures are counted.
    open       too many failures. Requests are refused instantly.
    half-open  cooldown elapsed. One trial request is allowed through.

"Closed" meaning "working" is confusing until you picture an electrical
circuit: a closed circuit conducts, an open one is broken. The names come from
that, and every other implementation uses them, so they are kept.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _Circuit:
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0
    trips: int = 0


@dataclass
class CircuitBreaker:
    """Per-provider failure tracking.

    Only CONSECUTIVE failures count. A provider that fails once an hour is
    working; a provider that fails five times in a row is not. Any success
    resets the count, so intermittent errors never trip it.
    """

    failure_threshold: int = 5
    cooldown_s: float = 30.0
    _circuits: dict[str, _Circuit] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def enabled(self) -> bool:
        return self.failure_threshold > 0

    def _circuit(self, provider_id: str) -> _Circuit:
        return self._circuits.setdefault(provider_id, _Circuit())

    def allows(self, provider_id: str, now: float | None = None) -> bool:
        """May we send a request to this provider right now?"""
        if not self.enabled:
            return True

        now = now if now is not None else time.monotonic()
        with self._lock:
            circuit = self._circuit(provider_id)

            if circuit.state is BreakerState.OPEN:
                if now - circuit.opened_at < self.cooldown_s:
                    return False
                # Cooldown done. Let exactly one request through to find out
                # whether the provider has recovered.
                circuit.state = BreakerState.HALF_OPEN
                logger.info("Circuit for %r is half-open; trying once.", provider_id)

            return True

    def record_success(self, provider_id: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            circuit = self._circuit(provider_id)
            if circuit.state is not BreakerState.CLOSED:
                logger.info("Circuit for %r closed; provider recovered.", provider_id)
            circuit.state = BreakerState.CLOSED
            circuit.consecutive_failures = 0

    def record_failure(self, provider_id: str, now: float | None = None) -> None:
        if not self.enabled:
            return

        now = now if now is not None else time.monotonic()
        with self._lock:
            circuit = self._circuit(provider_id)

            # A failed trial request means it is still down. Straight back to
            # open, without waiting for the threshold again.
            if circuit.state is BreakerState.HALF_OPEN:
                circuit.state = BreakerState.OPEN
                circuit.opened_at = now
                circuit.trips += 1
                logger.warning("Circuit for %r re-opened; still failing.", provider_id)
                return

            circuit.consecutive_failures += 1
            if circuit.consecutive_failures >= self.failure_threshold:
                circuit.state = BreakerState.OPEN
                circuit.opened_at = now
                circuit.trips += 1
                logger.warning(
                    "Circuit for %r opened after %d consecutive failures; "
                    "skipping it for %.0fs.",
                    provider_id,
                    circuit.consecutive_failures,
                    self.cooldown_s,
                )

    def state_of(self, provider_id: str) -> BreakerState:
        with self._lock:
            return self._circuit(provider_id).state

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {
                provider_id: {
                    "state": circuit.state.value,
                    "consecutive_failures": circuit.consecutive_failures,
                    "trips": circuit.trips,
                }
                for provider_id, circuit in self._circuits.items()
            }

    def reset(self) -> None:
        with self._lock:
            self._circuits.clear()
