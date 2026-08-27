"""Per-user request rate limiting.

Budgets stop someone spending too much over a month. They do nothing about
someone spending it all in ninety seconds. A runaway retry loop in a script can
burn a month's allowance before anyone notices, and hammer the provider hard
enough to get the whole organisation rate-limited.

The algorithm is a SLIDING WINDOW: count the requests made in the last sixty
seconds, and refuse the next one if that count is at the limit.

The obvious alternative, a fixed window, has a hole worth knowing about. If you
reset the count on the minute, someone can send their whole allowance at 11:59:59
and the whole allowance again at 12:00:00 - twice the intended rate, in one
second, entirely within the rules. A sliding window has no such edge.

This lives in memory, so each Switchboard process counts separately. Two
instances behind a load balancer will together allow twice the limit. Making it
exact needs Redis, and that is a whole extra service to run and monitor for a
guard rail whose job is to catch runaway loops rather than to meter billing.
The limitation is documented rather than papered over.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

WINDOW_S = 60.0


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    #: Seconds until the oldest request leaves the window and a slot frees up.
    retry_after_s: float

    def headers(self) -> dict[str, str]:
        """Standard rate-limit headers, so a client can back off intelligently."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
        }
        if not self.allowed:
            # Rounded up: telling a client to wait 0 seconds invites an
            # immediate retry that will also be refused.
            headers["Retry-After"] = str(max(1, int(self.retry_after_s + 0.999)))
        return headers


class RateLimiter:
    """Counts recent requests per user, in a rolling sixty-second window."""

    def __init__(self, default_limit: int = 60, window_s: float = WINDOW_S) -> None:
        self.default_limit = default_limit
        self.window_s = window_s
        self._events: dict[int, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.default_limit > 0

    def check(
        self, user_id: int, limit: int | None = None, now: float | None = None
    ) -> RateLimitDecision:
        """Record an attempt and say whether it is allowed.

        `limit` overrides the default for this user - a batch job may be
        allowed far more than an interactive tool.
        """
        effective = self.default_limit if limit is None else limit
        if effective <= 0:
            return RateLimitDecision(True, effective, 0, 0.0)

        now = now if now is not None else time.monotonic()
        cutoff = now - self.window_s

        with self._lock:
            events = self._events[user_id]
            while events and events[0] <= cutoff:
                events.popleft()

            if len(events) >= effective:
                # The window frees up when its oldest entry ages out.
                retry_after = events[0] - cutoff
                return RateLimitDecision(False, effective, 0, retry_after)

            events.append(now)
            return RateLimitDecision(
                True, effective, effective - len(events), 0.0
            )

    def used(self, user_id: int, now: float | None = None) -> int:
        now = now if now is not None else time.monotonic()
        cutoff = now - self.window_s
        with self._lock:
            events = self._events.get(user_id)
            return sum(1 for stamp in (events or ()) if stamp > cutoff)

    def reset(self, user_id: int | None = None) -> None:
        with self._lock:
            if user_id is None:
                self._events.clear()
            else:
                self._events.pop(user_id, None)

    def snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "default_limit_per_minute": self.default_limit,
            "window_s": self.window_s,
            "tracked_users": len(self._events),
        }
