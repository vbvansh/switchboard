"""Response caching: never pay twice for the same question.

The single biggest saving available to a gateway, and the simplest. If two
requests are byte-for-byte identical, the answer already computed is the answer
that would be computed again, so it is returned from memory for nothing.

Four decisions shape this, and each one is a way it could go wrong:

**Only identical requests hit.** The key covers the model, the messages, and
every sampling option that changes the output. "Similar" questions are NOT
matched. Semantic matching would return an answer to a question nobody asked,
and a cache that is occasionally confidently wrong is worse than no cache.

**Nothing with randomness is cached.** A request at temperature 0.8 is asking
for variety. Serving it a stored answer would silently break that, so anything
that is not deterministic is left alone.

**Cached answers cost nothing, and the ledger says so.** A cache hit that
recorded the full price would inflate every savings figure - the exact
self-flattering error this project keeps guarding against.

**Entries expire.** Models get replaced and prompts get re-tuned; an answer
from three weeks ago may no longer be what that model would say today.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Sampling options that change what the model produces. Any difference in
#: these means a different request, so they are part of the key.
KEYED_OPTIONS = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
    "n",
)

#: Above this temperature the caller is asking for variety, not for the best
#: answer. Serving a stored response would quietly defeat that.
DETERMINISTIC_TEMPERATURE = 0.0


def is_cacheable(payload: dict[str, Any]) -> tuple[bool, str]:
    """Should this request be served from, and stored in, the cache?

    Returns (cacheable, reason). The reason is recorded so an operator can see
    why their hit rate is what it is instead of guessing.
    """
    if payload.get("stream"):
        # Streaming could be cached by replaying stored chunks, but the added
        # machinery is not worth it until the simple case is proven useful.
        return False, "streaming"

    temperature = payload.get("temperature")
    if temperature is not None and float(temperature) > DETERMINISTIC_TEMPERATURE:
        return False, f"temperature={temperature}"

    if payload.get("n") is not None and int(payload["n"]) > 1:
        return False, "n>1"

    if not payload.get("messages"):
        return False, "no messages"

    return True, "cacheable"


def cache_key(payload: dict[str, Any], model: str) -> str:
    """A fingerprint of everything that determines the answer.

    Built from the served model plus the messages and sampling options - and
    deliberately NOT from the caller. Two developers asking the same question
    should share one answer; that is where the saving comes from.

    Nothing else about the request is included, so a key can never be affected
    by who is asking or when.
    """
    material: dict[str, Any] = {
        "model": model,
        "messages": payload.get("messages"),
    }
    for option in KEYED_OPTIONS:
        if option in payload:
            material[option] = payload[option]

    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    expirations: int = 0
    skipped: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "skipped": self.skipped,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass(frozen=True)
class CachedResponse:
    body: bytes
    status_code: int
    stored_at: float
    prompt_tokens: int
    completion_tokens: int

    def age_s(self, now: float | None = None) -> float:
        return (now or time.monotonic()) - self.stored_at


class ResponseCache:
    """A size-bounded, expiring, in-memory cache of completions.

    In-memory on purpose. A shared cache across several Switchboard instances
    would need Redis, another service to run and another thing to go wrong, and
    the saving from a local cache is already most of the available saving. The
    interface is small enough to swap later.

    Thread-safe because the server handles requests concurrently, and a dict
    mutated from two threads at once corrupts silently.
    """

    def __init__(self, max_entries: int = 1000, ttl_s: float = 3600.0) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self.stats = CacheStats()
        self._entries: OrderedDict[str, CachedResponse] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> CachedResponse | None:
        if not self.enabled:
            return None

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.stats.misses += 1
                return None

            if entry.age_s() > self.ttl_s:
                # Stale. Drop it and report a miss - serving a three-week-old
                # answer as if it were fresh is worse than paying again.
                del self._entries[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                return None

            # Recently used entries move to the end, so eviction removes the
            # least recently used first.
            self._entries.move_to_end(key)
            self.stats.hits += 1
            return entry

    def put(
        self,
        key: str,
        body: bytes,
        status_code: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        # Only successful answers are worth repeating. Caching an error would
        # keep returning it long after the provider recovered.
        if not self.enabled or status_code >= 400:
            return

        with self._lock:
            self._entries[key] = CachedResponse(
                body=body,
                status_code=status_code,
                stored_at=time.monotonic(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            self._entries.move_to_end(key)
            self.stats.stores += 1

            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self.stats.evictions += 1

    def skip(self, reason: str) -> None:
        """Record that a request was not eligible, and why."""
        self.stats.skipped += 1
        logger.debug("Cache skipped: %s", reason)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "ttl_s": self.ttl_s,
            **self.stats.as_dict(),
        }
