"""Response caching, and the rules about what must never be cached."""

from __future__ import annotations

import time

import pytest

from switchboard.cache import (
    CachedResponse,
    ResponseCache,
    cache_key,
    is_cacheable,
)


def payload(**overrides) -> dict:
    return {"messages": [{"role": "user", "content": "hello"}], **overrides}


# --- What may be cached -----------------------------------------------------


def test_a_plain_deterministic_request_is_cacheable() -> None:
    ok, _ = is_cacheable(payload())
    assert ok


def test_a_random_request_is_never_cached() -> None:
    """Temperature above zero means the caller WANTS variety.

    Handing back a stored answer would silently defeat that, and the caller
    would have no way to tell.
    """
    ok, why = is_cacheable(payload(temperature=0.8))
    assert not ok
    assert "temperature" in why


def test_temperature_zero_is_still_cacheable() -> None:
    ok, _ = is_cacheable(payload(temperature=0.0))
    assert ok


def test_streaming_is_not_cached() -> None:
    ok, why = is_cacheable(payload(stream=True))
    assert not ok
    assert why == "streaming"


def test_multiple_completions_are_not_cached() -> None:
    """`n>1` asks for several different answers, not one repeated."""
    ok, why = is_cacheable(payload(n=3))
    assert not ok
    assert "n>1" in why


def test_an_empty_request_is_not_cached() -> None:
    ok, _ = is_cacheable({"messages": []})
    assert not ok


# --- The key ----------------------------------------------------------------


def test_identical_requests_share_a_key() -> None:
    assert cache_key(payload(), "m") == cache_key(payload(), "m")


def test_a_different_question_is_a_different_key() -> None:
    other = {"messages": [{"role": "user", "content": "goodbye"}]}
    assert cache_key(payload(), "m") != cache_key(other, "m")


def test_a_different_model_is_a_different_key() -> None:
    """Two models answer the same question differently."""
    assert cache_key(payload(), "small") != cache_key(payload(), "big")


def test_sampling_options_change_the_key() -> None:
    """They change the answer, so they must change the key."""
    assert cache_key(payload(), "m") != cache_key(payload(max_tokens=50), "m")
    assert cache_key(payload(seed=1), "m") != cache_key(payload(seed=2), "m")


def test_key_order_does_not_matter() -> None:
    """The same request written in a different order is the same request."""
    a = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    b = {"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
    assert cache_key(a, "m") == cache_key(b, "m")


def test_the_key_ignores_who_is_asking() -> None:
    """Two developers asking the same question should share one answer.

    That sharing IS the saving. The key is built only from the request, so
    nothing about the caller can leak into it either way.
    """
    key = cache_key(payload(), "m")
    assert key == cache_key(dict(payload()), "m")


# --- Storage behaviour ------------------------------------------------------


def test_a_stored_response_comes_back() -> None:
    cache = ResponseCache()
    cache.put("k", b'{"answer": 1}', 200, prompt_tokens=10, completion_tokens=5)
    hit = cache.get("k")
    assert hit.body == b'{"answer": 1}'
    assert hit.prompt_tokens == 10


def test_a_missing_key_is_a_miss() -> None:
    assert ResponseCache().get("nothing here") is None


def test_errors_are_never_cached() -> None:
    """Caching a 500 would keep returning it after the provider recovered."""
    cache = ResponseCache()
    cache.put("k", b"boom", 500)
    assert cache.get("k") is None


def test_entries_expire() -> None:
    """A three-week-old answer may not be what that model says today.

    The margin between the TTL and the sleep is deliberately wide. An earlier
    version used 0.05s and 0.06s, and failed intermittently when the rest of
    the suite was competing for the CPU - a test that fails one run in twenty
    is a test people learn to ignore.
    """
    cache = ResponseCache(ttl_s=0.05)
    cache.put("k", b"x", 200)
    assert cache.get("k") is not None
    time.sleep(0.25)
    assert cache.get("k") is None
    assert cache.stats.expirations == 1


def test_the_cache_is_size_bounded() -> None:
    """Unbounded growth would eventually exhaust memory."""
    cache = ResponseCache(max_entries=3)
    for i in range(5):
        cache.put(f"k{i}", b"x", 200)
    assert len(cache) == 3
    assert cache.stats.evictions == 2


def test_eviction_drops_the_least_recently_used() -> None:
    cache = ResponseCache(max_entries=2)
    cache.put("a", b"1", 200)
    cache.put("b", b"2", 200)
    cache.get("a")          # `a` is now the most recently used
    cache.put("c", b"3", 200)

    assert cache.get("a") is not None
    assert cache.get("b") is None   # evicted
    assert cache.get("c") is not None


def test_a_cache_can_be_switched_off() -> None:
    cache = ResponseCache(max_entries=0)
    cache.put("k", b"x", 200)
    assert cache.enabled is False
    assert cache.get("k") is None


def test_statistics_are_tracked() -> None:
    cache = ResponseCache()
    cache.put("k", b"x", 200)
    cache.get("k")
    cache.get("absent")

    stats = cache.stats
    assert (stats.hits, stats.misses, stats.stores) == (1, 1, 1)
    assert stats.hit_rate == pytest.approx(0.5)


def test_hit_rate_with_no_lookups_is_zero() -> None:
    assert ResponseCache().stats.hit_rate == 0.0


def test_age_is_reported() -> None:
    entry = CachedResponse(b"x", 200, time.monotonic(), 0, 0)
    assert entry.age_s() >= 0.0


# --- End to end through the API --------------------------------------------


def _chat(content: str = "what is 2+2?") -> dict:
    return {
        "model": "auto",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }


def test_a_repeated_request_is_served_from_cache(client, auth, provider) -> None:
    from switchboard.cache import ResponseCache

    client.app.state.cache = ResponseCache()

    first = client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert first.headers["X-Switchboard-Cache"] == "miss"

    calls_after_first = provider.last_payload
    second = client.post("/v1/chat/completions", json=_chat(), headers=auth)

    assert second.headers["X-Switchboard-Cache"] == "hit"
    assert second.content == first.content
    # The provider was not asked again.
    assert provider.last_payload is calls_after_first


def test_a_cache_hit_costs_nothing(client, auth, ledger) -> None:
    """A hit billed at full price would inflate every savings figure."""
    from sqlalchemy import select

    from switchboard.cache import ResponseCache
    from switchboard.ledger.models import RequestLog

    client.app.state.cache = ResponseCache()
    client.post("/v1/chat/completions", json=_chat(), headers=auth)
    client.post("/v1/chat/completions", json=_chat(), headers=auth)

    with client.app.state.database.session() as session:
        rows = session.scalars(select(RequestLog)).all()

    cached = [r for r in rows if r.status == "cached"]
    assert len(cached) == 1
    assert cached[0].simulated_cost_usd == 0.0
    # The baseline is still recorded: that is exactly the saving.
    assert cached[0].baseline_cost_usd > 0


def test_a_different_question_is_not_served_from_cache(client, auth) -> None:
    from switchboard.cache import ResponseCache

    client.app.state.cache = ResponseCache()
    client.post("/v1/chat/completions", json=_chat("first"), headers=auth)
    second = client.post("/v1/chat/completions", json=_chat("second"), headers=auth)
    assert second.headers["X-Switchboard-Cache"] == "miss"


def test_a_random_request_bypasses_the_cache_entirely(client, auth) -> None:
    from switchboard.cache import ResponseCache

    client.app.state.cache = ResponseCache()
    body = _chat() | {"temperature": 0.9}
    first = client.post("/v1/chat/completions", json=body, headers=auth)
    second = client.post("/v1/chat/completions", json=body, headers=auth)

    assert first.headers["X-Switchboard-Cache"] == "skip"
    assert second.headers["X-Switchboard-Cache"] == "skip"


def test_cache_state_is_reported_in_health(client, auth) -> None:
    from switchboard.cache import ResponseCache

    client.app.state.cache = ResponseCache()
    client.post("/v1/chat/completions", json=_chat(), headers=auth)

    cache = client.get("/health").json()["cache"]
    assert cache["enabled"] is True
    assert cache["stores"] == 1


def test_cache_savings_appear_in_the_usage_report(client, auth, ledger) -> None:
    """A cache hit is a request the user made, and it cost nothing.

    Excluding hits from the count would let someone make a thousand requests
    and see "1". Including them at zero cost is what makes the cache's saving
    visible in the same place as the router's.
    """
    from switchboard.cache import ResponseCache

    client.app.state.cache = ResponseCache()
    for _ in range(3):
        client.post("/v1/chat/completions", json=_chat(), headers=auth)

    (row,) = ledger.usage()
    assert row.requests == 3           # all three happened
    assert row.baseline_usd > 0        # all three would have cost something
    # Only one was actually bought, so the saving is large.
    assert row.saved_pct > 60
