"""OpenAI-compatible HTTP surface.

Request lifecycle:

    identify (401) -> budget check (402) -> pick model -> find its provider
    -> serve -> measure -> record -> reply

Model choice is still fixed: `_resolve_model` is the seam the router plugs into.
What changed in A.2 is that the chosen model is now looked up in the catalog to
find which provider serves it, instead of everything going to one hardcoded
Ollama client.

The request body is deliberately not schema-validated. Passing it through keeps
compatibility with OpenAI client features this code does not model; only the
`model` field is ever rewritten.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from switchboard.cache import ResponseCache, cache_key, is_cacheable
from switchboard.catalog import ModelCatalog
from switchboard.config import AUTO_MODEL, settings
from switchboard.ledger import (
    STATUS_BLOCKED_BUDGET,
    STATUS_CACHED,
    STATUS_OK,
    STATUS_PROVIDER_ERROR,
    AuthenticationError,
    BudgetExceededError,
    Database,
    LedgerService,
    User,
)
from switchboard.ledger.keys import extract_bearer_token
from switchboard.ledger.service import estimate_tokens
from switchboard.providers import Provider, ProviderPool, ProviderUnavailable
from switchboard.providers.retry import RetryPolicy
from switchboard.routing.base import RoutingContext
from switchboard.routing.live import RequestLimits, build_router
from switchboard.schema import require_up_to_date
from switchboard.streaming import UsageSniffer, request_usage_in_stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Refuse to start against a database shaped differently to what this code
    # expects. Starting anyway does not fail cleanly - it fails later, mid
    # request, with an error pointing somewhere unhelpful.
    require_up_to_date(settings.database_url)

    catalog = ModelCatalog.load(settings.providers_file)
    database = Database(settings.database_url)

    retry = RetryPolicy(
        attempts=settings.retry_attempts,
        base_delay_s=settings.retry_base_delay_s,
    )
    pool = ProviderPool(catalog, local_only=settings.local_only, retry=retry)

    app.state.catalog = catalog
    app.state.pool = pool
    app.state.database = database
    app.state.ledger = LedgerService(database, catalog, settings.store_prompts)
    # A missing or stale router must never stop the server: requests fall back
    # to the default model and /health says routing is off.
    app.state.router = build_router(catalog, pool.available_models())
    app.state.cache = ResponseCache(
        max_entries=settings.cache_max_entries, ttl_s=settings.cache_ttl_s
    )
    try:
        yield
    finally:
        await app.state.pool.aclose()
        database.dispose()


app = FastAPI(
    title="Switchboard",
    description="Self-hostable AI model router with an auditable savings ledger.",
    version="0.3.0",
    lifespan=lifespan,
)


# --- Small helpers ---------------------------------------------------------


def _error(status: int, message: str, kind: str) -> JSONResponse:
    """OpenAI-shaped error body, so existing clients surface it properly."""
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": kind}}
    )


def _identify(request: Request) -> User:
    """Resolve the caller. Raises AuthenticationError."""
    token = extract_bearer_token(request.headers.get("authorization"))
    return request.app.state.ledger.authenticate(token)


def _resolve_model(
    request: Request, payload: dict[str, Any]
) -> tuple[str, str | None]:
    """Decide which model serves this request, and why.

    An explicit model name is always honoured - callers who name a model get
    that model, which is what makes per-model benchmarking possible and what
    keeps the proxy a faithful OpenAI endpoint.

    `auto` hands the choice to the router. With no usable router, that means
    the configured default, and the reason says so rather than pretending a
    decision was made.
    """
    requested = payload.get("model")
    if requested and requested != AUTO_MODEL:
        return str(requested), None

    router = getattr(request.app.state, "router", None)
    if router is None or not router.enabled:
        return settings.default_model, "routing unavailable; used default_model"

    limits = RequestLimits.from_headers(request.headers)
    decision = router.choose(
        RoutingContext(messages=payload.get("messages") or []), limits
    )
    return decision.model, decision.reason


def _usage_from_body(body: bytes) -> tuple[int, int] | None:
    """Token counts from a non-streaming response, if the provider sent them."""
    try:
        usage = json.loads(body).get("usage")
    except (ValueError, AttributeError):
        return None
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if isinstance(prompt, int) and isinstance(completion, int):
        return prompt, completion
    return None


def _prompt_text(messages: Any) -> str:
    """Flatten messages to text, for token estimation only."""
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        message["content"]
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), str)
    )


# --- Endpoints -------------------------------------------------------------


def _routing_status(request: Request) -> dict[str, Any]:
    router = getattr(request.app.state, "router", None)
    if router is None:
        return {"enabled": False, "reason": "no router artifact loaded"}
    if not router.enabled:
        return {
            "enabled": False,
            "reason": (
                "the router was trained for models this catalog cannot serve; "
                "add `benchmark_alias` entries in providers.yaml"
            ),
            "trained_for": router.metadata.models,
        }
    return {
        "enabled": True,
        "models": router.routable_models,
        "trained": router.metadata.describe(),
    }


@app.get("/health/live")
async def liveness() -> dict[str, str]:
    """Is this process alive? Nothing else.

    Deliberately checks no dependency. A failing liveness probe tells an
    orchestrator to kill and restart the container - so if this checked whether
    a provider were up, an outage at that provider would put Switchboard into a
    restart loop, fixing nothing and destroying its own logs. Dependency health
    belongs in /health/ready.
    """
    return {"status": "alive", "version": app.version}


@app.get("/health/ready")
async def readiness(request: Request) -> Response:
    """Can this process serve traffic right now?

    Returns 503 when it cannot, so a load balancer stops sending it requests
    while leaving the container running. Two things must hold: the ledger must
    be writable, and at least one provider must answer.
    """
    pool: ProviderPool = request.app.state.pool
    database: Database = request.app.state.database

    provider_health = await pool.health()
    database_ok = database.is_reachable()
    providers_ok = any(provider_health.values())

    ready = database_ok and providers_ok
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "database": database_ok,
            "providers": provider_health,
        },
    )


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Detailed status for a human. Open by design - a health check that needs
    credentials is useless to a monitoring system."""
    pool: ProviderPool = request.app.state.pool
    catalog: ModelCatalog = request.app.state.catalog
    database: Database = request.app.state.database
    provider_health = await pool.health()

    return {
        "status": "ok" if any(provider_health.values()) else "degraded",
        "version": app.version,
        "database_reachable": database.is_reachable(),
        "providers": provider_health,
        "unconfigured_providers": pool.unconfigured(),
        "available_models": pool.available_models(),
        "default_model": settings.default_model,
        "routing": _routing_status(request),
        "cache": request.app.state.cache.as_dict(),
        "local_only": settings.local_only,
        "store_prompts": settings.store_prompts,
        "simulated_pricing": catalog.has_simulated_pricing,
    }


@app.get("/v1/models")
async def list_models(request: Request) -> Response:
    """Every model this Switchboard can actually serve, OpenAI-shaped.

    Built from the catalog rather than proxied from one upstream, because with
    several providers there is no single upstream to ask.
    """
    try:
        _identify(request)
    except AuthenticationError as exc:
        return _error(401, str(exc), "authentication_error")

    pool: ProviderPool = request.app.state.pool
    catalog: ModelCatalog = request.app.state.catalog

    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": catalog.models[model_id].provider_id,
                    "switchboard": {
                        "tier": catalog.models[model_id].tier,
                        "input_per_mtok": catalog.models[model_id].input_per_mtok,
                        "output_per_mtok": catalog.models[model_id].output_per_mtok,
                        "context_window": catalog.models[model_id].context_window,
                        "simulated_pricing": catalog.models[
                            model_id
                        ].simulated_pricing,
                    },
                }
                for model_id in pool.available_models()
            ],
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    ledger: LedgerService = request.app.state.ledger
    pool: ProviderPool = request.app.state.pool

    try:
        payload: dict[str, Any] = await request.json()
    except ValueError:
        return _error(400, "Request body must be valid JSON.", "invalid_request_error")

    try:
        user = _identify(request)
    except AuthenticationError as exc:
        return _error(401, str(exc), "authentication_error")

    requested_model = str(payload.get("model") or AUTO_MODEL)
    served_model, routing_reason = _resolve_model(request, payload)
    messages = payload.get("messages")

    def record(status: str, **kwargs) -> None:
        ledger.record(
            user_id=user.id,
            requested_model=requested_model,
            served_model=served_model,
            tokens_estimated=False,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0,
            status=status,
            messages=messages,
            routing_reason=routing_reason,
            **kwargs,
        )

    try:
        ledger.assert_within_budget(user)
    except BudgetExceededError as exc:
        # Recorded so a blocked attempt is visible. Costs zero and is excluded
        # from spend totals, so retrying cannot dig a deeper hole.
        record(STATUS_BLOCKED_BUDGET, error_detail=str(exc))
        return _error(402, str(exc), "insufficient_quota")

    try:
        provider = pool.for_model(served_model)
    except ProviderUnavailable as exc:
        record(STATUS_PROVIDER_ERROR, error_detail=str(exc))
        return _error(503, str(exc), "provider_unavailable")

    payload["model"] = served_model
    started = time.perf_counter()

    # A cache hit costs nothing and must be RECORDED as costing nothing. A hit
    # billed at the full price would inflate every savings figure.
    cache: ResponseCache = request.app.state.cache
    cacheable, why = is_cacheable(payload)
    key = cache_key(payload, served_model) if cacheable else None

    if key is not None:
        if (hit := cache.get(key)) is not None:
            ledger.record(
                user_id=user.id,
                requested_model=requested_model,
                served_model=served_model,
                prompt_tokens=hit.prompt_tokens,
                completion_tokens=hit.completion_tokens,
                tokens_estimated=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                status=STATUS_CACHED,
                messages=messages,
                routing_reason=routing_reason,
            )
            return Response(
                content=hit.body,
                status_code=hit.status_code,
                media_type="application/json",
                headers={"X-Switchboard-Cache": "hit"},
            )
    elif not payload.get("stream"):
        cache.skip(why)

    if payload.get("stream"):
        return await _serve_streaming(
            payload=request_usage_in_stream(payload),
            provider=provider,
            ledger=ledger,
            user=user,
            requested_model=requested_model,
            served_model=served_model,
            messages=messages,
            started=started,
            routing_reason=routing_reason,
        )

    try:
        upstream = await provider.chat_completion(payload)
    except ProviderUnavailable as exc:
        record(
            STATUS_PROVIDER_ERROR,
            error_detail=str(exc),
        )
        return _error(503, str(exc), "provider_unavailable")

    latency_ms = int((time.perf_counter() - started) * 1000)
    usage = _usage_from_body(upstream.content)
    if usage is not None:
        prompt_tokens, completion_tokens = usage
        estimated = False
    else:
        prompt_tokens = estimate_tokens(_prompt_text(messages))
        completion_tokens = estimate_tokens(upstream.text)
        estimated = True

    ledger.record(
        user_id=user.id,
        requested_model=requested_model,
        served_model=served_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_estimated=estimated,
        latency_ms=latency_ms,
        status=STATUS_OK if upstream.status_code < 400 else STATUS_PROVIDER_ERROR,
        error_detail=None if upstream.status_code < 400 else upstream.text[:500],
        messages=messages,
        routing_reason=routing_reason,
    )

    if key is not None:
        cache.put(
            key,
            upstream.content,
            upstream.status_code,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type="application/json",
        headers={"X-Switchboard-Cache": "miss" if key is not None else "skip"},
    )


async def _serve_streaming(
    *,
    payload: dict[str, Any],
    provider: Provider,
    ledger: LedgerService,
    user: User,
    requested_model: str,
    served_model: str,
    messages: Any,
    started: float,
    routing_reason: str | None = None,
) -> Response:
    """Stream the answer through untouched, accounting for it as it passes."""
    stream = provider.stream_chat_completion(payload)

    # Pull the first chunk eagerly so a dead provider becomes a clean 503 rather
    # than a truncated stream the client has to interpret.
    try:
        first_chunk = await anext(stream)
    except ProviderUnavailable as exc:
        ledger.record(
            user_id=user.id,
            requested_model=requested_model,
            served_model=served_model,
            prompt_tokens=0,
            completion_tokens=0,
            tokens_estimated=False,
            latency_ms=int((time.perf_counter() - started) * 1000),
            status=STATUS_PROVIDER_ERROR,
            error_detail=str(exc),
            messages=messages,
            routing_reason=routing_reason,
        )
        return _error(503, str(exc), "provider_unavailable")
    except StopAsyncIteration:
        first_chunk = b""

    sniffer = UsageSniffer()

    async def body():
        try:
            sniffer.feed(first_chunk)
            yield first_chunk
            async for chunk in stream:
                sniffer.feed(chunk)
                yield chunk
        finally:
            # Runs even if the client disconnects mid-stream, so partial
            # generations are still charged for - matching how providers bill.
            if sniffer.found_usage:
                prompt_tokens = sniffer.prompt_tokens or 0
                completion_tokens = sniffer.completion_tokens or 0
                estimated = False
            else:
                prompt_tokens = estimate_tokens(_prompt_text(messages))
                completion_tokens = max(1, sniffer.text_length // 4)
                estimated = True

            ledger.record(
                user_id=user.id,
                requested_model=requested_model,
                served_model=served_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tokens_estimated=estimated,
                latency_ms=int((time.perf_counter() - started) * 1000),
                status=STATUS_OK,
                messages=messages,
                routing_reason=routing_reason,
            )

    return StreamingResponse(body(), media_type="text/event-stream")
