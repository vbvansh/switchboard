"""OpenAI-compatible HTTP surface, now metered.

Request lifecycle:

    identify (401) -> budget check (402) -> serve -> measure -> record -> reply

Milestone 2 scope: identity, budgets, and accounting. Model choice is still
fixed - `_resolve_model` is the seam the milestone 4 router plugs into.

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

from switchboard.config import AUTO_MODEL, settings
from switchboard.ledger import (
    STATUS_BLOCKED_BUDGET,
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
from switchboard.pricing import PriceTable
from switchboard.providers.ollama import OllamaProvider, ProviderUnavailable
from switchboard.schema import require_up_to_date
from switchboard.streaming import UsageSniffer, request_usage_in_stream

PROVIDER_DOWN_DETAIL = (
    "Cannot reach Ollama at {url}. Start it with `ollama serve` "
    "(or launch the Ollama desktop app) and try again."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Refuse to start against a database shaped differently to what this code
    # expects. Starting anyway does not fail cleanly - it fails later, mid
    # request, with an error pointing somewhere unhelpful.
    require_up_to_date(settings.database_url)

    database = Database(settings.database_url)
    prices = PriceTable.load(settings.prices_file)

    app.state.provider = OllamaProvider(settings)
    app.state.database = database
    app.state.ledger = LedgerService(database, prices, settings.store_prompts)
    try:
        yield
    finally:
        await app.state.provider.aclose()
        database.dispose()


app = FastAPI(
    title="Switchboard",
    description="Local-only AI model router. No API keys, no remote providers.",
    version="0.2.0",
    lifespan=lifespan,
)


# --- Small helpers ---------------------------------------------------------


def _error(status: int, message: str, kind: str) -> JSONResponse:
    """OpenAI-shaped error body, so existing clients surface it properly."""
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": kind}}
    )


def _provider_down() -> JSONResponse:
    return _error(
        503,
        PROVIDER_DOWN_DETAIL.format(url=settings.ollama_base_url),
        "provider_unavailable",
    )


def _identify(request: Request) -> User:
    """Resolve the caller. Raises AuthenticationError."""
    token = extract_bearer_token(request.headers.get("authorization"))
    return request.app.state.ledger.authenticate(token)


def _resolve_model(payload: dict[str, Any]) -> str:
    """Decide which model serves this request.

    Milestone 2 still returns the fixed default for `auto`. This function is the
    single seam the routing strategies replace in milestone 4.
    """
    requested = payload.get("model")
    if not requested or requested == AUTO_MODEL:
        return settings.default_model
    return str(requested)


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
    parts = []
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            parts.append(message["content"])
    return "\n".join(parts)


# --- Endpoints -------------------------------------------------------------


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Open by design - a health check that needs credentials is useless."""
    return {
        "status": "ok",
        "provider_reachable": await request.app.state.provider.is_healthy(),
        "provider_url": settings.ollama_base_url,
        "default_model": settings.default_model,
        "store_prompts": settings.store_prompts,
    }


@app.get("/v1/models")
async def list_models(request: Request) -> Response:
    try:
        _identify(request)
    except AuthenticationError as exc:
        return _error(401, str(exc), "authentication_error")

    try:
        upstream = await request.app.state.provider.list_models()
    except ProviderUnavailable:
        return _provider_down()
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type="application/json",
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    ledger: LedgerService = request.app.state.ledger
    provider: OllamaProvider = request.app.state.provider

    try:
        payload: dict[str, Any] = await request.json()
    except ValueError:
        return _error(400, "Request body must be valid JSON.", "invalid_request_error")

    try:
        user = _identify(request)
    except AuthenticationError as exc:
        return _error(401, str(exc), "authentication_error")

    requested_model = str(payload.get("model") or AUTO_MODEL)
    served_model = _resolve_model(payload)
    messages = payload.get("messages")

    try:
        ledger.assert_within_budget(user)
    except BudgetExceededError as exc:
        # Recorded so a blocked attempt is visible in the ledger. Costs zero and
        # is excluded from spend totals, so retrying cannot dig a deeper hole.
        ledger.record(
            user_id=user.id,
            requested_model=requested_model,
            served_model=served_model,
            prompt_tokens=0,
            completion_tokens=0,
            tokens_estimated=False,
            latency_ms=0,
            status=STATUS_BLOCKED_BUDGET,
            error_detail=str(exc),
            messages=messages,
        )
        return _error(402, str(exc), "insufficient_quota")

    payload["model"] = served_model
    started = time.perf_counter()

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
        )

    try:
        upstream = await provider.chat_completion(payload)
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
        )
        return _provider_down()

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
        status=(
            STATUS_OK if upstream.status_code < 400 else STATUS_PROVIDER_ERROR
        ),
        error_detail=None if upstream.status_code < 400 else upstream.text[:500],
        messages=messages,
    )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type="application/json",
    )


async def _serve_streaming(
    *,
    payload: dict[str, Any],
    provider: OllamaProvider,
    ledger: LedgerService,
    user: User,
    requested_model: str,
    served_model: str,
    messages: Any,
    started: float,
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
        )
        return _provider_down()
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
            )

    return StreamingResponse(body(), media_type="text/event-stream")
