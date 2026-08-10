"""OpenAI-compatible HTTP surface.

Milestone 1 scope: forward every chat completion to a single fixed model. No
routing, no ledger, no guardrails. The value being proved here is that an
unmodified OpenAI client can talk to Switchboard by changing only `base_url`.

The request body is intentionally NOT validated against a schema. Pass-through
keeps compatibility with client features this code does not model, and the
routing layer only ever needs to rewrite one field: `model`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from switchboard.config import AUTO_MODEL, settings
from switchboard.providers.ollama import OllamaProvider, ProviderUnavailable

PROVIDER_DOWN_DETAIL = (
    "Cannot reach Ollama at {url}. Start it with `ollama serve` "
    "(or launch the Ollama desktop app) and try again."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.provider = OllamaProvider(settings)
    try:
        yield
    finally:
        await app.state.provider.aclose()


app = FastAPI(
    title="Switchboard",
    description="Local-only AI model router. No API keys, no remote providers.",
    version="0.1.0",
    lifespan=lifespan,
)


def _provider(request: Request) -> OllamaProvider:
    return request.app.state.provider


def _provider_down() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": PROVIDER_DOWN_DETAIL.format(
                    url=settings.ollama_base_url
                ),
                "type": "provider_unavailable",
            }
        },
    )


def _resolve_model(payload: dict[str, Any]) -> str:
    """Decide which model actually serves this request.

    Milestone 1: always `default_model` when the client asks for `auto` or
    omits the field; otherwise honour the client's explicit choice. This is the
    single seam the routing strategies will plug into later.
    """
    requested = payload.get("model")
    if not requested or requested == AUTO_MODEL:
        return settings.default_model
    return str(requested)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    provider = _provider(request)
    return {
        "status": "ok",
        "provider_reachable": await provider.is_healthy(),
        "provider_url": settings.ollama_base_url,
        "default_model": settings.default_model,
    }


@app.get("/v1/models")
async def list_models(request: Request) -> Response:
    try:
        upstream = await _provider(request).list_models()
    except ProviderUnavailable:
        return _provider_down()
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type="application/json",
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    try:
        payload: dict[str, Any] = await request.json()
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Request body must be valid JSON.",
                    "type": "invalid_request_error",
                }
            },
        )

    payload["model"] = _resolve_model(payload)
    provider = _provider(request)

    if payload.get("stream"):
        # The generator is created eagerly so a dead provider surfaces as a
        # clean 503 here rather than as a truncated stream the client has to
        # guess about.
        stream = provider.stream_chat_completion(payload)
        try:
            first_chunk = await anext(stream)
        except ProviderUnavailable:
            return _provider_down()
        except StopAsyncIteration:
            first_chunk = b""

        async def body():
            yield first_chunk
            async for chunk in stream:
                yield chunk

        return StreamingResponse(body(), media_type="text/event-stream")

    try:
        upstream = await provider.chat_completion(payload)
    except ProviderUnavailable:
        return _provider_down()

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type="application/json",
    )
