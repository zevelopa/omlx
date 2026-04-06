# SPDX-License-Identifier: Apache-2.0
"""
Lightweight FastAPI app for proxy-only mode.

This module creates a minimal FastAPI application that forwards requests
to the upstream Anthropic API without importing MLX or any model inference
code. Used by ``omlx proxy`` to run on non-Apple-Silicon machines or to
minimize memory footprint.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .engine import ProxyEngine
from .settings import ProxySettings

logger = logging.getLogger(__name__)


class ProxyAppState:
    """Holds runtime state for the proxy app."""

    def __init__(self) -> None:
        self.proxy_engine: Optional[ProxyEngine] = None
        self.settings: Optional[ProxySettings] = None
        self.cache: Optional[object] = None  # ProxyResponseCache, set later
        self.metrics: Optional[object] = None  # ProxyMetrics, set later


_state = ProxyAppState()


def get_proxy_state() -> ProxyAppState:
    """Get the proxy app state."""
    return _state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop the proxy engine."""
    if _state.proxy_engine:
        await _state.proxy_engine.start()

    # Initialize cache if enabled
    if _state.settings and _state.settings.cache_enabled:
        try:
            from .cache import ProxyResponseCache
            from pathlib import Path

            base_path = Path.home() / ".omlx"
            cache_dir = _state.settings.get_cache_dir(base_path)
            _state.cache = ProxyResponseCache(
                cache_dir=cache_dir,
                max_size=_state.settings.cache_max_size,
                ttl_seconds=_state.settings.cache_ttl_seconds,
            )
            await _state.cache.initialize()
            logger.info("Proxy response cache initialized at %s", cache_dir)
        except Exception as e:
            logger.warning("Failed to initialize proxy cache: %s", e)
            _state.cache = None

    # Initialize metrics
    try:
        from .metrics import ProxyMetrics
        _state.metrics = ProxyMetrics()
        logger.info("Proxy metrics initialized")
    except Exception as e:
        logger.warning("Failed to initialize proxy metrics: %s", e)

    yield

    # Shutdown
    if _state.proxy_engine:
        await _state.proxy_engine.stop()
    if _state.cache:
        await _state.cache.close()


def create_proxy_app() -> FastAPI:
    """Create the FastAPI app for proxy mode."""
    app = FastAPI(
        title="oMLX Proxy",
        description="Caching proxy for Anthropic API",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "ok",
            "mode": "proxy",
            "upstream": _state.settings.upstream_url if _state.settings else None,
            "cache_enabled": _state.settings.cache_enabled if _state.settings else False,
        }

    @app.get("/v1/models")
    async def list_models():
        """Return a placeholder models list for proxy mode."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "claude-opus-4-6",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "anthropic",
                },
                {
                    "id": "claude-sonnet-4-6",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "anthropic",
                },
                {
                    "id": "claude-haiku-4-5-20251001",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "anthropic",
                },
            ],
        }

    @app.post("/v1/messages")
    async def proxy_messages(request: Request):
        """
        Forward Anthropic Messages API requests to upstream.

        This endpoint accepts requests in Anthropic Messages API format,
        optionally serves them from cache, or forwards them to the
        upstream Anthropic API.
        """
        if not _state.proxy_engine:
            return JSONResponse(
                status_code=503,
                content={"error": {"type": "server_error", "message": "Proxy engine not initialized"}},
            )

        # Parse request body
        body_bytes = await request.body()
        try:
            request_body = json.loads(body_bytes)
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": {"type": "invalid_request_error", "message": "Invalid JSON body"}},
            )

        # Optimize for Anthropic's native prompt caching
        from .prompt_optimizer import optimize_cache_breakpoints
        request_body = optimize_cache_breakpoints(request_body)

        is_streaming = request_body.get("stream", False)
        incoming_headers = dict(request.headers)

        # Check cache first
        cache_key = None
        if _state.cache and _state.settings and _state.settings.cache_enabled:
            from .cache import compute_cache_key, is_cacheable

            if is_cacheable(request_body, _state.settings.cache_nonzero_temp):
                cache_key = compute_cache_key(request_body)
                cached = await _state.cache.get(cache_key)
                if cached is not None:
                    logger.info(
                        "Proxy cache HIT: model=%s, key=%s",
                        request_body.get("model", "unknown"),
                        cache_key[:12],
                    )
                    if _state.metrics:
                        _state.metrics.record_cache_hit(cached)

                    if is_streaming and cached.sse_events:
                        # Replay cached SSE events
                        from .replay import replay_sse_events

                        return StreamingResponse(
                            replay_sse_events(cached.sse_events),
                            media_type="text/event-stream",
                            headers={
                                "X-Accel-Buffering": "no",
                                "Cache-Control": "no-cache",
                                "X-Proxy-Cache": "HIT",
                            },
                        )
                    elif cached.response_body:
                        # Return cached non-streaming response
                        return JSONResponse(
                            content=json.loads(cached.response_body),
                            headers={"X-Proxy-Cache": "HIT"},
                        )

        # Cache miss - forward to upstream
        logger.info(
            "Proxy cache MISS: model=%s, stream=%s",
            request_body.get("model", "unknown"),
            is_streaming,
        )

        if is_streaming:
            return await _handle_streaming(
                request_body, incoming_headers, cache_key
            )
        else:
            return await _handle_non_streaming(
                request_body, incoming_headers, cache_key
            )

    async def _handle_non_streaming(
        request_body: dict,
        incoming_headers: dict[str, str],
        cache_key: str | None,
    ):
        """Handle a non-streaming proxy request."""
        response = await _state.proxy_engine.forward_request(
            request_body, incoming_headers
        )

        response_body = response.content
        status_code = response.status_code

        # Cache successful responses
        if status_code == 200 and cache_key and _state.cache:
            try:
                from .cache import CachedResponse

                parsed = json.loads(response_body)
                usage = parsed.get("usage", {})
                cached_resp = CachedResponse(
                    response_body=response_body,
                    sse_events=None,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_creation_input_tokens=usage.get(
                        "cache_creation_input_tokens", 0
                    ),
                    cache_read_input_tokens=usage.get(
                        "cache_read_input_tokens", 0
                    ),
                    model=request_body.get("model", "unknown"),
                )
                await _state.cache.put(cache_key, cached_resp)
            except Exception as e:
                logger.warning("Failed to cache response: %s", e)

        # Track metrics
        if _state.metrics and status_code == 200:
            try:
                parsed = json.loads(response_body)
                _state.metrics.record_upstream_request(parsed)
            except Exception:
                pass

        # Forward response headers
        resp_headers = {"X-Proxy-Cache": "MISS"}
        return JSONResponse(
            content=json.loads(response_body) if status_code == 200 else response_body.decode("utf-8", errors="replace"),
            status_code=status_code,
            headers=resp_headers,
        )

    async def _handle_streaming(
        request_body: dict,
        incoming_headers: dict[str, str],
        cache_key: str | None,
    ):
        """Handle a streaming proxy request with cache buffering."""

        async def stream_and_buffer():
            """Stream SSE events while buffering for cache."""
            buffer: list[str] = []
            usage_data = {}

            async for sse_line, accumulated in _state.proxy_engine.stream_request(
                request_body, incoming_headers
            ):
                yield sse_line
                buffer = accumulated

                # Try to extract usage from message_delta events
                if "message_delta" in sse_line:
                    try:
                        # Parse the data line after "data: "
                        for line in sse_line.split("\n"):
                            if line.startswith("data: "):
                                data = json.loads(line[6:])
                                if "usage" in data:
                                    usage_data = data["usage"]
                    except (json.JSONDecodeError, KeyError):
                        pass

            # After stream completes, cache the buffered events
            if cache_key and _state.cache and buffer:
                try:
                    from .cache import CachedResponse

                    cached_resp = CachedResponse(
                        response_body=None,
                        sse_events=buffer,
                        input_tokens=usage_data.get("input_tokens", 0),
                        output_tokens=usage_data.get("output_tokens", 0),
                        cache_creation_input_tokens=usage_data.get(
                            "cache_creation_input_tokens", 0
                        ),
                        cache_read_input_tokens=usage_data.get(
                            "cache_read_input_tokens", 0
                        ),
                        model=request_body.get("model", "unknown"),
                    )
                    await _state.cache.put(cache_key, cached_resp)
                except Exception as e:
                    logger.warning("Failed to cache streamed response: %s", e)

            # Track metrics
            if _state.metrics and usage_data:
                _state.metrics.record_upstream_request({"usage": usage_data})

        return StreamingResponse(
            stream_and_buffer(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "X-Proxy-Cache": "MISS",
            },
        )

    # Proxy stats endpoint
    @app.get("/admin/api/proxy/stats")
    async def proxy_stats():
        """Return proxy cache and metrics statistics."""
        stats = {
            "mode": "proxy",
            "upstream_url": _state.settings.upstream_url if _state.settings else None,
        }

        if _state.metrics:
            stats.update(_state.metrics.to_dict())

        if _state.cache:
            cache_stats = await _state.cache.get_stats()
            stats["cache"] = cache_stats

        return stats

    return app


def init_proxy_app(settings: ProxySettings) -> FastAPI:
    """
    Create and initialize the proxy FastAPI app.

    Args:
        settings: Proxy configuration.

    Returns:
        Configured FastAPI application.
    """
    _state.settings = settings
    _state.proxy_engine = ProxyEngine(settings)

    return create_proxy_app()
