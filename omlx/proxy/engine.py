# SPDX-License-Identifier: Apache-2.0
"""
Proxy engine that forwards requests to the upstream Anthropic API.

Handles both streaming (SSE) and non-streaming requests, forwarding
headers and API keys transparently.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .settings import ProxySettings

logger = logging.getLogger(__name__)

# Headers to forward from client to upstream
_FORWARD_HEADER_PREFIXES = ("anthropic-", "x-stainless-")
_FORWARD_HEADERS = {
    "x-api-key",
    "authorization",
    "content-type",
    # Subscription mode may use these
    "cookie",
    "x-request-id",
}


class ProxyEngine:
    """
    Forwards Anthropic Messages API requests to upstream and streams responses back.

    Sits between Claude Code and api.anthropic.com, providing a transparent
    proxy layer that can be extended with caching and analytics.
    """

    def __init__(self, settings: ProxySettings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Initialize the httpx async client.

        Automatically detects and uses system HTTP proxy settings
        (GLOBAL_AGENT_HTTPS_PROXY, HTTPS_PROXY, etc.) so that
        subscription-mode Claude Code traffic flows through the
        egress gateway with JWT authentication intact.
        """
        import os

        # Detect proxy from environment (subscription mode uses these)
        proxy_url = (
            os.environ.get("GLOBAL_AGENT_HTTPS_PROXY")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("GLOBAL_AGENT_HTTP_PROXY")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
        )

        client_kwargs = dict(
            base_url=self.settings.upstream_url,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
            http2=True,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )

        if proxy_url:
            client_kwargs["proxy"] = proxy_url
            logger.info(
                "ProxyEngine using upstream proxy: %s",
                proxy_url.split("@")[-1] if "@" in proxy_url else proxy_url,
            )

        self._client = httpx.AsyncClient(**client_kwargs)
        logger.info(
            "ProxyEngine started: upstream=%s, proxy=%s",
            self.settings.upstream_url,
            "yes" if proxy_url else "no",
        )

    async def stop(self) -> None:
        """Close the httpx client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("ProxyEngine stopped")

    def _build_upstream_headers(
        self, incoming_headers: dict[str, str]
    ) -> dict[str, str]:
        """
        Build headers to send upstream from the incoming request headers.

        Forwards anthropic-* headers and the API key. If an upstream_api_key
        is configured, it overrides the client's key.
        """
        headers: dict[str, str] = {}

        for name, value in incoming_headers.items():
            lower = name.lower()
            # Forward anthropic-* headers
            if any(lower.startswith(p) for p in _FORWARD_HEADER_PREFIXES):
                headers[lower] = value
            # Forward specific headers
            elif lower in _FORWARD_HEADERS:
                headers[lower] = value

        # Override API key if configured
        if self.settings.upstream_api_key:
            headers["x-api-key"] = self.settings.upstream_api_key
            # Remove Bearer auth if we're using a direct key
            headers.pop("authorization", None)

        # Ensure content-type is set
        headers.setdefault("content-type", "application/json")

        return headers

    async def forward_request(
        self,
        request_body: dict[str, Any],
        incoming_headers: dict[str, str],
    ) -> httpx.Response:
        """
        Forward a non-streaming request to upstream.

        Args:
            request_body: The parsed JSON request body.
            incoming_headers: Headers from the incoming client request.

        Returns:
            The upstream httpx.Response.
        """
        if not self._client:
            raise RuntimeError("ProxyEngine not started")

        headers = self._build_upstream_headers(incoming_headers)
        # Ensure stream is False for non-streaming
        body = {**request_body, "stream": False}

        start = time.perf_counter()
        response = await self._client.post(
            "/v1/messages",
            json=body,
            headers=headers,
        )
        elapsed = time.perf_counter() - start

        logger.info(
            "Proxy forward: status=%d, elapsed=%.2fs, model=%s",
            response.status_code,
            elapsed,
            request_body.get("model", "unknown"),
        )

        return response

    async def stream_request(
        self,
        request_body: dict[str, Any],
        incoming_headers: dict[str, str],
    ) -> AsyncIterator[tuple[str, list[str]]]:
        """
        Forward a streaming request to upstream, yielding raw SSE lines.

        Yields each SSE line as it arrives from upstream. Also accumulates
        all lines in a buffer returned as the second element of the final
        yield for caching purposes.

        Args:
            request_body: The parsed JSON request body.
            incoming_headers: Headers from the incoming client request.

        Yields:
            Tuples of (sse_line, accumulated_buffer).
            Each yield provides the current line and the full buffer so far.
        """
        if not self._client:
            raise RuntimeError("ProxyEngine not started")

        headers = self._build_upstream_headers(incoming_headers)
        body = {**request_body, "stream": True}

        buffer: list[str] = []

        async with self._client.stream(
            "POST",
            "/v1/messages",
            json=body,
            headers=headers,
        ) as response:
            if response.status_code != 200:
                # Read error body and yield as error
                error_body = await response.aread()
                error_line = f"data: {error_body.decode('utf-8', errors='replace')}\n\n"
                buffer.append(error_line)
                yield error_line, buffer
                return

            async for line in response.aiter_lines():
                if not line:
                    continue
                # Reconstruct SSE format: each logical event is
                # "event: ...\ndata: ...\n\n" but aiter_lines splits on \n.
                # We forward lines exactly, adding the newline back.
                sse_line = line + "\n"
                buffer.append(sse_line)
                yield sse_line, buffer

    async def stream_request_raw(
        self,
        request_body: dict[str, Any],
        incoming_headers: dict[str, str],
    ) -> AsyncIterator[str]:
        """
        Forward a streaming request, yielding raw SSE text chunks.

        This is the simple version that just streams through without
        accumulating a buffer. Use stream_request() when you need the
        buffer for caching.

        Args:
            request_body: The parsed JSON request body.
            incoming_headers: Headers from the incoming client request.

        Yields:
            Raw SSE text lines from upstream.
        """
        if not self._client:
            raise RuntimeError("ProxyEngine not started")

        headers = self._build_upstream_headers(incoming_headers)
        body = {**request_body, "stream": True}

        async with self._client.stream(
            "POST",
            "/v1/messages",
            json=body,
            headers=headers,
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                yield f"data: {error_body.decode('utf-8', errors='replace')}\n\n"
                return

            async for line in response.aiter_lines():
                if not line:
                    continue
                yield line + "\n"
