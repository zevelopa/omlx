# SPDX-License-Identifier: Apache-2.0
"""
SSE event replay for cached streaming responses.

Replays buffered SSE events as an async iterator, compatible with
FastAPI's StreamingResponse.
"""

from __future__ import annotations

from collections.abc import AsyncIterator


async def replay_sse_events(events: list[str]) -> AsyncIterator[str]:
    """
    Replay cached SSE events as an async iterator.

    Yields each SSE line from the cached event list. No artificial
    delays are added - cached responses are served as fast as possible.

    Args:
        events: List of raw SSE line strings from the original stream.

    Yields:
        SSE line strings.
    """
    for event_line in events:
        yield event_line
