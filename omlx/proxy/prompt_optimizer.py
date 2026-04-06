# SPDX-License-Identifier: Apache-2.0
"""
Prompt cache optimizer for Anthropic API requests.

Analyzes outgoing requests and adds/adjusts cache_control breakpoints
to maximize Anthropic's native prompt caching, reducing costs for
repeated system prompts and tool definitions.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def optimize_cache_breakpoints(request_body: dict[str, Any]) -> dict[str, Any]:
    """
    Optimize a request for Anthropic's prompt caching.

    Adds ``cache_control: {"type": "ephemeral"}`` breakpoints at the end
    of stable content (system prompt, tool definitions) to maximize cache
    reuse across requests.

    The Anthropic API caches content up to and including breakpoints.
    Placing breakpoints at the end of system prompts and tool definitions
    means subsequent requests with the same prefix will use cached tokens.

    Args:
        request_body: The raw Anthropic Messages API request dict.

    Returns:
        Modified request body with cache_control breakpoints added.
    """
    body = dict(request_body)  # Shallow copy

    # Optimize system prompt
    system = body.get("system")
    if system:
        body["system"] = _add_breakpoint_to_system(system)

    # Optimize tool definitions
    tools = body.get("tools")
    if tools and isinstance(tools, list) and len(tools) > 0:
        body["tools"] = _add_breakpoint_to_tools(tools)

    return body


def _add_breakpoint_to_system(system: Any) -> Any:
    """
    Add cache_control breakpoint to the end of the system prompt.

    Handles both string and list-of-blocks system formats.
    """
    cache_control = {"type": "ephemeral"}

    if isinstance(system, str):
        # Convert string to block format so we can add cache_control
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": cache_control,
            }
        ]

    if isinstance(system, list) and len(system) > 0:
        # Add breakpoint to the last block
        result = list(system)  # Shallow copy
        last = dict(result[-1]) if isinstance(result[-1], dict) else result[-1]
        if isinstance(last, dict) and "cache_control" not in last:
            last["cache_control"] = cache_control
            result[-1] = last
        return result

    return system


def _add_breakpoint_to_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add cache_control breakpoint to the last tool definition.

    This caches the entire tool definition block for reuse across
    requests that share the same tool set.
    """
    if not tools:
        return tools

    result = list(tools)  # Shallow copy
    last = dict(result[-1])
    if "cache_control" not in last:
        last["cache_control"] = {"type": "ephemeral"}
        result[-1] = last

    return result
