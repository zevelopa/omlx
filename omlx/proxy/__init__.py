# SPDX-License-Identifier: Apache-2.0
"""
Proxy module for oMLX.

Provides a caching proxy between Claude Code (or any Anthropic API client)
and the upstream Anthropic API. When enabled, oMLX forwards /v1/messages
requests to api.anthropic.com while caching responses for cost and latency
savings.
"""

from .engine import ProxyEngine
from .settings import ProxySettings

__all__ = [
    "ProxyEngine",
    "ProxySettings",
]
