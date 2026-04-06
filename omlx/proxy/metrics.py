# SPDX-License-Identifier: Apache-2.0
"""
Metrics tracking for the Anthropic API proxy.

Tracks token usage, cache hit rates, and estimated cost savings.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Anthropic pricing per 1M tokens (approximate, USD)
# These are estimates and may change. Update as needed.
_PRICING = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0, "cache_read": 0.08, "cache_write": 1.0},
}

# Default pricing for unknown models
_DEFAULT_PRICING = {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75}


def _get_pricing(model: str) -> dict[str, float]:
    """Get pricing for a model, with fallback to default."""
    for key, pricing in _PRICING.items():
        if key in model:
            return pricing
    return _DEFAULT_PRICING


class ProxyMetrics:
    """Thread-safe metrics tracker for the proxy."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.start_time = time.time()

        # Counters
        self.total_requests: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

        # Token tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cache_creation_tokens: int = 0
        self.total_cache_read_tokens: int = 0

        # Tokens saved by proxy cache (not sent upstream)
        self.tokens_saved_input: int = 0
        self.tokens_saved_output: int = 0

        # Cost tracking (estimated USD)
        self.total_cost_usd: float = 0.0
        self.cost_saved_usd: float = 0.0

    def record_cache_hit(self, cached: Any) -> None:
        """Record a cache hit and the tokens that were saved."""
        with self._lock:
            self.total_requests += 1
            self.cache_hits += 1
            self.tokens_saved_input += cached.input_tokens
            self.tokens_saved_output += cached.output_tokens

            # Estimate cost saved
            pricing = _get_pricing(cached.model)
            saved = (
                cached.input_tokens * pricing["input"] / 1_000_000
                + cached.output_tokens * pricing["output"] / 1_000_000
            )
            self.cost_saved_usd += saved

    def record_upstream_request(self, response_data: dict[str, Any]) -> None:
        """Record metrics from an upstream API response."""
        usage = response_data.get("usage", {})
        model = response_data.get("model", "unknown")

        with self._lock:
            self.total_requests += 1
            self.cache_misses += 1

            in_tokens = usage.get("input_tokens", 0)
            out_tokens = usage.get("output_tokens", 0)
            cache_create = usage.get("cache_creation_input_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)

            self.total_input_tokens += in_tokens
            self.total_output_tokens += out_tokens
            self.total_cache_creation_tokens += cache_create
            self.total_cache_read_tokens += cache_read

            # Estimate cost
            pricing = _get_pricing(model)
            cost = (
                in_tokens * pricing["input"] / 1_000_000
                + out_tokens * pricing["output"] / 1_000_000
                + cache_create * pricing["cache_write"] / 1_000_000
                + cache_read * pricing["cache_read"] / 1_000_000
            )
            self.total_cost_usd += cost

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as a dictionary."""
        with self._lock:
            uptime = time.time() - self.start_time
            hit_rate = (
                self.cache_hits / self.total_requests * 100
                if self.total_requests > 0
                else 0.0
            )

            return {
                "uptime_seconds": round(uptime, 1),
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate_pct": round(hit_rate, 1),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_cache_creation_tokens": self.total_cache_creation_tokens,
                "total_cache_read_tokens": self.total_cache_read_tokens,
                "tokens_saved_input": self.tokens_saved_input,
                "tokens_saved_output": self.tokens_saved_output,
                "total_cost_usd": round(self.total_cost_usd, 4),
                "cost_saved_usd": round(self.cost_saved_usd, 4),
            }
