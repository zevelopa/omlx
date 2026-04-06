# SPDX-License-Identifier: Apache-2.0
"""
Proxy settings for oMLX caching proxy mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ProxySettings:
    """Configuration for the Anthropic API caching proxy."""

    enabled: bool = False
    upstream_url: str = "https://api.anthropic.com"
    # If set, use this key for upstream. Otherwise forward the client's key.
    upstream_api_key: str | None = None
    cache_enabled: bool = True
    cache_dir: str | None = None  # None means ~/.omlx/proxy_cache
    cache_max_size: str = "1GB"
    cache_ttl_seconds: int = 86400  # 24 hours
    cache_nonzero_temp: bool = False

    def get_cache_dir(self, base_path: Path) -> Path:
        """Resolve the proxy cache directory path."""
        if self.cache_dir:
            return Path(self.cache_dir).expanduser().resolve()
        return base_path / "proxy_cache"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "upstream_url": self.upstream_url,
            "upstream_api_key": self.upstream_api_key,
            "cache_enabled": self.cache_enabled,
            "cache_dir": self.cache_dir,
            "cache_max_size": self.cache_max_size,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "cache_nonzero_temp": self.cache_nonzero_temp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProxySettings:
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            upstream_url=data.get("upstream_url", "https://api.anthropic.com"),
            upstream_api_key=data.get("upstream_api_key"),
            cache_enabled=data.get("cache_enabled", True),
            cache_dir=data.get("cache_dir"),
            cache_max_size=data.get("cache_max_size", "1GB"),
            cache_ttl_seconds=data.get("cache_ttl_seconds", 86400),
            cache_nonzero_temp=data.get("cache_nonzero_temp", False),
        )
