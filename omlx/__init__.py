# SPDX-License-Identifier: Apache-2.0
"""
omlx: LLM inference server, optimized for your Mac

This package provides native Apple Silicon GPU acceleration using
Apple's MLX framework and mlx-lm for LLMs.

Features:
- Continuous batching via vLLM-style scheduler
- OpenAI-compatible API server
- Paged KV cache with prefix sharing
- Tiered cache (GPU + paged SSD offloading)
- Anthropic API caching proxy for Claude Code

Note: Core inference symbols (Scheduler, EngineCore, etc.) are lazily
imported so that submodules like ``omlx.proxy`` can be used on machines
without MLX (e.g. Linux servers running the proxy).
"""

from omlx._version import __version__

__all__ = [
    # Request management
    "Request",
    "RequestOutput",
    "RequestStatus",
    "SamplingParams",
    # Scheduler
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    # Engine
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
    # Model registry
    "get_registry",
    "ModelOwnershipError",
    # Prefix cache (paged SSD-only)
    "BlockAwarePrefixCache",
    # Paged cache (memory efficiency)
    "PagedCacheManager",
    "CacheBlock",
    "BlockTable",
    "PagedCacheStats",
    "CacheStats",  # Backward compatibility alias
    # Version
    "__version__",
]

# Lazy imports: these pull in MLX which is only available on Apple Silicon.
# Using __getattr__ allows `omlx.proxy` to be imported on any platform.
_LAZY_IMPORTS = {
    "Request": "omlx.request",
    "RequestOutput": "omlx.request",
    "RequestStatus": "omlx.request",
    "SamplingParams": "omlx.request",
    "Scheduler": "omlx.scheduler",
    "SchedulerConfig": "omlx.scheduler",
    "SchedulerOutput": "omlx.scheduler",
    "EngineCore": "omlx.engine_core",
    "AsyncEngineCore": "omlx.engine_core",
    "EngineConfig": "omlx.engine_core",
    "BlockAwarePrefixCache": "omlx.cache.prefix_cache",
    "PagedCacheManager": "omlx.cache.paged_cache",
    "CacheBlock": "omlx.cache.paged_cache",
    "BlockTable": "omlx.cache.paged_cache",
    "PrefixCacheStats": "omlx.cache.stats",
    "PagedCacheStats": "omlx.cache.stats",
    "get_registry": "omlx.model_registry",
    "ModelOwnershipError": "omlx.model_registry",
}


def __getattr__(name: str):
    if name == "CacheStats":
        # Backward compatibility alias
        from omlx.cache.stats import PagedCacheStats
        return PagedCacheStats
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(mod, name)
        globals()[name] = val  # Cache for fast subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
