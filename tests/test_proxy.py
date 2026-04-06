# SPDX-License-Identifier: Apache-2.0
"""
Tests for the oMLX proxy module.

Tests cover:
- ProxySettings serialization/deserialization
- Cache key computation and canonicalization
- ProxyResponseCache operations (put, get, expiry, eviction)
- SSE event replay
- ProxyMetrics tracking
- Prompt cache optimizer

NOTE: These tests import proxy modules directly to avoid pulling in MLX
via omlx/__init__.py, since MLX is only available on Apple Silicon.
"""

import importlib
import json
import sys
import time
from pathlib import Path

import pytest

# Prevent omlx top-level __init__ from running (it imports MLX).
# We insert a mock omlx package and import submodules directly.
if "omlx" not in sys.modules:
    import types
    _mock_omlx = types.ModuleType("omlx")
    _mock_omlx.__path__ = [str(Path(__file__).parent.parent / "omlx")]
    _mock_omlx.__package__ = "omlx"
    sys.modules["omlx"] = _mock_omlx

# Now we can import proxy submodules without triggering MLX imports
from omlx.proxy.settings import ProxySettings
from omlx.proxy.cache import (
    CachedResponse,
    ProxyResponseCache,
    compute_cache_key,
    is_cacheable,
    _canonicalize_content,
)
from omlx.proxy.replay import replay_sse_events
from omlx.proxy.metrics import ProxyMetrics, _get_pricing
from omlx.proxy.prompt_optimizer import (
    optimize_cache_breakpoints,
    _add_breakpoint_to_system,
    _add_breakpoint_to_tools,
)


# =============================================================================
# ProxySettings Tests
# =============================================================================


class TestProxySettings:
    def test_defaults(self):
        s = ProxySettings()
        assert s.enabled is False
        assert s.upstream_url == "https://api.anthropic.com"
        assert s.upstream_api_key is None
        assert s.cache_enabled is True
        assert s.cache_dir is None
        assert s.cache_max_size == "1GB"
        assert s.cache_ttl_seconds == 86400
        assert s.cache_nonzero_temp is False

    def test_to_dict(self):
        s = ProxySettings(enabled=True, upstream_url="https://custom.api")
        d = s.to_dict()
        assert d["enabled"] is True
        assert d["upstream_url"] == "https://custom.api"

    def test_from_dict(self):
        data = {
            "enabled": True,
            "upstream_url": "https://custom.api",
            "cache_ttl_seconds": 3600,
        }
        s = ProxySettings.from_dict(data)
        assert s.enabled is True
        assert s.upstream_url == "https://custom.api"
        assert s.cache_ttl_seconds == 3600

    def test_from_dict_defaults(self):
        s = ProxySettings.from_dict({})
        assert s.enabled is False
        assert s.cache_max_size == "1GB"

    def test_roundtrip(self):
        original = ProxySettings(
            enabled=True,
            upstream_api_key="sk-test",
            cache_nonzero_temp=True,
        )
        restored = ProxySettings.from_dict(original.to_dict())
        assert restored.enabled == original.enabled
        assert restored.upstream_api_key == original.upstream_api_key
        assert restored.cache_nonzero_temp == original.cache_nonzero_temp

    def test_get_cache_dir_default(self):
        s = ProxySettings()
        p = s.get_cache_dir(Path("/home/user/.omlx"))
        assert p == Path("/home/user/.omlx/proxy_cache")

    def test_get_cache_dir_custom(self):
        s = ProxySettings(cache_dir="/tmp/my_cache")
        p = s.get_cache_dir(Path("/home/user/.omlx"))
        assert p == Path("/tmp/my_cache")


# =============================================================================
# Cache Key Tests
# =============================================================================


class TestCacheKey:
    def test_same_request_same_key(self):
        req = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        assert compute_cache_key(req) == compute_cache_key(req)

    def test_different_message_different_key(self):
        req1 = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        req2 = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "World"}],
        }
        assert compute_cache_key(req1) != compute_cache_key(req2)

    def test_stream_flag_excluded(self):
        req1 = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        req2 = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        assert compute_cache_key(req1) == compute_cache_key(req2)

    def test_metadata_excluded(self):
        req1 = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {"request_id": "abc"},
        }
        req2 = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {"request_id": "xyz"},
        }
        assert compute_cache_key(req1) == compute_cache_key(req2)

    def test_temperature_affects_key(self):
        req1 = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
        }
        req2 = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 1.0,
        }
        assert compute_cache_key(req1) != compute_cache_key(req2)

    def test_image_content_hashed(self):
        """Images should be hashed, not included raw."""
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "data": "abc123", "media_type": "image/png"},
            }
        ]
        result = _canonicalize_content(content)
        assert result[0]["type"] == "image"
        assert "hash" in result[0]
        assert "data" not in result[0].get("source", {})


class TestIsCacheable:
    def test_temp_zero_cacheable(self):
        assert is_cacheable({"temperature": 0}) is True

    def test_temp_none_cacheable(self):
        assert is_cacheable({}) is True

    def test_temp_nonzero_not_cacheable(self):
        assert is_cacheable({"temperature": 0.7}) is False

    def test_temp_nonzero_with_opt_in(self):
        assert is_cacheable({"temperature": 0.7}, allow_nonzero_temp=True) is True


# =============================================================================
# ProxyResponseCache Tests
# =============================================================================


class TestProxyResponseCache:
    @pytest.fixture
    def cache_dir(self, tmp_path):
        return tmp_path / "test_cache"

    @pytest.fixture
    async def cache(self, cache_dir):
        c = ProxyResponseCache(
            cache_dir=cache_dir,
            max_size="10MB",
            ttl_seconds=3600,
        )
        await c.initialize()
        yield c
        await c.close()

    @pytest.mark.asyncio
    async def test_put_and_get(self, cache):
        resp = CachedResponse(
            response_body=b'{"id": "msg_123"}',
            sse_events=None,
            input_tokens=100,
            output_tokens=50,
            model="claude-sonnet-4-6",
        )
        await cache.put("test_key", resp)

        result = await cache.get("test_key")
        assert result is not None
        assert result.response_body == b'{"id": "msg_123"}'
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.model == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_get_missing_key(self, cache):
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_sse_events_cached(self, cache):
        events = [
            'event: message_start\ndata: {"type":"message_start"}\n',
            'event: content_block_delta\ndata: {"type":"content_block_delta"}\n',
            'event: message_stop\ndata: {"type":"message_stop"}\n',
        ]
        resp = CachedResponse(
            response_body=None,
            sse_events=events,
            input_tokens=200,
            output_tokens=100,
            model="claude-opus-4-6",
        )
        await cache.put("stream_key", resp)

        result = await cache.get("stream_key")
        assert result is not None
        assert result.sse_events == events

    @pytest.mark.asyncio
    async def test_expired_entry_not_returned(self, cache_dir):
        """Entries past TTL should not be returned."""
        c = ProxyResponseCache(cache_dir=cache_dir, max_size="10MB", ttl_seconds=1)
        await c.initialize()

        resp = CachedResponse(
            response_body=b"test",
            sse_events=None,
            model="test",
        )
        await c.put("exp_key", resp)

        # Manually expire the entry
        c._conn.execute(
            "UPDATE responses SET expires_at = ? WHERE key = ?",
            (time.time() - 1, "exp_key"),
        )
        c._conn.commit()

        result = await c.get("exp_key")
        assert result is None

        await c.close()

    @pytest.mark.asyncio
    async def test_get_stats(self, cache):
        resp = CachedResponse(
            response_body=b'{"test": true}',
            sse_events=None,
            input_tokens=50,
            output_tokens=25,
            model="test",
        )
        await cache.put("k1", resp)
        await cache.put("k2", resp)

        stats = await cache.get_stats()
        assert stats["entries"] == 2
        assert stats["total_size_bytes"] > 0
        assert stats["total_cached_input_tokens"] == 100


# =============================================================================
# SSE Replay Tests
# =============================================================================


class TestSSEReplay:
    @pytest.mark.asyncio
    async def test_replay_events(self):
        events = [
            'event: message_start\ndata: {"type":"message_start"}\n',
            'event: content_block_start\ndata: {"type":"content_block_start"}\n',
            'event: content_block_delta\ndata: {"delta":{"text":"Hello"}}\n',
            'event: message_stop\ndata: {"type":"message_stop"}\n',
        ]

        replayed = []
        async for line in replay_sse_events(events):
            replayed.append(line)

        assert replayed == events

    @pytest.mark.asyncio
    async def test_replay_empty(self):
        replayed = []
        async for line in replay_sse_events([]):
            replayed.append(line)
        assert replayed == []


# =============================================================================
# ProxyMetrics Tests
# =============================================================================


class TestProxyMetrics:
    def test_initial_state(self):
        m = ProxyMetrics()
        d = m.to_dict()
        assert d["total_requests"] == 0
        assert d["cache_hits"] == 0
        assert d["cache_misses"] == 0
        assert d["cache_hit_rate_pct"] == 0.0

    def test_record_cache_hit(self):
        m = ProxyMetrics()
        cached = CachedResponse(
            response_body=b"test",
            sse_events=None,
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-6",
        )
        m.record_cache_hit(cached)

        d = m.to_dict()
        assert d["total_requests"] == 1
        assert d["cache_hits"] == 1
        assert d["tokens_saved_input"] == 1000
        assert d["tokens_saved_output"] == 500
        assert d["cost_saved_usd"] > 0

    def test_record_upstream_request(self):
        m = ProxyMetrics()
        m.record_upstream_request({
            "usage": {
                "input_tokens": 2000,
                "output_tokens": 1000,
                "cache_creation_input_tokens": 500,
                "cache_read_input_tokens": 1500,
            },
            "model": "claude-sonnet-4-6",
        })

        d = m.to_dict()
        assert d["total_requests"] == 1
        assert d["cache_misses"] == 1
        assert d["total_input_tokens"] == 2000
        assert d["total_output_tokens"] == 1000
        assert d["total_cost_usd"] > 0

    def test_cache_hit_rate(self):
        m = ProxyMetrics()
        cached = CachedResponse(
            response_body=b"test", sse_events=None,
            input_tokens=100, output_tokens=50, model="test",
        )
        m.record_cache_hit(cached)
        m.record_cache_hit(cached)
        m.record_upstream_request({"usage": {"input_tokens": 100}, "model": "test"})

        d = m.to_dict()
        assert d["cache_hit_rate_pct"] == pytest.approx(66.7, abs=0.1)

    def test_pricing_lookup(self):
        p = _get_pricing("claude-opus-4-6")
        assert p["input"] == 15.0

        p = _get_pricing("claude-sonnet-4-6")
        assert p["input"] == 3.0

        p = _get_pricing("unknown-model")
        assert p["input"] == 3.0  # default


# =============================================================================
# Prompt Optimizer Tests
# =============================================================================


class TestPromptOptimizer:
    def test_string_system_gets_breakpoint(self):
        result = _add_breakpoint_to_system("You are a helpful assistant.")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "You are a helpful assistant."
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_list_system_gets_breakpoint_on_last(self):
        system = [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]
        result = _add_breakpoint_to_system(system)
        assert "cache_control" not in result[0]
        assert result[1]["cache_control"] == {"type": "ephemeral"}

    def test_existing_cache_control_not_overwritten(self):
        system = [
            {"type": "text", "text": "Prompt", "cache_control": {"type": "ephemeral"}},
        ]
        result = _add_breakpoint_to_system(system)
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_tools_get_breakpoint(self):
        tools = [
            {"name": "read_file", "description": "Read a file"},
            {"name": "write_file", "description": "Write a file"},
        ]
        result = _add_breakpoint_to_tools(tools)
        assert "cache_control" not in result[0]
        assert result[1]["cache_control"] == {"type": "ephemeral"}

    def test_optimize_full_request(self):
        request = {
            "model": "claude-sonnet-4-6",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "read_file", "description": "Read a file"}],
        }
        result = optimize_cache_breakpoints(request)

        # System should be converted to block format with breakpoint
        assert isinstance(result["system"], list)
        assert result["system"][0]["cache_control"] == {"type": "ephemeral"}

        # Last tool should have breakpoint
        assert result["tools"][-1]["cache_control"] == {"type": "ephemeral"}

        # Messages should be unchanged
        assert result["messages"] == request["messages"]

    def test_no_system_no_tools(self):
        request = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = optimize_cache_breakpoints(request)
        assert "system" not in result or result.get("system") is None

    def test_empty_tools_list(self):
        result = _add_breakpoint_to_tools([])
        assert result == []


# =============================================================================
# ProxyEngine Tests
# =============================================================================


class TestProxyEngine:
    def test_build_upstream_headers(self):
        from omlx.proxy.engine import ProxyEngine

        settings = ProxySettings(upstream_url="https://api.anthropic.com")
        engine = ProxyEngine(settings)

        headers = engine._build_upstream_headers({
            "x-api-key": "sk-ant-test123",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "messages-2024-12-19",
            "host": "localhost:8000",
            "user-agent": "Claude/1.0",
            "content-type": "application/json",
        })

        assert headers["x-api-key"] == "sk-ant-test123"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["anthropic-beta"] == "messages-2024-12-19"
        assert headers["content-type"] == "application/json"
        # host should not be forwarded
        assert "host" not in headers
        # user-agent should not be forwarded
        assert "user-agent" not in headers

    def test_upstream_api_key_override(self):
        from omlx.proxy.engine import ProxyEngine

        settings = ProxySettings(
            upstream_url="https://api.anthropic.com",
            upstream_api_key="sk-ant-override",
        )
        engine = ProxyEngine(settings)

        headers = engine._build_upstream_headers({
            "x-api-key": "sk-ant-original",
            "authorization": "Bearer sk-ant-original",
        })

        assert headers["x-api-key"] == "sk-ant-override"
        assert "authorization" not in headers
