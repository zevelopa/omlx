# SPDX-License-Identifier: Apache-2.0
"""
SQLite-backed response cache for the Anthropic API proxy.

Caches complete API responses keyed by a SHA-256 hash of the request
content. Supports TTL-based expiration and LRU size-based eviction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """A cached API response."""

    response_body: bytes | None  # For non-streaming responses
    sse_events: list[str] | None  # For streaming responses
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    model: str = ""


def is_cacheable(request_body: dict[str, Any], allow_nonzero_temp: bool = False) -> bool:
    """
    Determine if a request is cacheable.

    Only caches deterministic requests (temperature=0 or None) by default.
    """
    temp = request_body.get("temperature")
    if temp is not None and temp > 0 and not allow_nonzero_temp:
        return False
    return True


def _canonicalize_content(content: Any) -> Any:
    """
    Canonicalize message content for hashing.

    Handles string content, list content blocks, and image data.
    Images are hashed by their data to avoid huge cache keys.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        result = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "image":
                    # Hash image data instead of including raw base64
                    source = block.get("source", {})
                    source_hash = hashlib.sha256(
                        json.dumps(source, sort_keys=True).encode()
                    ).hexdigest()
                    result.append({"type": "image", "hash": source_hash})
                elif block_type == "tool_result":
                    # Include tool results but canonicalize nested content
                    result.append({
                        "type": "tool_result",
                        "tool_use_id": block.get("tool_use_id", ""),
                        "content": _canonicalize_content(block.get("content", "")),
                        "is_error": block.get("is_error"),
                    })
                else:
                    result.append(block)
            else:
                result.append(block)
        return result
    return content


def _canonicalize_system(system: Any) -> Any:
    """Canonicalize system prompt, stripping cache_control directives."""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        result = []
        for block in system:
            if isinstance(block, dict):
                # Strip cache_control from system blocks for key purposes
                cleaned = {k: v for k, v in block.items() if k != "cache_control"}
                result.append(cleaned)
            else:
                result.append(block)
        return result
    return system


def compute_cache_key(request_body: dict[str, Any]) -> str:
    """
    Compute a deterministic cache key from a request.

    Includes only fields that affect the response content.
    Excludes: stream, metadata, cache_control directives.
    """
    key_data = {
        "model": request_body.get("model"),
        "max_tokens": request_body.get("max_tokens"),
        "temperature": request_body.get("temperature"),
        "top_p": request_body.get("top_p"),
        "top_k": request_body.get("top_k"),
        "stop_sequences": request_body.get("stop_sequences"),
        "system": _canonicalize_system(request_body.get("system")),
        "messages": [
            {
                "role": msg.get("role", ""),
                "content": _canonicalize_content(msg.get("content", "")),
            }
            for msg in request_body.get("messages", [])
        ],
        "tools": request_body.get("tools"),
        "tool_choice": request_body.get("tool_choice"),
    }

    serialized = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ProxyResponseCache:
    """
    SQLite-backed cache for proxy API responses.

    Uses synchronous sqlite3 with run_in_executor pattern would be
    ideal, but for simplicity we use synchronous access since cache
    operations are fast (< 1ms for reads, < 5ms for writes).
    """

    def __init__(
        self,
        cache_dir: Path,
        max_size: str = "1GB",
        ttl_seconds: int = 86400,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self._db_path = self.cache_dir / "responses.db"
        self._conn: sqlite3.Connection | None = None

        # Parse max size
        from ..config import parse_size
        self.max_size_bytes = parse_size(max_size)

    async def initialize(self) -> None:
        """Create the cache database and tables."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                key TEXT PRIMARY KEY,
                model TEXT,
                response_body BLOB,
                sse_events TEXT,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_creation_input_tokens INTEGER DEFAULT 0,
                cache_read_input_tokens INTEGER DEFAULT 0,
                created_at REAL,
                expires_at REAL,
                size_bytes INTEGER DEFAULT 0,
                last_accessed_at REAL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires ON responses(expires_at)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_accessed ON responses(last_accessed_at)
        """)
        self._conn.commit()
        logger.info("Proxy cache DB initialized at %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    async def get(self, key: str) -> CachedResponse | None:
        """
        Retrieve a cached response by key.

        Returns None if not found or expired.
        """
        if not self._conn:
            return None

        now = time.time()
        cursor = self._conn.execute(
            "SELECT response_body, sse_events, input_tokens, output_tokens, "
            "cache_creation_input_tokens, cache_read_input_tokens, model "
            "FROM responses WHERE key = ? AND expires_at > ?",
            (key, now),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        # Update last accessed time
        self._conn.execute(
            "UPDATE responses SET last_accessed_at = ? WHERE key = ?",
            (now, key),
        )
        self._conn.commit()

        response_body, sse_events_json, in_tok, out_tok, cache_create, cache_read, model = row

        sse_events = None
        if sse_events_json:
            try:
                sse_events = json.loads(sse_events_json)
            except json.JSONDecodeError:
                pass

        return CachedResponse(
            response_body=response_body,
            sse_events=sse_events,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_creation_input_tokens=cache_create,
            cache_read_input_tokens=cache_read,
            model=model or "",
        )

    async def put(self, key: str, response: CachedResponse) -> None:
        """Store a response in the cache."""
        if not self._conn:
            return

        now = time.time()
        expires_at = now + self.ttl_seconds

        # Serialize SSE events
        sse_json = None
        if response.sse_events:
            sse_json = json.dumps(response.sse_events)

        # Calculate size
        size = 0
        if response.response_body:
            size += len(response.response_body)
        if sse_json:
            size += len(sse_json.encode("utf-8"))

        self._conn.execute(
            "INSERT OR REPLACE INTO responses "
            "(key, model, response_body, sse_events, input_tokens, output_tokens, "
            "cache_creation_input_tokens, cache_read_input_tokens, "
            "created_at, expires_at, size_bytes, last_accessed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                response.model,
                response.response_body,
                sse_json,
                response.input_tokens,
                response.output_tokens,
                response.cache_creation_input_tokens,
                response.cache_read_input_tokens,
                now,
                expires_at,
                size,
                now,
            ),
        )
        self._conn.commit()

        # Evict expired entries periodically
        self._conn.execute("DELETE FROM responses WHERE expires_at <= ?", (now,))

        # Evict by size if needed
        await self._evict_by_size()

        self._conn.commit()

    async def _evict_by_size(self) -> None:
        """Evict oldest entries if total cache size exceeds max."""
        if not self._conn:
            return

        cursor = self._conn.execute("SELECT SUM(size_bytes) FROM responses")
        total = cursor.fetchone()[0] or 0

        if total <= self.max_size_bytes:
            return

        # Delete oldest entries until under limit
        while total > self.max_size_bytes:
            cursor = self._conn.execute(
                "SELECT key, size_bytes FROM responses "
                "ORDER BY last_accessed_at ASC LIMIT 10"
            )
            rows = cursor.fetchall()
            if not rows:
                break
            for row_key, row_size in rows:
                self._conn.execute("DELETE FROM responses WHERE key = ?", (row_key,))
                total -= row_size

        logger.debug("Cache eviction complete, total size: %d bytes", total)

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if not self._conn:
            return {"status": "not initialized"}

        now = time.time()
        cursor = self._conn.execute(
            "SELECT COUNT(*), SUM(size_bytes), "
            "SUM(input_tokens), SUM(output_tokens) "
            "FROM responses WHERE expires_at > ?",
            (now,),
        )
        count, total_size, total_in, total_out = cursor.fetchone()

        return {
            "entries": count or 0,
            "total_size_bytes": total_size or 0,
            "total_size_mb": round((total_size or 0) / (1024 * 1024), 2),
            "max_size_mb": round(self.max_size_bytes / (1024 * 1024), 2),
            "total_cached_input_tokens": total_in or 0,
            "total_cached_output_tokens": total_out or 0,
            "ttl_seconds": self.ttl_seconds,
            "db_path": str(self._db_path),
        }
