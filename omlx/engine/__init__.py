# SPDX-License-Identifier: Apache-2.0
"""
Engine abstraction for oMLX inference.

Provides multiple engine implementations:
- BatchedEngine: Continuous batching for multiple concurrent users
- VLMBatchedEngine: Vision-language model engine with image support
- EmbeddingEngine: Batch embedding generation using mlx-embeddings
- RerankerEngine: Document reranking using SequenceClassification models

Also re-exports core engine components for backwards compatibility.

Note: engine_core imports are lazy to avoid pulling in MLX when running
in proxy-only mode (e.g. ``omlx proxy``).
"""

import importlib as _importlib

from .base import BaseEngine, BaseNonStreamingEngine, GenerationOutput
from .batched import BatchedEngine
from .embedding import EmbeddingEngine
from .reranker import RerankerEngine
from .stt import STTEngine
from .sts import STSEngine
from .tts import TTSEngine
from .vlm import VLMBatchedEngine

__all__ = [
    "BaseEngine",
    "BaseNonStreamingEngine",
    "GenerationOutput",
    "BatchedEngine",
    "VLMBatchedEngine",
    "EmbeddingEngine",
    "RerankerEngine",
    "STTEngine",
    "STSEngine",
    "TTSEngine",
    # Core engine components (lazy)
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
]


def __getattr__(name: str):
    """Lazy import for engine_core symbols to avoid pulling in MLX at import time."""
    if name in ("EngineCore", "AsyncEngineCore", "EngineConfig"):
        mod = _importlib.import_module("..engine_core", __name__)
        val = getattr(mod, name)
        # Cache on the module so subsequent access is fast
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
