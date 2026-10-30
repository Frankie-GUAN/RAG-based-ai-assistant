"""语义缓存层（三层记忆的第三层）。

对外只有 `get_semantic_cache()` 与 `CachedAnswer`。
换后端（Chroma → Redis Stack → pgvector）只改 `semantic.py` 里的实现。
"""
from app.cache.semantic import (
    CachedAnswer,
    SemanticCacheBackend,
    get_semantic_cache,
    reset_semantic_cache,
)

__all__ = [
    "CachedAnswer",
    "SemanticCacheBackend",
    "get_semantic_cache",
    "reset_semantic_cache",
]
