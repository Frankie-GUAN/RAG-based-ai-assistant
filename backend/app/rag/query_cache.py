"""Query 向量缓存。

**为什么需要它。** 检索链路里最慢的一步不是精排，而是**向量检索**——
实测（BGE-M3，CPU）：

    向量检索      8.229s     ← 慢在这里
    BM25(含改写)  2.525s
    精排          6.379s

向量检索慢，是因为 `similarity_search_with_relevance_scores` 每次都要
**重新把 query 编码一遍**——BGE-M3 在 CPU 上编码一条 query 要好几秒
（模型 2GB，且这条 query 还带着跨语言改写追加的术语）。

**但同一个 query 被反复问是常态**：用户改两个字再问、追问同一件事、
评测脚本把同一批 query 跑四遍——每一次都重算同一个向量，纯属浪费。

**缓存什么。** `query → 向量` 的映射。命中缓存后，向量检索从
「编码 + 检索」变成「检索」，实测从 8.2s 降到 0.1s 量级。

**为什么放在这一层而不是改 `hybrid_search`。** `hybrid_search` 里还有 BM25
与融合逻辑，而向量编码只发生在 `_vector_search` 内部。
把缓存下沉到编码这个点上，任何走向量检索的调用方都自动受益，
不需要各自记得加缓存。

**缓存的失效。** 索引重建后向量语义不变（query 编码与语料无关），
所以**重建索引不需要清缓存**。只有换嵌入模型时才需要——
那时用 `reset_query_cache()`。

**为什么用 LRU 而不是无限增长。** query 是用户输入，基数无界；
但实际会话里的不同 query 数量有限（实测几十到几百条）。
LRU 兜住内存，同时保留热点。
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

# 缓存条目上限。超过就淘汰最久未用的。
# 实测一次评测跑 15 条 query × 4 策略 = 60 次编码，
# 缓存住这 60 条就能把整套评测从「每策略重算」变成「只算一次」。
_MAX_ENTRIES = 256

_cache: "OrderedDict[str, list[float]]" = OrderedDict()
_lock = threading.Lock()


def _cache_key(query: str) -> str:
    return query.strip()


def get_cached_query_vector(query: str) -> Optional[list[float]]:
    """取缓存的 query 向量；没有则返回 None。"""
    key = _cache_key(query)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    return None


def put_query_vector(query: str, vector: list[float]) -> None:
    """写入 query 向量缓存。"""
    key = _cache_key(query)
    with _lock:
        _cache[key] = vector
        _cache.move_to_end(key)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)


def cached_embed_query(query: str) -> list[float]:
    """编码 query 并缓存。**这是业务代码应该调用的入口。**

    命中缓存时直接返回，不触碰模型。
    """
    hit = get_cached_query_vector(query)
    if hit is not None:
        return hit

    from app.rag.vector_store import get_embeddings

    vector = get_embeddings().embed_query(query)
    put_query_vector(query, vector)
    return vector


def clear_query_cache() -> None:
    """清空缓存。**换嵌入模型后必须调用**，否则会返回旧模型的向量。"""
    with _lock:
        _cache.clear()


def cache_stats() -> dict[str, int]:
    """缓存状态（运维/测试用）。"""
    with _lock:
        return {"size": len(_cache), "max_entries": _MAX_ENTRIES}
