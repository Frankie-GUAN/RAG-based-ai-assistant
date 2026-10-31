"""评测策略构造：把四种检索路径统一成 `(query, k) -> [(Document, score)]`。

抽这一层的原因：`app/rag/hybrid_search.py` 里的 `_vector_search` / `_rrf_fuse` / `_rerank`
都是下划线私有的，但它们恰恰是评测需要对比的各个阶段。
与其复制粘贴一份实现（很快会与真实代码分叉），
不如在这里**直接调用私有函数**——它们是同一个包内的协作模块，
私有命名只是约定，不是边界。

`_rrf_fuse` 与 `_rerank` 的参数默认值取自 `settings`，
与生产路径完全一致，所以评测结果对生产有参考价值。
"""
from __future__ import annotations

from typing import Callable

from langchain_core.documents import Document

from app.config import settings
from app.rag.bm25_index import get_bm25_index
from app.rag.hybrid_search import _rerank, _rrf_fuse, _vector_search
from app.rag.vector_store import get_vectorstore


def _vector_only(query: str, k: int) -> list[tuple[Document, float]]:
    """仅向量召回。"""
    return _vector_search(query, top_k=k)


def _bm25_only(query: str, k: int) -> list[tuple[Document, float]]:
    """仅 BM25 稀疏召回。

    **走 `hybrid_search` 的改写路径**，而不是直接调 `bm25_index.search`。
    直接调会绕过生产链路里的跨语言改写（`_expand_for_sparse`），
    那样测出来的就不是线上真实表现——
    实测踩过这个坑：改写开关打开与关闭，评测结果完全一样，
    因为评测压根没走那条路。
    """
    from app.rag.hybrid_search import _expand_for_sparse
    return get_bm25_index().search(_expand_for_sparse(query), top_k=k)


def _rrf_only(query: str, k: int) -> list[tuple[Document, float]]:
    """两路 RRF 融合，不精排。"""
    candidates = settings.rerank_candidates if settings.enable_reranker else k
    vector = _vector_search(query, top_k=settings.hybrid_top_k)
    bm25 = get_bm25_index().search(query, top_k=settings.hybrid_top_k)
    return _rrf_fuse(vector, bm25, top_k=max(candidates, k))


def _rrf_rerank(query: str, k: int) -> list[tuple[Document, float]]:
    """生产默认路径：RRF 融合 + Cross-Encoder 精排。"""
    vector = _vector_search(query, top_k=settings.hybrid_top_k)
    bm25 = get_bm25_index().search(query, top_k=settings.hybrid_top_k)
    fused = _rrf_fuse(vector, bm25, top_k=max(settings.rerank_candidates, k))
    return _rerank(query, fused, top_k=k)


# 策略注册表。键名即报告中显示的名字。
STRATEGIES: dict[str, Callable[[str, int], list[tuple[Document, float]]]] = {
    "vector": _vector_only,
    "bm25": _bm25_only,
    "rrf": _rrf_only,
    "rrf_rerank": _rrf_rerank,
}


def build_strategy(name: str) -> Callable[[str, int], list[tuple[Document, float]]]:
    """按名字取策略。未知名字直接报错——静默回退到默认策略会毁掉评测可信度。"""
    if name not in STRATEGIES:
        raise KeyError(f"未知策略 {name!r}，可选：{', '.join(STRATEGIES)}")
    return STRATEGIES[name]


def corpus_stats() -> dict[str, int]:
    """语料规模，用于报告头。"""
    vectorstore = get_vectorstore()
    vector_count = vectorstore._collection.count() if vectorstore else 0
    # document_count 是 @property，不是方法
    bm25_count = get_bm25_index().document_count
    return {"vector": vector_count, "bm25": bm25_count}
