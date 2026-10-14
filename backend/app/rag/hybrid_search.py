import logging
from typing import List, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.rag.bm25_index import BM25Index
from app.rag.vector_store import get_vectorstore

logger = logging.getLogger(__name__)


def _doc_key(doc: Document) -> str:
    """跨路去重用的片段标识。

    **不要用 `page_content[:200]`。** 同一文档相邻分片的开头高度相似
    （「第三章 劳动合同的订立……」），截断后会被判成同一片段而互相覆盖；
    反过来，两份内容相同但来源不同的文档也会被合并，于是检索结果里少一条，
    而 `doc_map[key] = doc` 会让后写入的那条顶掉先写入的那条 ——
    丢的是**先召回的那一路的证据**。

    优先用入库时写进 metadata 的 chunk_id（见 rag/loader.py）；没有它时
    退回到 (source, 页码, 内容摘要) 组合，仍然比纯内容前缀稳。
    """
    metadata = doc.metadata or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"
    source = metadata.get("source", "")
    page = metadata.get("page")
    digest = hash(doc.page_content)  # 内容相同的两片视为同一片段
    return f"meta:{source}:{page}:{digest}"


def _vector_search(query: str, top_k: int) -> List[Tuple[Document, float]]:
    if top_k <= 0:
        # 0 是合法取值（「向量这一路不要候选」），所以在这里短路，而不是构造一个
        # k=0 的 retriever —— chromadb 接受构造，但 invoke 时会抛
        # TypeError: Number of requested results 0, cannot be negative, or zero.
        return []
    vectorstore = get_vectorstore()
    if vectorstore is None:
        raise ValueError("No persisted vector store found")

    # 用带分数的检索，而不是 as_retriever().invoke()：后者只给 Document，
    # 相似度得由调用方自己编 —— 旧实现就编了一个 `1.0 - i * 0.05` 的名次占位分，
    # 然后让它一路流到 search_documents 的输出里展示成「相关度」。模型看到
    # 「相关度 0.03」会以为资料不相关而拒答，这是会主动降低回答质量的 bug。
    pairs: List[Tuple[Document, float]] = vectorstore.similarity_search_with_relevance_scores(
        query, k=top_k
    )
    # chroma 的 relevance score 理论上是 [0, 1]，但不同版本/距离度量下可能越界或为负
    # （余弦距离转相似度时尤其如此）。夹到 [0, 1] 只为让它可读，不影响排序 ——
    # 排序完全由 similarity_search 返回的顺序决定。
    return [(doc, max(0.0, min(1.0, float(score)))) for doc, score in pairs]


def _rrf_fuse(
    vector_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    k: int = 60,
    top_k: int = 10,
) -> List[Tuple[Document, float]]:
    if top_k <= 0:
        # 与 _vector_search / BM25Index.search 统一：非正数即不要结果。负数尤其不能
        # 靠切片表达 —— [:-1] 会返回「除最后一个之外的全部」，既不空也不报错。
        return []
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(vector_results):
        key = _doc_key(doc)
        doc_map[key] = doc
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)

    for rank, (doc, _) in enumerate(bm25_results):
        key = _doc_key(doc)
        doc_map[key] = doc
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_map[key], score) for key, score in ranked]


def _rerank(
    query: str,
    candidates: List[Tuple[Document, float]],
    top_k: int,
) -> List[Tuple[Document, float]]:
    """用 Cross-Encoder 重排候选顺序；任何失败都退回 RRF 原序。

    **精排是增强，不是依赖。** 模型没下载、内存不够、CPU 上加载超时——任何一种
    都不该让整条检索路径失败。所以这里捕获全部异常并原样返回 RRF 结果：最坏情况
    是「精排没生效」，而不是「搜不出来」。

    **只改顺序，不改分数语义。** CrossEncoder 打的是 logit，量纲随模型而异
    （常见 -10 ~ +10），把它当「相关度」展示给 LLM 会让模型误判资料是否相关 ——
    那正是当年伪造相似度带来的问题。返回的仍是 RRF 融合分。
    """
    from app.rag.reranker import get_reranker

    if not candidates:
        return []

    try:
        reranked = get_reranker().rerank(query, candidates, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 — 见上方docstring：降级而非失败
        logger.warning(
            "精排失败，本次退回 RRF 融合顺序：%s: %s",
            type(exc).__name__,
            exc,
        )
        return candidates[:top_k]

    if not reranked:
        # 精排返回空不该静默变成「没有结果」——那比不精排糟得多。
        return candidates[:top_k]

    # 按文档标识回填 RRF 分，保持分数语义不变
    fusion_scores = {_doc_key(doc): score for doc, score in candidates}
    return [(doc, fusion_scores.get(_doc_key(doc), 0.0)) for doc, _ in reranked]


def hybrid_search(
    query: str,
    bm25_index: BM25Index,
    vector_top_k: int | None = None,
    bm25_top_k: int | None = None,
    final_top_k: int | None = None,
) -> List[Tuple[Document, float]]:
    """两路各召回 hybrid_top_k 个候选，RRF 融合后按需精排，返回 top_k 个。

    精排开着时，融合阶段取`rerank_candidates` 个（默认 10）送进 Cross-Encoder，
    再截到 `top_k`（默认 4）；关掉时融合阶段直接产出 `top_k` 个，少一次重排开销。
    """
    # 用 is None 而不是 `or`：0 是合法取值（"这一路不要候选"），不该被当成"没传"
    if vector_top_k is None:
        vector_top_k = settings.hybrid_top_k
    if bm25_top_k is None:
        bm25_top_k = settings.hybrid_top_k
    if final_top_k is None:
        final_top_k = settings.top_k
    if final_top_k <= 0:
        return []

    use_reranker = settings.enable_reranker
    fusion_top_k = max(settings.rerank_candidates, final_top_k) if use_reranker else final_top_k

    vector_results = _vector_search(query, top_k=vector_top_k)
    bm25_results = bm25_index.search(query, top_k=bm25_top_k)
    fused = _rrf_fuse(vector_results, bm25_results, top_k=fusion_top_k)

    if not use_reranker:
        return fused[:final_top_k]

    return _rerank(query, fused, top_k=final_top_k)
