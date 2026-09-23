from typing import List, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.rag.bm25_index import BM25Index
from app.rag.vector_store import get_retriever


def _vector_search(query: str, top_k: int) -> List[Tuple[Document, float]]:
    if top_k <= 0:
        # 0 是合法取值（「向量这一路不要候选」），所以在这里短路，而不是构造一个
        # k=0 的 retriever —— chromadb 接受构造，但 invoke 时会抛
        # TypeError: Number of requested results 0, cannot be negative, or zero.
        return []
    retriever = get_retriever(top_k)          # ← 此前这个 k 被忽略
    docs = retriever.invoke(query)
    # 这里是名次占位分，不是相似度。RRF 只用名次，故不影响融合结果，
    # 但不要把它当相关度读。
    return [(doc, 1.0 - i * 0.05) for i, doc in enumerate(docs)][:top_k]


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
        key = doc.page_content[:200]
        doc_map[key] = doc
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)

    for rank, (doc, _) in enumerate(bm25_results):
        key = doc.page_content[:200]
        doc_map[key] = doc
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_map[key], score) for key, score in ranked]


def hybrid_search(
    query: str,
    bm25_index: BM25Index,
    vector_top_k: int | None = None,
    bm25_top_k: int | None = None,
    final_top_k: int | None = None,
) -> List[Tuple[Document, float]]:
    """两路各召回 hybrid_top_k 个候选，RRF 融合后返回 top_k 个。"""
    # 用 is None 而不是 `or`：0 是合法取值（"这一路不要候选"），不该被当成"没传"
    if vector_top_k is None:
        vector_top_k = settings.hybrid_top_k
    if bm25_top_k is None:
        bm25_top_k = settings.hybrid_top_k
    if final_top_k is None:
        final_top_k = settings.top_k

    vector_results = _vector_search(query, top_k=vector_top_k)
    bm25_results = bm25_index.search(query, top_k=bm25_top_k)
    return _rrf_fuse(vector_results, bm25_results, top_k=final_top_k)
