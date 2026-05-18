from typing import List, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.rag.bm25_index import BM25Index
from app.rag.vector_store import get_retriever


def _vector_search(query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return [(doc, 1.0 - i * 0.05) for i, doc in enumerate(docs)][:top_k]


def _rrf_fuse(
    vector_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    k: int = 60,
    top_k: int = 10,
) -> List[Tuple[Document, float]]:
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
    vector_top_k: int = 10,
    bm25_top_k: int = 10,
    final_top_k: int = 10,
) -> List[Tuple[Document, float]]:
    vector_results = _vector_search(query, top_k=vector_top_k)
    bm25_results = bm25_index.search(query, top_k=bm25_top_k)
    return _rrf_fuse(vector_results, bm25_results, top_k=final_top_k)
