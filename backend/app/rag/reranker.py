import threading
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config import settings


class Reranker:
    def __init__(self, model_name: str | None = None):
        self._model = CrossEncoder(model_name or settings.reranker_model_path)

    def rerank(
        self, query: str, candidates: List[Tuple[Document, float]], top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc, _ in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(score)) for (doc, _), score in ranked[:top_k]]


_reranker: "Reranker | None" = None
_reranker_lock = threading.Lock()


def get_reranker() -> "Reranker":
    """Cross-Encoder 的进程内单例。

    本轮不接入查询链路（CPU 上精排 10 个候选约需 1~3 秒，会顶掉「首 token < 800ms」
    的验收目标）。此处只把底座准备好，接入与检索指标（Hit Rate / NDCG）一起留到 P1。
    """
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = Reranker()
    return _reranker
