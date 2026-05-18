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
