import pickle
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings


class BM25Index:
    def __init__(self):
        self._index: BM25Okapi | None = None
        self._documents: List[Document] = []
        self._index_path = settings.data_dir / "bm25_index.pkl"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re

        return [t.lower() for t in re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text)]

    def build(self, documents: List[Document]) -> None:
        self._documents = documents
        corpus = [self._tokenize(doc.page_content) for doc in documents]
        self._index = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        if self._index is None:
            return []
        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        max_score = scores[ranked[0][0]] if ranked and scores[ranked[0][0]] > 0 else 1.0
        return [(self._documents[i], score / max_score) for i, score in ranked if score > 0]

    def save(self) -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "wb") as f:
            pickle.dump({"documents": self._documents, "index": self._index}, f)

    def load(self) -> bool:
        if not self._index_path.exists():
            return False
        with open(self._index_path, "rb") as f:
            data = pickle.load(f)
        self._documents = data["documents"]
        self._index = data["index"]
        return True

    @property
    def document_count(self) -> int:
        return len(self._documents)
