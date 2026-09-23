import pickle
import threading
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings


class BM25Index:
    def __init__(self):
        # (documents, index) 作为**一对**整体替换，search 永远读不到长度不匹配的组合
        self._pair: tuple[List[Document], BM25Okapi | None] = ([], None)
        self._write_lock = threading.Lock()
        self._index_path = settings.data_dir / "bm25_index.pkl"

    @property
    def _documents(self) -> List[Document]:
        return self._pair[0]

    @property
    def _index(self) -> BM25Okapi | None:
        return self._pair[1]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re

        return [t.lower() for t in re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text)]

    def build(self, documents: List[Document]) -> None:
        corpus = [self._tokenize(doc.page_content) for doc in documents]
        index = BM25Okapi(corpus) if corpus else None
        self._pair = (list(documents), index)  # 原子替换

    def add(self, chunks: List[Document]) -> None:
        """把新片段并入索引并落盘。两条建索引路径共用此方法，避免 Chroma 与 BM25 分叉。"""
        with self._write_lock:
            self.build([*self._pair[0], *chunks])
            self.save()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        documents, index = self._pair  # 快照：整个 search 用这一对
        if index is None:
            return []
        tokens = self._tokenize(query)
        scores = index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        max_score = scores[ranked[0][0]] if ranked and scores[ranked[0][0]] > 0 else 1.0
        return [(documents[i], score / max_score) for i, score in ranked if score > 0]

    def save(self) -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        documents, index = self._pair
        with open(self._index_path, "wb") as f:
            pickle.dump({"documents": documents, "index": index}, f)

    def load(self) -> bool:
        if not self._index_path.exists():
            return False
        with open(self._index_path, "rb") as f:
            data = pickle.load(f)
        self._pair = (data["documents"], data["index"])
        return True

    @property
    def document_count(self) -> int:
        return len(self._pair[0])


_bm25: BM25Index | None = None
_bm25_lock = threading.Lock()


def get_bm25_index() -> BM25Index:
    """BM25 索引的进程内单例，懒加载磁盘上的 pickle。"""
    global _bm25
    if _bm25 is None:
        with _bm25_lock:
            if _bm25 is None:
                index = BM25Index()
                index.load()
                _bm25 = index
    return _bm25
