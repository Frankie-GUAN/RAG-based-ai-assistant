import logging
import os
import pickle
import threading
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings

logger = logging.getLogger(__name__)


class BM25Index:
    def __init__(self):
        # (documents, index) 作为**一对**整体替换，search 永远读不到长度不匹配的组合
        self._pair: tuple[List[Document], BM25Okapi | None] = ([], None)
        self._write_lock = threading.Lock()
        self._index_path = settings.data_dir / "bm25_index.pkl"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re

        return [t.lower() for t in re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text)]

    def build(self, documents: List[Document]) -> None:
        corpus = [self._tokenize(doc.page_content) for doc in documents]
        # any(corpus) 而不是 corpus：`corpus = [[]]` 是**真值**，而 BM25Okapi._calc_idf
        # 会做 idf_sum / len(self.idf) —— 语料整体分不出词就是 ZeroDivisionError。
        # 空 PDF 页、纯空白 .txt 都会走到这里，而 upload 接口不做空内容过滤。
        index = BM25Okapi(corpus) if any(corpus) else None
        self._pair = (list(documents), index)  # 原子替换

    def add(self, chunks: List[Document]) -> None:
        """把新片段并入索引并落盘。两条 BM25 建索引路径共用此方法。

        注意它统一的只是 BM25 这一侧：Chroma 由 vector_store.add_documents() 独立写入，
        两者之间没有事务 —— 中途失败会留下「Chroma 有、BM25 无」的分叉。跨存储的原子性
        是 Task 6/7 的待办，不在本方法职责内。
        """
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
        """先写同目录下的临时文件，再 os.replace() 换上去。

        os.replace 在 POSIX 与 Win32 上都是原子的，所以并发的读者要么看到旧文件、
        要么看到新文件，不会读到写了一半的 pickle（实测未加原子写时 352 次写里
        有 325 次被另一个进程读成 UnpicklingError / EOFError）。
        """
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        documents, index = self._pair
        tmp_path = self._index_path.parent / (self._index_path.name + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump({"documents": documents, "index": index}, f)
        os.replace(tmp_path, self._index_path)

    def load(self) -> bool:
        """索引加载不了时返回 False，不抛异常。

        这里的 except 刻意宽到 Exception，而不是只捕字节级损坏（UnpicklingError /
        EOFError / OSError）：pickle 里嵌着**活的** BM25Okapi 实例与 Document 对象，
        所以「加载不了」的成因不止文件损坏 —— rank_bm25 升级、类改名或移动、dict
        结构变更分别会抛 AttributeError / KeyError，那些是窄元组接不住的。

        索引是可重建的缓存，宽容没有代价；而 get_bm25_index() 现在跑在 lifespan 里，
        抛出去就不是「某次检索失败」，而是整个 API 起不来。
        """
        if not self._index_path.exists():
            return False
        try:
            with open(self._index_path, "rb") as f:
                data = pickle.load(f)
            self._pair = (data["documents"], data["index"])
        except Exception:
            logger.exception("BM25 索引加载失败，已忽略：%s", self._index_path)
            return False
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
