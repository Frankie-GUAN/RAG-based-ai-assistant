import logging
import os
import pickle
import threading
import time
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings

logger = logging.getLogger(__name__)

# 加载失败后的重试间隔。见 get_bm25_index()。
_RETRY_INTERVAL_SECONDS = 30.0


class BM25Index:
    def __init__(self):
        # (documents, index) 作为**一对**整体替换，search 永远读不到长度不匹配的组合
        self._pair: tuple[List[Document], BM25Okapi | None] = ([], None)
        self._write_lock = threading.Lock()
        self._index_path = settings.data_dir / "bm25_index.pkl"
        # 磁盘上存在一份我们读不出来的索引。此时内存里是空的，而那份文件是**唯一**的
        # 一份 —— 绝不能让 add() 拿这个空白基础把它覆盖掉。
        self._load_failed = False
        self._last_load_attempt = 0.0

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
            if self._load_failed:
                # 磁盘那份额外的索引读不出来（且仓库里没有从 Chroma 重建的路径），
                # 它此刻是唯一副本。用空白基础落盘会把 2196 篇变成新上传的这几篇 ——
                # 这是不可逆的。宁可用例失败（响的），也不能静默毁数据。
                raise RuntimeError(
                    "BM25 索引文件存在但无法加载，拒绝以空白基础覆盖落盘以免丢失数据："
                    f"{self._index_path}"
                )
            self.build([*self._pair[0], *chunks])
            self.save()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        documents, index = self._pair  # 快照：整个 search 用这一对
        if index is None or top_k <= 0:
            # 负数不能靠切片表达：sorted(...)[:-1] 会返回「除最后一个之外的全部」，
            # 既不空也不报错。与 _vector_search 统一成「非正数即不要候选」。
            return []
        tokens = self._tokenize(query)
        scores = index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        max_score = scores[ranked[0][0]] if ranked and scores[ranked[0][0]] > 0 else 1.0
        return [(documents[i], score / max_score) for i, score in ranked if score > 0]

    def save(self) -> None:
        """先写同目录下的临时文件，再 os.replace() 换上去。

        临时文件名带 pid：固定名字会让两个进程互相 rename 对方写了一半的文件。

        关于原子性：os.replace 在 POSIX 上是原子的；在 Win32 上它同样是「全有或全无」，
        但若目标文件正被另一个进程（或杀软/备份）打开着，它会抛 PermissionError ——
        此时**旧索引仍然完好**，这正是先写临时文件的意义，只是本次落盘失败。单进程内
        不会发生（get_bm25_index 在同一把锁内 load，且赋值前文件已关闭）；多 worker
        共享 data/ 时才会，属已知待办。实测未加原子写时，另一个进程 352 次读里有
        325 次读到半截 pickle。
        """
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        documents, index = self._pair
        tmp_path = self._index_path.parent / f"{self._index_path.name}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump({"documents": documents, "index": index}, f)
        os.replace(tmp_path, self._index_path)

    def load(self) -> bool:
        """索引加载不了时返回 False 且置 _load_failed，不抛异常。

        这里的 except 刻意宽到 Exception，而不是只捕字节级损坏（UnpicklingError /
        EOFError / OSError）：pickle 里嵌着**活的** BM25Okapi 实例与 Document 对象，
        所以「加载不了」的成因不止文件损坏 —— rank_bm25 升级、类改名或移动、dict
        结构变更分别会抛 AttributeError / KeyError，那些是窄元组接不住的。

        宽容的代价是：瞬时故障（MemoryError、Win32 的文件共享冲突）与真损坏从此无法
        区分，两者都会走到这里。所以这份宽容必须由调用方配保险 ——
        get_bm25_index() 会重试，add() 在 _load_failed 时拒绝落盘。
        """
        self._last_load_attempt = time.monotonic()
        try:
            if not self._index_path.exists():
                self._load_failed = False  # 尚无索引文件是全新状态，不是失败
                return False
            with open(self._index_path, "rb") as f:
                data = pickle.load(f)
            self._pair = (data["documents"], data["index"])
        except Exception:
            logger.exception("BM25 索引加载失败，已忽略：%s", self._index_path)
            self._load_failed = True
            return False
        self._load_failed = False
        return True

    @property
    def document_count(self) -> int:
        return len(self._pair[0])


_bm25: BM25Index | None = None
_bm25_lock = threading.Lock()


def get_bm25_index() -> BM25Index:
    """BM25 索引的进程内单例，懒加载磁盘上的 pickle。

    加载失败时不把那个空索引就此固化：距上次尝试超过 _RETRY_INTERVAL_SECONDS 就再试
    一次。否则一次瞬时故障会让 BM25 整条路径在整个进程生命周期内都退化成空，而且
    下一次上传会因为 add() 抛错而失败（这是刻意的，见 add()）。
    """
    global _bm25
    if _bm25 is None:
        with _bm25_lock:
            if _bm25 is None:
                index = BM25Index()
                index.load()
                _bm25 = index
        return _bm25

    if _bm25._load_failed:
        with _bm25_lock:
            if _bm25._load_failed and (
                time.monotonic() - _bm25._last_load_attempt > _RETRY_INTERVAL_SECONDS
            ):
                _bm25.load()
    return _bm25
