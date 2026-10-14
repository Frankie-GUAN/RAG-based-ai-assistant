import logging
import os
import pickle
import threading
import time
from typing import List, Tuple

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings

logger = logging.getLogger(__name__)

# 加载失败后的重试间隔。见 get_bm25_index()。
_RETRY_INTERVAL_SECONDS = 30.0

# jieba 首次加载会往 stderr 打「Building prefix dict… / Loading model cost…」。
# 那是一次性的库初始化噪声，与应用日志混在一起只会干扰排查。
jieba.setLogLevel(logging.WARNING)

# 刻意保持很小的停用词表：只收纯功能词。BM25 的 IDF 本来就靠「在多数文档里出现」
# 压制高频词，这份表只需挡掉那些几乎不携带区分度的词。收录「不」「没有」这类
# 有语义否定作用的词会误伤检索结果。
_STOPWORDS = frozenset("""
的 了 和 是 就 都 而 及 与 着 或 一个 没有 我们 你们 他们 她们 它们
这个 那个 之 也 在 有 不 人 你 我 他 她 它 但 但是 因为 所以 如果 虽然
可以 需要 什么 怎么 这样 那样 吗 呢 吧 啊 呀 哦 嗯 请 帮 一下 一些
时候 觉得 应该 可能 以及 或者 关于 对于 通过 进行 由于 因此
""".split())


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
        """中文用 jieba 切词，英文数字按原样保留。

        **为什么不用单字切分。** 旧实现用 `re.findall(r"[一-鿿]|...")` 把中文切成
        单字。那会让「劳动合同」变成「劳/动/合/同」：查询侧若也按单字切，双方都退化
        成「字的命中」，「的/了/是」这类高频字主导打分，而「劳动/合同」这类真正携带
        区分度的二元词被拆散 —— BM25 最擅长的精确匹配正好是被这一步毁掉的。

        `lcut_for_search`（搜索模式）比 `cut` 更适合检索：它同时产出较长的复合词
        与其组成部分，既提高命中又不漏召回。

        **磁盘上的旧索引与本函数不兼容。** pickle 里存的是**已分词**的语料，改分词器
        不会让旧文件失效、只会让它变成「新词切法查旧字粒度」的静默错误（查「劳动合同」
        在按单字索引的旧文档上命中不了）。换分词器后必须跑
        `python -m app.rag.bm25_index rebuild` 从 Chroma 全量重建。
        """
        tokens: List[str] = []
        for token in jieba.lcut_for_search(text.lower()):
            token = token.strip()
            if not token or token in _STOPWORDS:
                continue
            # 丢掉纯标点/空白：jieba 会把它们单独切出来，而 BM25 不该为它们分配权重。
            if not any(ch.isalnum() for ch in token):
                continue
            tokens.append(token)
        return tokens

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

        磁盘那份索引读不出来时，save() 会拒绝落盘。判断放在 save() 而不是这里，是因为
        build()+save() 才是覆盖磁盘的那条路径 —— 不管谁调，都得过 save() 这一关。
        """
        with self._write_lock:
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

    def remove_source(self, source: str) -> int:
        """把某个来源的全部片段移出索引，返回移除条数。

        删除文档时必须走这里而不是「重建整个索引」：重建的输入是 Chroma，
        而 Chroma 与 BM25 之间没有事务（见 add 的说明），此刻两边可能已经不一致，
        拿 Chroma 覆盖 BM25 会把 BM25 独有的片段悄悄抹掉。

        与 add 一样受 _write_lock 保护，且**在降级状态下拒绝执行**：
        此时内存里是空索引，「删掉某个来源」会变成「清空全部」。
        """
        with self._write_lock:
            documents, index = self._pair
            kept = [doc for doc in documents if (doc.metadata or {}).get("source") != source]
            removed = len(documents) - len(kept)
            if removed == 0:
                return 0
            self.build(kept)
            self.save()
            return removed

    def save(self, force: bool = False) -> None:
        """先写同目录下的临时文件，再 os.replace() 换上去。

        _load_failed 时**拒绝落盘**：那份读不出来的索引是唯一副本，用内存里这个空索引
        覆盖它会把 2196 篇变成新上传的这几篇，而仓库里目前没有从 Chroma 重建的路径，
        不可逆。这里是唯一的收口点 —— 刻意重建（调用方已确保 documents 是完整集合）
        时传 force=True，它同时会清掉 _load_failed。

        临时文件名带 pid：固定名字会让两个进程互相 rename 对方写了一半的文件。

        关于原子性：os.replace 在 POSIX 上是原子的；在 Win32 上它同样是「全有或全无」，
        但若目标文件正被另一个进程（或杀软/备份）打开着，它会抛 PermissionError ——
        此时**旧索引仍然完好**，这正是先写临时文件的意义，只是本次落盘失败。单进程内
        不可达：load() 是在把 (documents, index) 那一对发布出去之后才可能让别的线程
        观察到 _load_failed=False 的，所以 add() 不会在空白基础上落盘。多 worker 共享
        data/ 时才会，属已知待办。实测未加原子写时，另一个进程 352 次读里有 325 次
        读到半截 pickle。
        """
        if self._load_failed and not force:
            raise RuntimeError(
                "BM25 索引文件存在但无法加载，拒绝以空白基础覆盖落盘以免丢失数据："
                f"{self._index_path}。恢复方式：修好或删除该文件（删除后重试会把它当作"
                f"全新库），或用 force=True 表明你已重建出完整文档集。"
            )

        settings.data_dir.mkdir(parents=True, exist_ok=True)
        documents, index = self._pair
        tmp_path = self._index_path.parent / f"{self._index_path.name}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump({"documents": documents, "index": index}, f)
        os.replace(tmp_path, self._index_path)
        if force:
            self._load_failed = False  # 磁盘已被完整集合取代，自愈完成

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


def rebuild_from_chroma() -> int:
    """从 Chroma 全量重建 BM25 索引，返回重建的片段数。

    这条路径的存在理由：BM25 索引是**缓存**，Chroma 才是检索事实的来源。此前仓库
    里没有从 Chroma 重建的路径，于是 `BM25 索引是可重建的缓存` 只是句空头支票 ——
    索引真坏掉时，除了手工 pickle 没有别的办法把它救回来。

    两种必须重建的情形：

    1. **分词器变更**。索引里存的是已分词的语料，改 `_tokenize` 不会让旧 pickle
       失效，只会让它变成「新切法查旧粒度」的静默错误。
    2. **索引缺失或损坏**。此时 Chroma 里的文档仍然完好，应当能从它恢复。

    `save(force=True)`：重建意味着内存里的集合就是完整集合，因此可以合法覆盖磁盘，
    并顺带清掉 `_load_failed` 标志（这是该标志唯一的自愈入口）。
    """
    # 局部导入：vector_store 会 import 配置与 langchain_community，而 bm25_index
    # 被检索栈的其他模块导入，放成顶层容易把导入顺序绑死。
    from app.rag.vector_store import get_vectorstore

    vectorstore = get_vectorstore()
    if vectorstore is None:
        raise RuntimeError("Chroma 索引不存在，无法重建（先上传文档或确认 data/chroma）")

    raw = vectorstore.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(raw.get("documents") or [], raw.get("metadatas") or [])
        if text
    ]
    if not documents:
        raise RuntimeError("Chroma 中没有可重建的片段（返回 0 条）")

    index = get_bm25_index()
    index.build(documents)
    index.save(force=True)
    logger.info("BM25 索引已从 Chroma 重建：%d 个片段", len(documents))
    return len(documents)


def main() -> None:
    """运维入口：python -m app.rag.bm25_index rebuild"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="BM25 索引维护")
    parser.add_argument("command", choices=["rebuild"], help="rebuild = 从 Chroma 全量重建")
    args = parser.parse_args()

    if args.command == "rebuild":
        count = rebuild_from_chroma()
        print(f"已重建 {count} 个片段")
        return

    sys.exit(1)


if __name__ == "__main__":
    main()
