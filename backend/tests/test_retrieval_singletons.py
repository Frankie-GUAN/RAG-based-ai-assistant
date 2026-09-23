"""检索链路单例化测试。

最关键的一条是 test_embeddings_singleton_holds_under_concurrency：
它锁死「不许退回 functools.lru_cache」—— lru_cache 在 miss 时允许多线程同时进入
被包装函数，对 2GB 模型就是偶发双份常驻内存。
"""
import logging
import pickle
import sys
import threading

import pytest
from langchain_core.documents import Document

from app.rag import bm25_index as bm25_mod
from app.rag import reranker as reranker_mod
from app.rag import vector_store as vs_mod


class _CountingEmbeddings:
    """替身：记录构造次数，不加载任何权重。"""

    constructions = []

    def __init__(self, model_name=None, model_kwargs=None, encode_kwargs=None):
        type(self).constructions.append(model_name)
        self.model_name = model_name

    def embed_query(self, text):
        return [0.0]


class _SlowCountingEmbeddings(_CountingEmbeddings):
    """构造时故意停顿，放大并发窗口，让「构造了多次」必然暴露出来。"""

    def __init__(self, *args, **kwargs):
        import time

        time.sleep(0.05)
        super().__init__(*args, **kwargs)


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """每个用例前把模块级单例清空，避免相互污染。"""
    monkeypatch.setattr(vs_mod, "_embeddings", None, raising=False)
    monkeypatch.setattr(vs_mod, "_vectorstore", None, raising=False)
    monkeypatch.setattr(bm25_mod, "_bm25", None, raising=False)
    monkeypatch.setattr(reranker_mod, "_reranker", None, raising=False)


def test_embeddings_constructed_once_and_reused(monkeypatch):
    _CountingEmbeddings.constructions = []
    monkeypatch.setattr(vs_mod, "HuggingFaceEmbeddings", _CountingEmbeddings)

    first = vs_mod.get_embeddings()
    second = vs_mod.get_embeddings()

    assert first is second
    assert len(_CountingEmbeddings.constructions) == 1


def test_embeddings_singleton_holds_under_concurrency(monkeypatch):
    """8 个线程同时抢 —— 构造必须只发生一次。"""
    _SlowCountingEmbeddings.constructions = []
    monkeypatch.setattr(vs_mod, "HuggingFaceEmbeddings", _SlowCountingEmbeddings)

    barrier = threading.Barrier(8)
    results = []
    errors = []

    def worker():
        try:
            barrier.wait()
            results.append(vs_mod.get_embeddings())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(_SlowCountingEmbeddings.constructions) == 1
    assert len({id(r) for r in results}) == 1


def test_bm25_index_is_shared(monkeypatch, tmp_path):
    monkeypatch.setattr(bm25_mod.settings, "data_dir", tmp_path, raising=False)

    assert bm25_mod.get_bm25_index() is bm25_mod.get_bm25_index()


def _tmp_index(monkeypatch, tmp_path) -> bm25_mod.BM25Index:
    """构造一个落盘目录指向 tmp_path 的索引。

    断言 _index_path 确实落在 tmp_path 里是道安全阀：真实 data/ 下躺着一份 2196 篇
    文档的索引，下面这些用例要往 index_path 里写垃圾字节，monkeypatch 一旦失效就会把它
    覆盖掉 —— 宁可用例失败，也不能毁掉真索引。
    """
    monkeypatch.setattr(bm25_mod.settings, "data_dir", tmp_path, raising=False)
    index = bm25_mod.BM25Index()
    assert index._index_path.parent == tmp_path, "拒绝在 tmp_path 之外写索引文件"
    return index


# 加载不了的几种真实形态：进程被 SIGKILL / OOM 打断留下的截断 pickle、磁盘上的垃圾
# 字节、刚 open() 就崩掉留下的空文件，以及结构变更后少键的 pickle。
_CORRUPT_PAYLOADS = {
    "truncated": pickle.dumps(
        {"documents": [Document(page_content="片段")], "index": None}
    )[:20],
    "garbage": b"\x80\x05not-a-pickle\xff\xff",
    "empty": b"",
    "missing_key": pickle.dumps({"documents": []}),
}


@pytest.mark.parametrize(
    "payload", list(_CORRUPT_PAYLOADS.values()), ids=list(_CORRUPT_PAYLOADS)
)
def test_bm25_load_returns_false_on_corrupt_file(
    monkeypatch, tmp_path, payload, caplog
):
    """索引文件损坏时 load() 返回 False，**不抛异常**。

    load() 跑在 lifespan 的预载里，抛出去就不是「某次检索失败」，而是整个 API 起不来。
    索引是可重建的，没什么值得为它崩掉。
    """
    index = _tmp_index(monkeypatch, tmp_path)
    index._index_path.write_bytes(payload)

    with caplog.at_level(logging.ERROR, logger="app.rag.bm25_index"):
        assert index.load() is False

    assert index.document_count == 0
    assert index.search("片段") == []
    assert caplog.records, "损坏文件必须留一条日志，不能静默吞掉"


class _VanishingGlobal:
    """空类。测试里先 pickle 它的实例，再把名字从本模块删掉，用来制造「类已不存在」。"""


def test_bm25_load_returns_false_when_a_pickled_class_is_gone(
    monkeypatch, tmp_path, caplog
):
    """pickle 里嵌的是**活的** BM25Okapi 实例，所以「加载不了」不止字节损坏一种。

    rank_bm25 升级、类改名或移动之后，旧 pickle 里的类引用就失效了，抛的是
    AttributeError（不是 UnpicklingError）—— 只捕字节级异常的窄元组接不住它，
    而 get_bm25_index() 的懒加载路径会照样抛出去。索引是可重建的缓存，
    任何「加载不了」都该降级成 False。
    """
    payload = pickle.dumps({"documents": [], "index": _VanishingGlobal()})
    # 现在把类名抹掉：反序列化时 pickle 在本模块里找不到它
    monkeypatch.delattr(sys.modules[__name__], "_VanishingGlobal")

    index = _tmp_index(monkeypatch, tmp_path)
    # 先放入一对已知状态：用来证明加载失败时旧的一对**原样保留**，而不是被赋成半成品。
    # 只断言「document_count == 0」区分不了「没动过」与「documents 赋了、index 没赋」。
    # 这里不断言 search() 的结果：单篇语料下查询词的 idf 为负，会整条被 score > 0 滤掉，
    # 那是 BM25 的数学（见下方 test_negative_top_k 的注释），与本用例要证明的事无关。
    index.build([Document(page_content="既有片段")])
    index._index_path.write_bytes(payload)

    with caplog.at_level(logging.ERROR, logger="app.rag.bm25_index"):
        assert index.load() is False

    assert index.document_count == 1
    assert index._pair[1] is not None
    assert caplog.records, "加载失败必须留一条日志，不能静默吞掉"


def test_bm25_add_refuses_to_overwrite_an_unreadable_index(monkeypatch, tmp_path):
    """数据丢失护栏：索引文件读不出来时，它是**唯一**副本。

    旧行为下 get_bm25_index() 会把空索引固化，之后任意一次上传都会走 add() → save()，
    用「只有新 chunk」的索引覆盖上去 —— 2196 篇会变成新上传的这几篇，而仓库里没有从
    Chroma 重建 BM25 的路径，不可恢复。宁可让这次上传响亮地失败。
    """
    monkeypatch.setattr(bm25_mod.settings, "data_dir", tmp_path, raising=False)
    index_path = tmp_path / "bm25_index.pkl"
    unreadable = b"\x80\x05not-a-pickle"
    index_path.write_bytes(unreadable)

    index = bm25_mod.BM25Index()
    assert index.load() is False
    assert index._load_failed is True

    with pytest.raises(RuntimeError, match="拒绝以空白基础覆盖落盘"):
        index.add([Document(page_content="新片段")])

    assert index_path.read_bytes() == unreadable, "磁盘上那份必须原封不动"


def test_bm25_save_does_not_refuse_when_there_is_no_index_yet(monkeypatch, tmp_path):
    """反向护栏：全新库（没有索引文件）不是「加载失败」，首次上传必须能落盘。"""
    monkeypatch.setattr(bm25_mod.settings, "data_dir", tmp_path, raising=False)
    index = bm25_mod.BM25Index()
    assert index.load() is False
    assert index._load_failed is False

    index.add([Document(page_content="劳动合同法第三十六条")])

    assert index.document_count == 1
    assert (tmp_path / "bm25_index.pkl").exists()


def test_get_bm25_index_retries_after_a_failed_load(monkeypatch, tmp_path):
    """一次瞬时故障不该让 BM25 整条路径在整个进程生命周期内都退化成空。"""
    monkeypatch.setattr(bm25_mod.settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(bm25_mod, "_bm25", None, raising=False)
    monkeypatch.setattr(bm25_mod, "_RETRY_INTERVAL_SECONDS", 0.0, raising=False)

    (tmp_path / "bm25_index.pkl").write_bytes(b"\x80\x05broken")

    first = bm25_mod.get_bm25_index()
    assert first._load_failed is True
    assert first.document_count == 0

    # 故障排除：写一份真实可读的索引
    healthy = bm25_mod.BM25Index()
    healthy.build([Document(page_content="劳动合同法第三十六条")])
    healthy.save()

    retried = bm25_mod.get_bm25_index()

    assert retried is first, "重试应作用于同一个单例，而不是另建一个"
    assert retried._load_failed is False
    assert retried.document_count == 1


def test_negative_top_k_returns_nothing(monkeypatch, tmp_path):
    """负数不能靠切片表达：sorted(...)[:-1] 会返回「除最后一个之外的全部」，
    既不空也不报错。三处路径必须一致地当成「不要结果」。

    语料刻意让查询词只出现在少数文档里：BM25Okapi 的 idf 是
    log((N-df+0.5)/(df+0.5))，只有 df < N/2 才为正；df == N 时（或 N=1 的单篇语料）
    idf 为负，会被 epsilon 地板压成非正，于是 search() 里 `score > 0` 的过滤把结果
    全丢掉 —— 那是 BM25 的数学，不是索引坏了，写小语料 fixture 时很容易撞上。
    """
    monkeypatch.setattr(bm25_mod.settings, "data_dir", tmp_path, raising=False)
    index = bm25_mod.BM25Index()
    index.build(
        [
            Document(page_content="劳动合同法第三十六条规定协商一致可以解除劳动合同"),
            Document(page_content="劳动者提前三十日书面通知用人单位可以解除劳动合同"),
            Document(page_content="Python是一种广泛使用的高级编程语言"),
            Document(page_content="FastAPI是现代Python Web框架支持异步处理"),
            Document(page_content="机器学习模型的评估指标包括准确率与召回率"),
        ]
    )

    assert index.search("劳动合同", top_k=5) != []
    assert index.search("劳动合同", top_k=0) == []
    assert index.search("劳动合同", top_k=-1) == []


def test_get_retriever_does_not_treat_zero_as_unset(monkeypatch):
    """`k or default` 会把 0 当成「没传」而回退成 hybrid_top_k —— 与 hybrid_search 里
    同一个 bug，只是隔了一个调用点。改用 is None 之后 0 原样传下去。

    （跳过向量这一路的正确入口是 hybrid_search(vector_top_k=0)，由 _vector_search
    短路；直接给 get_retriever 传 0 属误用，但也不该被静默改成 10。）
    """
    captured = {}

    class _FakeVectorStore:
        def as_retriever(self, search_kwargs):
            captured.update(search_kwargs)
            return object()

    monkeypatch.setattr(vs_mod, "get_vectorstore", lambda create=False: _FakeVectorStore())
    monkeypatch.setattr(vs_mod.settings, "hybrid_top_k", 10, raising=False)

    vs_mod.get_retriever(0)

    assert captured["k"] == 0


def test_bm25_singleton_survives_corrupt_file(monkeypatch, tmp_path):
    """lifespan 预载走的就是这条路径：损坏文件不该让 get_bm25_index() 抛出去。"""
    monkeypatch.setattr(bm25_mod.settings, "data_dir", tmp_path, raising=False)
    (tmp_path / "bm25_index.pkl").write_bytes(b"\x80\x05half a pickle")

    index = bm25_mod.get_bm25_index()  # 不抛
    assert index._index_path.parent == tmp_path, "拒绝在 tmp_path 之外写索引文件"

    assert index.document_count == 0
    assert index.search("片段") == []


def test_bm25_build_with_untokenizable_corpus_does_not_raise(monkeypatch, tmp_path):
    """空 PDF 页 / 纯空白 txt 会产出 `corpus = [[]]`，那是**真值**，但 BM25Okapi 内部
    要算 idf_sum / len(idf)，直接 ZeroDivisionError；upload 接口不做空内容过滤。

    守卫必须是 any(corpus)：一条文档都分不出词时，索引退化成 None，
    search() 返回 [] 而不是炸掉。
    """
    monkeypatch.setattr(bm25_mod.settings, "data_dir", tmp_path, raising=False)
    index = bm25_mod.BM25Index()

    index.build([Document(page_content="")])
    assert index.document_count == 1
    assert index._pair[1] is None
    assert index.search("片段") == []

    index.build([Document(page_content="----")])  # 正则分不出任何 token
    assert index.document_count == 1
    assert index._pair[1] is None
    assert index.search("----") == []

    index.build([])  # 空文档列表同样是退化路径
    assert index.document_count == 0
    assert index.search("片段") == []


def test_bm25_build_with_mixed_empty_and_real_pages(monkeypatch, tmp_path):
    """真实上传里空页与有内容的页混在一起，这时索引必须正常建起来。

    注意这里放了三篇而不是两篇：BM25Okapi 对「2 篇里出现 1 篇」的词算出的 idf 恰好是
    log(1.5)-log(1.5)=0，检索会返回空列表 —— 那是 BM25 的数学，不是索引坏了。
    """
    monkeypatch.setattr(bm25_mod.settings, "data_dir", tmp_path, raising=False)
    index = bm25_mod.BM25Index()

    index.build(
        [
            Document(page_content=""),
            Document(page_content="劳动合同法第三十六条规定协商一致可以解除劳动合同"),
            Document(page_content="Python是一种广泛使用的高级编程语言"),
        ]
    )

    assert index.document_count == 3
    assert index._pair[1] is not None
    results = index.search("劳动合同", top_k=5)
    assert [doc.page_content for doc, _ in results] == [
        "劳动合同法第三十六条规定协商一致可以解除劳动合同"
    ]


def test_bm25_search_is_safe_while_rebuilding(monkeypatch, tmp_path):
    """build() 把 (documents, index) 作为一对整体换掉，search() 只快照一次 —— 交错不抛异常。

    真实触发场景：同一个 ReAct 轮次里模型并发发起 parse_document 与 search_documents。
    若 build 分两步写（旧实现先写 _documents、分词建完索引再写 _index），search 就会读到
    长度不匹配的一对，`documents[i]` 直接 IndexError。

    重建刻意分成「先涨后缩」两段：**只有增长段测不出问题** —— 旧实现被抢占时留下的是
    「documents 长、index 短」的失配，而 search 只按下标取 documents，多出来的文档根本
    碰不到，所以不越界。必须让文档数缩回去（重解析后 chunk 变少就是这种情况），失配才
    翻转成「index 长、documents 短」，越界才暴露出来。
    对旧实现实测：只跑增长段 9 次运行 0 次触发；加上收缩段后 9/9 触发
    IndexError('list index out of range')（换短文本重复 10/10）。
    """
    monkeypatch.setattr(bm25_mod.settings, "data_dir", tmp_path, raising=False)

    index = bm25_mod.BM25Index()
    index.build([Document(page_content=f"旧片段{i}") for i in range(3)])

    stop = threading.Event()
    failures = []

    def searcher():
        while not stop.is_set():
            try:
                index.search("旧片段", top_k=5)
            except Exception as exc:  # noqa: BLE001
                failures.append(exc)
                return

    threads = [threading.Thread(target=searcher) for _ in range(4)]
    for t in threads:
        t.start()

    # 增长段：3 -> 100 个片段（新文档入库）
    for round_no in range(50):
        index.build([Document(page_content=f"新片段{round_no}-{i}") for i in range(100)])

    # 收缩段：100 -> 3 个片段（重解析后 chunk 变少）。这一段的失配方向才是致命的。
    for round_no in range(50):
        index.build([Document(page_content=f"小片段{round_no}-{i}") for i in range(3)])

    stop.set()
    for t in threads:
        t.join()

    assert failures == []
    assert index.document_count == 3


def test_reranker_is_singleton(monkeypatch):
    constructions = []

    class _FakeCrossEncoder:
        def __init__(self, model_name=None):
            constructions.append(model_name)

    monkeypatch.setattr(reranker_mod, "CrossEncoder", _FakeCrossEncoder)

    assert reranker_mod.get_reranker() is reranker_mod.get_reranker()
    assert len(constructions) == 1
