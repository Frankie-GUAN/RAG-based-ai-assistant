"""检索链路单例化测试。

最关键的一条是 test_embeddings_singleton_holds_under_concurrency：
它锁死「不许退回 functools.lru_cache」—— lru_cache 在 miss 时允许多线程同时进入
被包装函数，对 2GB 模型就是偶发双份常驻内存。
"""
import threading

import pytest

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


def test_bm25_search_is_safe_while_rebuilding():
    """build() 整体替换 (documents, index)，search() 快照一次 —— 二者交错不得抛异常。

    真实触发场景：同一个 ReAct 轮次里模型并发发起 parse_document 与 search_documents。
    若 build 分别替换两个属性，search 会读到长度不匹配的一对，直接 IndexError。
    """
    from langchain_core.documents import Document

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

    for round_no in range(50):
        index.build([Document(page_content=f"新片段{round_no}-{i}") for i in range(100)])

    stop.set()
    for t in threads:
        t.join()

    assert failures == []


def test_reranker_is_singleton(monkeypatch):
    constructions = []

    class _FakeCrossEncoder:
        def __init__(self, model_name=None):
            constructions.append(model_name)

    monkeypatch.setattr(reranker_mod, "CrossEncoder", _FakeCrossEncoder)

    assert reranker_mod.get_reranker() is reranker_mod.get_reranker()
    assert len(constructions) == 1
