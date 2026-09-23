import pytest
from langchain_core.documents import Document

from app.rag import hybrid_search as hybrid_mod
from app.rag.hybrid_search import _rrf_fuse


def test_rrf_fusion_ranks_both_sources():
    docs = [
        Document(page_content="A"),
        Document(page_content="B"),
        Document(page_content="C"),
    ]
    vector = [(docs[0], 0.9), (docs[1], 0.5)]
    bm25 = [(docs[1], 0.8), (docs[2], 0.3)]

    result = _rrf_fuse(vector, bm25, top_k=3)
    ids = [doc.page_content for doc, _ in result]
    # B appears in both lists -> should rank highest
    assert ids[0] == "B"
    assert len(result) == 3


def test_rrf_empty_inputs():
    result = _rrf_fuse([], [], top_k=5)
    assert result == []


def test_rrf_handles_duplicate_content():
    doc = Document(page_content="same")
    result = _rrf_fuse([(doc, 1.0)], [(doc, 1.0)], top_k=5)
    assert len(result) == 1


class _EmptyRetriever:
    def invoke(self, query):
        return []


class _EmptyBM25:
    def search(self, query, top_k=10):
        return []


def test_vector_side_requests_hybrid_top_k(monkeypatch):
    """回归：向量侧过去固定只召回 settings.top_k=4 个候选。

    get_retriever() 把 k 硬编码成 settings.top_k，于是 hybrid_search 的
    vector_top_k 参数被无声忽略、settings.hybrid_top_k 成为死配置 ——
    RRF 实际融合的是 4 路向量结果 + 10 路 BM25 结果，而非 10+10。

    这里刻意把 hybrid_top_k 设成 7（**不是** app/config.py 里的默认值 10）：
    若断了回归写成 `vector_top_k or 10` 这类硬编码，断言就会失败。
    """
    requested = {}

    def _fake_get_retriever(k=None):
        requested["k"] = k
        return _EmptyRetriever()

    monkeypatch.setattr(hybrid_mod, "get_retriever", _fake_get_retriever)
    monkeypatch.setattr(hybrid_mod.settings, "hybrid_top_k", 7, raising=False)

    hybrid_mod.hybrid_search("劳动合同如何解除", _EmptyBM25())

    assert requested["k"] == 7


def test_explicit_vector_top_k_is_passed_through(monkeypatch):
    """显式传入的值必须原样到达 retriever，不能被配置值覆盖。"""
    requested = {}

    def _fake_get_retriever(k=None):
        requested["k"] = k
        return _EmptyRetriever()

    monkeypatch.setattr(hybrid_mod, "get_retriever", _fake_get_retriever)
    monkeypatch.setattr(hybrid_mod.settings, "hybrid_top_k", 10, raising=False)

    hybrid_mod.hybrid_search("劳动合同如何解除", _EmptyBM25(), vector_top_k=3)

    assert requested["k"] == 3


def test_zero_vector_top_k_skips_the_vector_side(monkeypatch):
    """0 是合法取值（「这一路不要候选」），不该被当成「没传」而回退成 hybrid_top_k。

    _vector_search 直接短路，连 retriever 都不构造 —— 也就不会把 k=0 交给 chroma
    的 n_results（那个行为未定义）。
    """
    called = []

    def _fake_get_retriever(k=None):
        called.append(k)
        return _EmptyRetriever()

    monkeypatch.setattr(hybrid_mod, "get_retriever", _fake_get_retriever)

    result = hybrid_mod.hybrid_search("劳动合同如何解除", _EmptyBM25(), vector_top_k=0)

    assert called == [], "vector_top_k=0 时不该构造 retriever"
    assert result == []
