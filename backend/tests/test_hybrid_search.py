import pytest
from langchain_core.documents import Document

from app.rag import hybrid_search as hybrid_mod
from app.rag.hybrid_search import _doc_key, _rrf_fuse


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


class _FakeVectorStore:
    """只实现 hybrid_search 用到的那一个方法。"""

    def __init__(self, results=None):
        self.results = results or []
        self.requested_k = None
        self.calls = 0

    def similarity_search_with_relevance_scores(self, query, k):
        self.calls += 1
        self.requested_k = k
        return self.results[:k]


class _EmptyBM25:
    def search(self, query, top_k=10):
        return []


def test_vector_side_requests_hybrid_top_k(monkeypatch):
    """回归：向量侧过去固定只召回 settings.top_k=4 个候选。

    彼时 get_retriever() 把 k 硬编码成 settings.top_k，于是 hybrid_search 的
    vector_top_k 参数被无声忽略、settings.hybrid_top_k 成为死配置 ——
    RRF 实际融合的是 4 路向量结果 + 10 路 BM25 结果，而非 10+10。

    这里刻意把 hybrid_top_k 设成 7（**不是** app/config.py 里的默认值 10）：
    若断了回归写成 `vector_top_k or 10` 这类硬编码，断言就会失败。
    """
    store = _FakeVectorStore()
    monkeypatch.setattr(hybrid_mod, "get_vectorstore", lambda create=False: store)
    monkeypatch.setattr(hybrid_mod.settings, "hybrid_top_k", 7, raising=False)

    hybrid_mod.hybrid_search("劳动合同如何解除", _EmptyBM25())

    assert store.requested_k == 7


def test_explicit_vector_top_k_is_passed_through(monkeypatch):
    """显式传入的值必须原样到达向量库，不能被配置值覆盖。"""
    store = _FakeVectorStore()
    monkeypatch.setattr(hybrid_mod, "get_vectorstore", lambda create=False: store)
    monkeypatch.setattr(hybrid_mod.settings, "hybrid_top_k", 10, raising=False)

    hybrid_mod.hybrid_search("劳动合同如何解除", _EmptyBM25(), vector_top_k=3)

    assert store.requested_k == 3


def test_zero_vector_top_k_skips_the_vector_side(monkeypatch):
    """0 是合法取值（「这一路不要候选」），不该被当成「没传」而回退成 hybrid_top_k。

    _vector_search 直接短路，连向量库句柄都不取 —— 也就不会把 k=0 交给 chroma，
    后者接受构造但会在查询时抛
    TypeError: Number of requested results 0, cannot be negative, or zero.
    """
    store = _FakeVectorStore()
    monkeypatch.setattr(hybrid_mod, "get_vectorstore", lambda create=False: store)

    result = hybrid_mod.hybrid_search("劳动合同如何解除", _EmptyBM25(), vector_top_k=0)

    assert store.calls == 0, "vector_top_k=0 时不该查询向量库"
    assert result == []


def test_vector_scores_are_real_similarity_not_rank_placeholders(monkeypatch):
    """回归：分数曾经是 `1.0 - i * 0.05` 编出来的名次占位值。

    它一路流到 search_documents 的输出里、被渲染成「相关度: 0.03」喂给 LLM，
    模型看到低分会以���资料不相关而拒答 —— 一个会主动降低回答质量的 bug。
    """
    docs = [
        Document(page_content="劳动合同法第三十六条", metadata={"chunk_id": "a#p0#0"}),
        Document(page_content="Python 是一门语言", metadata={"chunk_id": "b#p0#0"}),
    ]
    store = _FakeVectorStore([(docs[0], 0.87), (docs[1], 0.12)])
    monkeypatch.setattr(hybrid_mod, "get_vectorstore", lambda create=False: store)

    results = hybrid_mod._vector_search("劳动合同", top_k=2)

    assert [score for _, score in results] == [0.87, 0.12]


def test_vector_scores_are_clamped_into_the_unit_interval(monkeypatch):
    """余弦距离换算出的分数可能越界；夹到 [0,1] 只为可读，不影响排序。"""
    docs = [
        Document(page_content="A", metadata={"chunk_id": "a#p0#0"}),
        Document(page_content="B", metadata={"chunk_id": "b#p0#0"}),
    ]
    store = _FakeVectorStore([(docs[0], 1.4), (docs[1], -0.3)])
    monkeypatch.setattr(hybrid_mod, "get_vectorstore", lambda create=False: store)

    results = hybrid_mod._vector_search("x", top_k=2)

    assert [score for _, score in results] == [1.0, 0.0]


def test_missing_vector_store_is_reported(monkeypatch):
    """没有索引时必须明确报错，而不是静默返回空结果。"""
    monkeypatch.setattr(hybrid_mod, "get_vectorstore", lambda create=False: None)

    with pytest.raises(ValueError, match="No persisted vector store"):
        hybrid_mod._vector_search("x", top_k=4)


# --- 跨路去重键 ---

def test_neighbouring_chunks_are_not_merged_by_content_prefix():
    """回归：去重键曾经是 page_content[:200]。

    同一文档相邻分片开头高度相似（「第三章 劳动合同的订立……」），截断后被判成
    同一片段而互相覆盖 —— 于是 RRF 只剩一条，另一路的证据被 `doc_map[key] = doc`
    顶掉。
    """
    prefix = "第三章 劳动合同的订立" + "内容" * 40
    a = Document(page_content=prefix + "甲", metadata={"chunk_id": "law.pdf#p3#0"})
    b = Document(page_content=prefix + "乙", metadata={"chunk_id": "law.pdf#p3#1"})

    assert _doc_key(a) != _doc_key(b)

    result = _rrf_fuse([(a, 0.9)], [(b, 0.8)], top_k=5)
    assert len(result) == 2, "开头相似但 chunk_id 不同的两个片段必须都保留"


def test_same_chunk_from_both_sources_is_merged():
    chunk = Document(page_content="完全一样的内容", metadata={"chunk_id": "x.pdf#p0#0"})
    result = _rrf_fuse([(chunk, 0.9)], [(chunk, 0.8)], top_k=5)

    assert len(result) == 1


def test_doc_key_falls_back_when_metadata_is_absent():
    """没有 chunk_id 的旧数据也要有稳定的键，且内容相同的视为同一片段。"""
    a = Document(page_content="同样的内容", metadata={"source": "s.pdf", "page": 2})
    b = Document(page_content="同样的内容", metadata={"source": "s.pdf", "page": 2})
    c = Document(page_content="同样的内容", metadata={"source": "t.pdf", "page": 2})

    assert _doc_key(a) == _doc_key(b)
    assert _doc_key(a) != _doc_key(c), "不同来源的同内容片段不应被合并"


def test_non_positive_final_top_k_yields_nothing():
    """final_top_k 直接来自 LLM 的工具参数，所以非正数必须是「不要结果」而不是
    「除最后一个之外的全部」—— python 的 [:-1] 正是后者。"""
    docs = [Document(page_content=f"doc{i}") for i in range(5)]
    ranked = [(doc, 1.0) for doc in docs]

    assert hybrid_mod._rrf_fuse(ranked, [], top_k=5) != []
    assert hybrid_mod._rrf_fuse(ranked, [], top_k=0) == []
    assert hybrid_mod._rrf_fuse(ranked, [], top_k=-1) == []