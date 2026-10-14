"""Reranker 接入查询链路的测试。

全部用假精排器，**不加载 Cross-Encoder 的 2GB 模型**，因此不需要 requires_model 标记。
"""
import pytest
from langchain_core.documents import Document

from app.rag import hybrid_search as hybrid_mod


class _FakeReranker:
    """按预设顺序重排；可指定为「抛异常」以验证降级。"""

    def __init__(self, order=None, boom=False, empty=False):
        self._order = order          # list[str]，page_content 顺序
        self.boom = boom
        self.empty = empty
        self.calls = 0

    def rerank(self, query, candidates, top_k=4):
        self.calls += 1
        if self.boom:
            raise RuntimeError("模拟模型未下载")
        if self.empty:
            return []
        if self._order is None:
            return candidates[:top_k]
        by_content = {doc.page_content: (doc, score) for doc, score in candidates}
        picked = [by_content[c] for c in self._order if c in by_content]
        return picked[:top_k]


def _chunk(text, chunk_id):
    return Document(page_content=text, metadata={"chunk_id": chunk_id})


@pytest.fixture
def fused(monkeypatch):
    """固定融合结果：三篇候选，RRF 分递减。"""
    docs = [
        _chunk("劳动合同法第三十六条", "law#p0#0"),
        _chunk("FastAPI 支持异步", "py#p0#0"),
        _chunk("酿酒工艺流程", "wine#p0#0"),
    ]
    monkeypatch.setattr(
        hybrid_mod, "_vector_search", lambda q, top_k: [(docs[0], 0.9), (docs[1], 0.5)]
    )

    class _BM25:
        def search(self, query, top_k=10):
            return [(docs[2], 0.4), (docs[0], 0.8)]

    return docs, _BM25()


def _install(monkeypatch, reranker):
    import app.rag.reranker as reranker_mod

    monkeypatch.setattr(reranker_mod, "get_reranker", lambda: reranker)


def test_reranker_reorders_the_fused_candidates(monkeypatch, fused):
    docs, bm25 = fused
    fake = _FakeReranker(order=["FastAPI 支持异步", "劳动合同法第三十六条"])
    _install(monkeypatch, fake)
    monkeypatch.setattr(hybrid_mod.settings, "enable_reranker", True, raising=False)
    monkeypatch.setattr(hybrid_mod.settings, "rerank_candidates", 10, raising=False)
    monkeypatch.setattr(hybrid_mod.settings, "top_k", 2, raising=False)

    results = hybrid_mod.hybrid_search("q", bm25)

    assert fake.calls == 1
    assert [doc.page_content for doc, _ in results] == ["FastAPI 支持异步", "劳动合同法第三十六条"]


def test_scores_stay_fusion_scores_after_reranking(monkeypatch, fused):
    """精排只改顺序：CrossEncoder 的 logit 没有可解释的量纲，不能当成相关度展示。"""
    docs, bm25 = fused
    _install(monkeypatch, _FakeReranker(order=["FastAPI 支持异步"]))
    monkeypatch.setattr(hybrid_mod.settings, "enable_reranker", True, raising=False)
    monkeypatch.setattr(hybrid_mod.settings, "top_k", 1, raising=False)

    results = hybrid_mod.hybrid_search("q", bm25)

    assert len(results) == 1
    # 分数仍是 RRF 量级（1/(60+rank+1) 的量级），而不是 CE logit
    assert 0.0 < results[0][1] < 0.1


def test_failing_reranker_degrades_to_fusion_order(monkeypatch, fused):
    """精排挂了就退回 RRF 顺序，绝不能让整条检索路径失败。"""
    docs, bm25 = fused
    _install(monkeypatch, _FakeReranker(boom=True))
    monkeypatch.setattr(hybrid_mod.settings, "enable_reranker", True, raising=False)
    monkeypatch.setattr(hybrid_mod.settings, "top_k", 3, raising=False)

    results = hybrid_mod.hybrid_search("q", bm25)

    # RRF 融合顺序（k=60）：劳动合同法两路都出现（1/61 + 1/62）排第一；
    # 酿酒只在 BM25 里排第 0（1/61），FastAPI 只在向量里排第 1（1/62），
    # 两者只差一个名次的量 —— 精排失败不应改变这个顺序。
    assert [doc.page_content for doc, _ in results] == [
        "劳动合同法第三十六条",
        "酿酒工艺流程",
        "FastAPI 支持异步",
    ]


def test_empty_rerank_result_does_not_become_no_results(monkeypatch, fused):
    """精排返回空时不能静默变成「搜不出来」——那比不精排糟得多。"""
    docs, bm25 = fused
    _install(monkeypatch, _FakeReranker(empty=True))
    monkeypatch.setattr(hybrid_mod.settings, "enable_reranker", True, raising=False)
    monkeypatch.setattr(hybrid_mod.settings, "top_k", 2, raising=False)

    results = hybrid_mod.hybrid_search("q", bm25)

    assert len(results) == 2


def test_disabling_the_reranker_skips_it_entirely(monkeypatch, fused):
    docs, bm25 = fused
    fake = _FakeReranker()
    _install(monkeypatch, fake)
    monkeypatch.setattr(hybrid_mod.settings, "enable_reranker", False, raising=False)
    monkeypatch.setattr(hybrid_mod.settings, "top_k", 2, raising=False)

    results = hybrid_mod.hybrid_search("q", bm25)

    assert fake.calls == 0, "关掉精排时不该加载/调用 Cross-Encoder"
    assert len(results) == 2


def test_rerank_receives_the_configured_candidate_count(monkeypatch, fused):
    """融合阶段必须产出 rerank_candidates 个，而不是直接截到 top_k。

    否则「精排 10 个里的最优」实际只在 4 个里挑，精排形同虚设。
    """
    docs, bm25 = fused
    captured = {}
    real_fuse = hybrid_mod._rrf_fuse

    def _capturing_fuse(v, b, top_k):
        captured["fusion_top_k"] = top_k
        return real_fuse(v, b, top_k=top_k)

    monkeypatch.setattr(hybrid_mod, "_rrf_fuse", _capturing_fuse)
    fake = _FakeReranker()
    _install(monkeypatch, fake)
    monkeypatch.setattr(hybrid_mod.settings, "enable_reranker", True, raising=False)
    monkeypatch.setattr(hybrid_mod.settings, "rerank_candidates", 7, raising=False)
    monkeypatch.setattr(hybrid_mod.settings, "top_k", 3, raising=False)

    hybrid_mod.hybrid_search("q", bm25)

    assert captured["fusion_top_k"] == 7