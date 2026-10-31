"""检索工具这一层的接线测试。

**为什么单独测这一层。** 删除的「立刻不再命中」语义由两个模块共同完成：
`knowledge_service` 维护软删除名单，`tools/document_search` 负责过滤。之前的测试
只测了前者——过滤器那一段补丁因为字符串不匹配**静默失败**，151 个测试全绿，
而线上「删掉的文档仍然被引用」。所以这里必须走真实的 search_documents。
"""
import pytest
from langchain_core.documents import Document

from app.services import knowledge_service as svc
from app.tools import document_search as tool_mod


class _FakeIndex:
    def __init__(self):
        self.chunks = []

    def search(self, query, top_k=10):
        return [(c, 0.1) for c in self.chunks[:top_k]]


@pytest.fixture
def wired(monkeypatch):
    """把检索链路的三段都替换成内存实现：BM25、hybrid_search、软删除名单。"""
    index = _FakeIndex()

    def _chunk(source, text):
        return Document(page_content=text, metadata={"source": source, "chunk_id": f"{source}#p0#0"})

    index.chunks = [
        _chunk("keep.pdf", "保留文档的内容"),
        _chunk("gone.pdf", "已删除文档的内容"),
        _chunk("keep.pdf", "保留文档的另一段"),
    ]

    monkeypatch.setattr(tool_mod, "get_bm25_index", lambda: index)
    monkeypatch.setattr(tool_mod, "hybrid_search", lambda q, idx, final_top_k=None: idx.search(q, top_k=final_top_k or 4))
    # 名单由这个开关控制
    hidden = {"gone.pdf"}
    monkeypatch.setattr("app.services.knowledge_service.deleted_sources", lambda *a, **k: set(hidden))
    return index, hidden


def test_deleted_document_is_not_returned(wired):
    """核心断言：软删除的来源不得出现在检索结果里。"""
    _index, hidden = wired

    results = tool_mod.search_documents("查询", top_k=4)

    sources = {r["source"] for r in results}
    assert "gone.pdf" not in sources, f"已删除文档仍在结果里：{sources}"
    assert "keep.pdf" in sources


def test_no_hidden_documents_means_no_extra_filtering(wired):
    """名单为空时走原路径，不额外放宽候选数（否则每次检索都白取三倍）。"""
    _index, hidden = wired
    hidden.clear()

    results = tool_mod.search_documents("查询", top_k=2)

    assert len(results) == 2, "top_k 应被如实遵守"


def test_filter_still_fills_up_to_top_k(wired):
    """过滤之后要还能凑满 top_k —— 否则删掉几篇文档就只剩两条结果了。

    候选池宽度是 max(top_k*3, top_k+隐藏数) = 9。构造 6 条已删除 + 7 条保留：
    池子里 6 条被滤掉，仍应取满 3 条保留的 —— 这正是放宽候选宽度的意义。
    """
    index, _hidden = wired
    for i in range(5):
        index.chunks.append(Document(page_content=f"保留-{i}", metadata={"source": "keep.pdf", "chunk_id": f"k{i}"}))
    for i in range(6):
        index.chunks.insert(0, Document(page_content=f"已删-{i}", metadata={"source": "gone.pdf", "chunk_id": f"g{i}"}))

    results = tool_mod.search_documents("查询", top_k=3)

    assert len(results) == 3
    assert all(r["source"] == "keep.pdf" for r in results)


def test_filter_expands_pool_when_first_attempt_is_short(wired):
    """候选池要**渐进放宽**：一次定死会在「删了很多文档」时凑不满。

    上一条测试覆盖的是「初值刚好够」的情形。这一条构造初值不够的场景：
    top_k=3、隐藏 1 篇 → 初值池宽 9；而前 9 条候选里 8 条都是已删文档，
    过滤后只剩 1 条，凑不满 3。此时必须翻倍重试（池宽 18）才能取满。

    修这个 bug 的原因：旧实现池宽一次定死，表现为
    「明明有相关文档却检索不到」，且删除的文档越多越明显。
    """
    index, _hidden = wired
    # 8 条已删文档排在最前，后面才是保留文档
    for i in range(8):
        index.chunks.insert(0, Document(page_content=f"已删-{i}", metadata={"source": "gone.pdf", "chunk_id": f"g{i}"}))
    for i in range(6):
        index.chunks.append(Document(page_content=f"保留-{i}", metadata={"source": "keep.pdf", "chunk_id": f"k{i}"}))

    results = tool_mod.search_documents("查询", top_k=3)

    assert len(results) == 3, f"候选池未放宽，只取到 {len(results)} 条"
    assert all(r["source"] == "keep.pdf" for r in results)


def test_filter_gives_up_when_nothing_survives(wired):
    """全部候选都被删时，如实返回空，不硬凑、不无限取候选。"""
    index, hidden = wired
    hidden.clear()
    hidden.add("keep.pdf")          # 保留的那份也标记为已删
    hidden.add("gone.pdf")

    results = tool_mod.search_documents("查询", top_k=3)

    assert results == []


def test_wiring_is_really_there(monkeypatch):
    """防回归：过滤器那一段曾因补丁字符串不匹配而静默失效，测试却全绿。

    这条断言直接读源码——它不需要数据库、不需要索引，也就不会被任何 monkeypatch
    绕过。
    """
    import inspect

    source = inspect.getsource(tool_mod.search_documents)
    assert "deleted_sources" in source, "search_documents 没有读取软删除名单"
    assert "hidden" in source, "search_documents 没有按名单过滤"
    # 名单为空时的分支也要在：否则每次检索都会放宽三倍候选
    assert "if not hidden" in source
