"""引用溯源：Agent Core 从检索工具结果里提取片段并外发 CitationEvent。

走真实的 MCP 客户端/服务端链路（与 test_agent_core.py 同一套 stub server），
因此测的是「工具信封 → CitationEvent」这条真实路径，而不是替身之间的约定。
不调真实模型：ScriptedChatModel 按脚本产出 tool_call 与文本。
"""
import pytest
from mcp.server import MCPServer

from app.agent.core import AgentDeps, AgentTask, run_agent_stream
from app.agent.protocol import CitationEvent, FinalEvent
from app.context.engine import ContextBudget, ContextEngine
from app.context.tokens import HeuristicTokenCounter
from app.mcp.client import open_tool_client
from tests.fakes import ScriptedChatModel, text_turn, tool_turn


def stub_server(
    hits=None,
    ok: bool = True,
    error: str | None = None,
    extra: dict | None = None,
) -> MCPServer:
    """一个可控的 MCP 工具集：search_documents 返回指定载荷。"""
    server = MCPServer(name="citation-stub", version="0.1.0")

    @server.tool()
    async def search_documents(query: str, top_k: int = 4) -> dict:
        """检索私有文档库。"""
        return {"ok": ok, "data": hits, "error": error}

    tools_extra = extra or {}

    @server.tool()
    async def search_web(query: str, num: int = 5) -> dict:
        """联网搜索。"""
        return {"ok": True, "data": tools_extra.get("web"), "error": None}

    return server


def make_engine() -> ContextEngine:
    return ContextEngine(
        budget=ContextBudget(total_tokens=4_000),
        counter=HeuristicTokenCounter(),
    )


def _hits(*doc_ids: str, source: str = "labor_law.pdf", score: float = 0.032) -> list[dict]:
    return [
        {"doc_id": doc_id, "content": f"片段内容 {doc_id}", "source": source, "score": score}
        for doc_id in doc_ids
    ]


async def _run(model, server):
    async with open_tool_client(server) as tools:
        return [
            event async for event in run_agent_stream(
                AgentTask(question="劳动合同如何解除"),
                AgentDeps(model=model, tools=tools, context=make_engine(), max_iterations=4),
            )
        ]


def _citations(events) -> list[CitationEvent]:
    return [e for e in events if isinstance(e, CitationEvent)]


@pytest.mark.anyio
async def test_citations_precede_the_final_event():
    """引用必须先于 final 外发：前端拿到 final 时就该能渲染来源。"""
    model = ScriptedChatModel([
        tool_turn("search_documents", {"query": "劳动合同"}),
        text_turn("协商一致可以解除。"),
    ])

    events = await _run(model, stub_server(hits=_hits("a#p0#0", "b#p0#0")))

    types = [e.type for e in events]
    assert types[-1] == "final"
    assert "citation" in types
    assert types.index("citation") < types.index("final")


@pytest.mark.anyio
async def test_citation_payload_points_at_the_chunk():
    model = ScriptedChatModel([
        tool_turn("search_documents", {"query": "x"}),
        text_turn("答案"),
    ])

    events = await _run(model, stub_server(hits=_hits("law.pdf#p3#1", score=0.041)))

    citation = _citations(events)[0]
    assert citation.doc_id == "law.pdf#p3#1"
    assert citation.source == "labor_law.pdf"
    assert "片段内容" in citation.chunk
    assert citation.score == pytest.approx(0.041)


@pytest.mark.anyio
async def test_same_chunk_found_twice_is_emitted_once():
    """同一片段被多轮反复召回时，来源列表不该出现重复条目。"""
    model = ScriptedChatModel([
        tool_turn("search_documents", {"query": "第一轮"}, call_id="c1"),
        tool_turn("search_documents", {"query": "第二轮"}, call_id="c2"),
        text_turn("答案"),
    ])

    events = await _run(model, stub_server(hits=_hits("same#p0#0")))

    assert [c.doc_id for c in _citations(events)] == ["same#p0#0"]


@pytest.mark.anyio
async def test_no_search_means_no_citations():
    """纯推理的回答不该凭空长出来源 —— 那是在骗用户。"""
    model = ScriptedChatModel([text_turn("直接回答，不检索。")])

    events = await _run(model, stub_server(hits=_hits("x#p0#0")))

    assert _citations(events) == []


@pytest.mark.anyio
async def test_failed_search_produces_no_citations():
    model = ScriptedChatModel([
        tool_turn("search_documents", {"query": "x"}),
        text_turn("我不知道。"),
    ])

    events = await _run(model, stub_server(hits=None, ok=False, error="索引不可用"))

    assert _citations(events) == []


@pytest.mark.anyio
async def test_web_results_are_not_treated_as_document_citations():
    """联网搜索的结果不是文档片段，不该混进「引用来源」。"""
    model = ScriptedChatModel([
        tool_turn("search_web", {"query": "x"}),
        text_turn("答案"),
    ])
    server = stub_server(hits=_hits("x#p0#0"), extra={"web": [{"title": "网页", "link": "http://x"}]})

    events = await _run(model, server)

    assert _citations(events) == []


@pytest.mark.anyio
async def test_unexpected_payload_shape_does_not_break_the_answer():
    """载荷形状变了就当没有引用，而不是让整个工具结果解析失败。"""
    model = ScriptedChatModel([
        tool_turn("search_documents", {"query": "x"}),
        text_turn("答案"),
    ])

    events = await _run(model, stub_server(hits="unexpected string"))

    assert _citations(events) == []
    assert any(isinstance(e, FinalEvent) and e.answer == "答案" for e in events)


@pytest.mark.anyio
async def test_long_chunks_are_truncated_for_transport():
    """片段正文可能很长；截断后才不会把整篇文档塞进 SSE 帧。"""
    model = ScriptedChatModel([
        tool_turn("search_documents", {"query": "x"}),
        text_turn("答案"),
    ])
    long_content = "长" * 5000

    events = await _run(
        model,
        stub_server(hits=[{"doc_id": "d#p0#0", "content": long_content, "source": "s.pdf", "score": 0.1}]),
    )

    citation = _citations(events)[0]
    assert len(citation.chunk) <= 500