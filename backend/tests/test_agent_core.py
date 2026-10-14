import pytest
from mcp.server import MCPServer

from app.agent.core import AgentDeps, AgentTask, run_agent_stream
from app.context.engine import ContextBudget, ContextEngine
from app.context.tokens import HeuristicTokenCounter
from app.mcp.client import open_tool_client
from tests.fakes import ScriptedChatModel, text_turn, tool_turn, two_tool_turn

SLOW_TOOL_MARKER = "slow"


def make_stub_server() -> MCPServer:
    server = MCPServer(name="stub", version="0.1.0")

    @server.tool()
    async def search_documents(query: str, top_k: int = 4) -> dict:
        """检索私有文档库。"""
        return {"ok": True, "data": [{"doc_id": "d1", "content": f"{query} 的结果"}], "error": None}

    @server.tool()
    async def broken_tool(query: str = "") -> dict:
        """总是失败，用于验证工具失败不中断循环。"""
        return {"ok": False, "data": None, "error": "RuntimeError: 服务不可用"}

    return server


def make_engine() -> ContextEngine:
    return ContextEngine(
        budget=ContextBudget(total_tokens=4_000),
        counter=HeuristicTokenCounter(),
    )


def events_of(events):
    return [e.type for e in events]


@pytest.mark.anyio
async def test_agent_answers_directly_without_tools():
    model = ScriptedChatModel([text_turn("你好", "！")])

    async with open_tool_client(make_stub_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="你好"),
                AgentDeps(model=model, tools=tools, context=make_engine()),
            )
        ]

    assert events_of(events) == ["token", "token", "final"]
    assert events[-1].answer == "你好！"


@pytest.mark.anyio
async def test_agent_executes_tool_then_streams_the_answer():
    model = ScriptedChatModel([
        tool_turn("search_documents", {"query": "合同解除"}),
        text_turn("根据", "资料", "答案是……"),
    ])

    async with open_tool_client(make_stub_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="劳动合同解除需要什么条件？"),
                AgentDeps(model=model, tools=tools, context=make_engine()),
            )
        ]

    assert events_of(events) == [
        # 检索工具返回过片段，因此 final 之前会先外发引用（溯源）事件
        "tool_call", "tool_result", "token", "token", "token", "citation", "final",
    ]
    call = events[0]
    assert call.name == "search_documents"
    assert call.args == {"query": "合同解除"}
    assert events[1].ok is True


@pytest.mark.anyio
async def test_tool_result_is_fed_back_to_the_model():
    model = ScriptedChatModel([
        tool_turn("search_documents", {"query": "合同解除"}),
        text_turn("完成"),
    ])

    async with open_tool_client(make_stub_server()) as tools:
        _ = [
            e async for e in run_agent_stream(
                AgentTask(question="q"),
                AgentDeps(model=model, tools=tools, context=make_engine()),
            )
        ]

    second_call_messages = model.calls[1]
    tool_messages = [m for m in second_call_messages if type(m).__name__ == "ToolMessage"]

    assert len(tool_messages) == 1
    assert "合同解除 的结果" in tool_messages[0].content
    assert tool_messages[0].tool_call_id == "call_1"


@pytest.mark.anyio
async def test_failing_tool_does_not_abort_the_loop():
    """工具失败要以 ok=false 回灌给模型，让模型自行决定重试或降级。"""
    model = ScriptedChatModel([
        tool_turn("broken_tool", {"query": "x"}),
        text_turn("工具不可用，我根据已有知识回答"),
    ])

    async with open_tool_client(make_stub_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="q"),
                AgentDeps(model=model, tools=tools, context=make_engine()),
            )
        ]

    assert events_of(events)[-1] == "final"
    assert events[1].ok is False
    assert "服务不可用" in events[1].error

    tool_messages = [m for m in model.calls[1] if type(m).__name__ == "ToolMessage"]
    assert '"ok": false' in tool_messages[0].content


@pytest.mark.anyio
async def test_independent_tool_calls_run_in_parallel():
    """同一轮内的多个工具调用必须并发，且全部 tool_call 先于任何 tool_result。"""
    model = ScriptedChatModel([
        two_tool_turn(
            ("search_documents", {"query": "a"}, "call_a"),
            ("search_documents", {"query": "b"}, "call_b"),
        ),
        text_turn("好了"),
    ])

    async with open_tool_client(make_stub_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="q"),
                AgentDeps(model=model, tools=tools, context=make_engine()),
            )
        ]

    types = events_of(events)
    assert types[:2] == ["tool_call", "tool_call"]
    assert types[2:4] == ["tool_result", "tool_result"]
    assert {events[0].id, events[1].id} == {"call_a", "call_b"}


@pytest.mark.anyio
async def test_loop_stops_at_max_iterations():
    """模型一直调工具不收敛时，必须止步而不是无限循环。"""
    model = ScriptedChatModel([
        tool_turn("search_documents", {"query": f"q{i}"}, f"call_{i}")
        for i in range(10)
    ])

    async with open_tool_client(make_stub_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="q"),
                AgentDeps(model=model, tools=tools, context=make_engine(), max_iterations=3),
            )
        ]

    assert events[-1].type == "error"
    assert events[-1].code == "max_iterations"
    assert len(model.calls) == 3


@pytest.mark.anyio
async def test_tool_schemas_are_bound_to_the_model():
    model = ScriptedChatModel([text_turn("hi")])

    async with open_tool_client(make_stub_server()) as tools:
        _ = [
            e async for e in run_agent_stream(
                AgentTask(question="q"),
                AgentDeps(model=model, tools=tools, context=make_engine()),
            )
        ]

    names = {t["function"]["name"] for t in model.bound_tools}
    assert names == {"search_documents", "broken_tool"}
