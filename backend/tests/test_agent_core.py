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


# 迁移到 StateGraph 后新增的两类「元事件」：它们描述系统正在做什么
# （规划、自检），不属于回答内容。内容流断言在过滤后的序列上验证，
# 元事件的存在与内容由 test_plan_and_reflect_events_are_emitted 单独覆盖。
_META_EVENT_TYPES = {"plan", "reflect"}


def core_only(events):
    """只看内容流：token / 工具调用 / 工具结果 / 引用 / 最终答案。"""
    return [e for e in events if e.type not in _META_EVENT_TYPES]


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

    assert events_of(core_only(events)) == ["token", "token", "final"]
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

    core = core_only(events)
    assert events_of(core) == [
        # 检索工具返回过片段，因此 final 之前会先外发引用（溯源）事件
        "tool_call", "tool_result", "token", "token", "token", "citation", "final",
    ]
    call = core[0]
    assert call.name == "search_documents"
    assert call.args == {"query": "合同解除"}
    assert core[1].ok is True


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

    core = core_only(events)
    assert events_of(core)[-1] == "final"
    assert core[1].ok is False
    assert "服务不可用" in core[1].error

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

    core = core_only(events)
    types = events_of(core)
    assert types[:2] == ["tool_call", "tool_call"]
    assert types[2:4] == ["tool_result", "tool_result"]
    assert {core[0].id, core[1].id} == {"call_a", "call_b"}


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


# ---------------------------------------------------------------------------
# 迁移到 StateGraph 后新增的能力测试
# ---------------------------------------------------------------------------


class FakePlanner:
    """规划模型替身：固定返回一段 JSON。

    ScriptedChatModel 只有 astream，没有 ainvoke——规划节点走的是
    ainvoke（一次性拿完整 JSON，不需要流式）。所以单独做一个替身。
    """

    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    async def ainvoke(self, prompt):
        from langchain_core.messages import AIMessage
        import json as _json

        self.prompts.append(prompt)
        return AIMessage(content=_json.dumps(self.payload, ensure_ascii=False))


def make_empty_server():
    """检索永远返回空结果的 stub —— 用于验证反思补检索。"""
    server = MCPServer(name="empty", version="0.1.0")

    @server.tool()
    async def search_documents(query: str, top_k: int = 4) -> dict:
        """检索私有文档库（永远返回空）。"""
        return {"ok": True, "data": [], "error": None}

    return server


@pytest.mark.anyio
async def test_graph_exposes_six_nodes():
    """图里必须有六个节点。

    **这条用例的价值是防止「宣称用了 LangGraph 但其实是手写循环」再次发生。**
    迁移前的代码里，`AgentState` 零引用、`langgraph` 只 import 了一个
    `add_messages`，但文档写着「LangGraph Agent」。这条断言把
    「图里真有这些节点」变成可自动验证的事实。
    """
    from app.agent.graph import build_graph

    model = ScriptedChatModel([])
    async with open_tool_client(make_stub_server()) as tools:
        deps = AgentDeps(model=model, tools=tools, context=make_engine())
        mermaid = build_graph(deps).get_graph().draw_mermaid()

    for node in ["plan", "retrieve", "agent", "tools", "reflect", "answer"]:
        assert node in mermaid, f"图里缺少节点：{node}"


@pytest.mark.anyio
async def test_plan_and_reflect_events_are_emitted():
    """规划与自检事件必须真的被发出，且内容自洽。"""
    model = ScriptedChatModel([text_turn("答案")])

    async with open_tool_client(make_stub_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="你好"),
                AgentDeps(model=model, tools=tools, context=make_engine()),
            )
        ]

    types = events_of(events)
    assert "plan" in types, "缺少规划事件"
    assert "reflect" in types, "缺少反思事件"

    plan = next(e for e in events if e.type == "plan")
    assert plan.need_retrieval is False, "未启用 planner 时不应预检索"
    assert plan.reason, "规划应给出理由，便于排查"

    reflect = next(e for e in events if e.type == "reflect")
    assert reflect.need_more is False
    assert reflect.reason

    # 顺序契约：规划在最前，反思在 final 之前
    assert types[0] == "plan"
    assert types.index("reflect") < types.index("final")


@pytest.mark.anyio
async def test_planner_enables_pre_retrieval_before_reasoning():
    """启用 planner 后，先检索再推理。**初次检索用的是用户原话，不是改写 query。**

    这一条是实测得出的：规划改写会丢术语、引入语义偏移，
    实测「常量传播分析怎么做」改写成「常量传播分析 实现方法」后，
    召回从「命中常量传播章节」变成「召回指针分析的实现章节」。
    改写 query 只留给反思补检索用。
    """
    planner = FakePlanner({
        "need_retrieval": True,
        "query": "劳动合同 解除 条件",
        "reason": "需要查文档",
    })
    model = ScriptedChatModel([text_turn("根据资料，答案是……")])

    async with open_tool_client(make_stub_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="劳动合同解除要什么条件"),
                AgentDeps(
                    model=model, tools=tools, context=make_engine(), planner=planner
                ),
            )
        ]

    core = core_only(events)
    types = events_of(core)

    # 检索发生在推理之前：tool_call 先于第一个 token
    assert types[0] == "tool_call"
    assert types[0:2] == ["tool_call", "tool_result"]
    assert "token" in types
    assert types.index("tool_call") < types.index("token")

    # 初次检索用用户原话（保留术语与措辞），不是规划改写值
    assert core[0].args == {"query": "劳动合同解除要什么条件"}

    # 检索到了引用，随答案一起外发
    assert "citation" in types


@pytest.mark.anyio
async def test_reflect_triggers_one_supplementary_retrieval():
    """规划说需要检索但一条引用都没拿到时，反思应补检索一次——且只补一次。"""
    planner = FakePlanner({
        "need_retrieval": True,
        "query": "不存在的主题",
        "reason": "需要查文档",
    })
    model = ScriptedChatModel([text_turn("资料不足"), text_turn("确实没有")])

    async with open_tool_client(make_empty_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="q"),
                AgentDeps(
                    model=model, tools=tools, context=make_engine(), planner=planner
                ),
            )
        ]

    reflects = [e for e in events if e.type == "reflect"]
    retrievals = [e for e in events if e.type == "tool_call"]

    assert len(reflects) >= 2, "应至少反思两次：一次判需补检索，一次判最终完成"
    assert reflects[0].need_more is True
    assert reflects[-1].need_more is False, "补检索后必须收敛，不能无限重试"

    # 初次检索 + 反思补检索 = 2 次，不多不少
    assert len(retrievals) == 2, f"检索次数应为 2，实际 {len(retrievals)}"

    # 最终仍然给出答案（资料不足也要如实作答，不是报错）
    assert events[-1].type == "final"


@pytest.mark.anyio
async def test_planner_failure_degrades_to_direct_retrieval():
    """规划模型抛异常时必须降级，不能中断问答。"""

    class BrokenPlanner:
        async def ainvoke(self, prompt):
            raise RuntimeError("规划服务不可用")

    model = ScriptedChatModel([text_turn("答案")])

    async with open_tool_client(make_stub_server()) as tools:
        events = [
            e async for e in run_agent_stream(
                AgentTask(question="q"),
                AgentDeps(
                    model=model, tools=tools, context=make_engine(),
                    planner=BrokenPlanner(),
                ),
            )
        ]

    assert events[-1].type == "final", "规划失败也要给出答案"
    plan = next(e for e in events if e.type == "plan")
    assert plan.need_retrieval is True, "降级策略是「按需要检索」更安全"
    assert "失败" in plan.reason


def test_state_fields_are_all_used():
    """状态字段必须在节点里真的有读写点。

    **防的是「定义了但零引用」再次发生。** 迁移前 `AgentState` 有 8 个字段，
    其中 `has_docs` / `route` / `context` / `summary` 四个从没被任何代码碰过——
    它们只是让这个类「看起来像个 LangGraph 状态」。

    这条用例读 nodes.py / graph.py 的源码，确认每个字段名都出现过。
    它不能证明「语义正确」，但能抓住「加了字段却忘了用」这类退化。
    """
    import inspect

    from app.agent import graph as graph_mod
    from app.agent import nodes as nodes_mod
    from app.agent.state import AgentState

    source = inspect.getsource(nodes_mod) + inspect.getsource(graph_mod)

    annotations = AgentState.__annotations__
    unused = [name for name in annotations if name not in source]

    assert not unused, f"状态字段没有生产者：{unused}"
