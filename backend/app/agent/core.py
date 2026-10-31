"""Agent 入口：把「一次问答」跑成 StateGraph。

**这个文件在 2026-10-30 完成了一次迁移。** 迁移前它是 192 行的
`for` 循环，靠 `if accumulated.tool_calls:` 隐式表达流程；
`langgraph` 只在 `state.py` 里 import 了一个 `add_messages`，
`AgentState` 零引用——也就是说「基于 LangGraph 的 Agent」当时是假的。

现在：

· `run_agent_stream` = 编译好的 StateGraph（`app/agent/graph.py`）
· `run_agent_stream_legacy` = 旧循环，**保留一个版本周期**用于对照实验，
  之后删除。保留的理由不是「怕新实现不行」，而是这个项目的风格是
  「改动要有前后对照数据」——新旧两条链的事件序列与延迟可以并排比较。

对外接口（`AgentTask` / `AgentDeps` / `run_agent_stream` 的签名）
完全没变，`chat_service.py` 与前端都不需要改。

**事件形状也没变。** 节点里写出去的是 `event.to_dict()`，
入口这里再用 `event_from_dict` 还原成事件对象——所以上层拿到的
仍然是 `AgentEvent` 子类，`to_sse()` 照常工作。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import fields
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, ToolMessage

from app.agent.graph import build_graph
from app.agent.nodes import (
    MAX_CITATIONS,
    SYSTEM_PROMPT,
    AgentDeps,
    AgentNodes,
    AgentTask,
    _as_text,
    _citations_from,
)
from app.agent.protocol import (
    AgentEvent,
    CitationEvent,
    ErrorEvent,
    FinalEvent,
    PlanEvent,
    ReflectEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)

logger = logging.getLogger(__name__)

# 重新导出：保持 `from app.agent.core import AgentTask, AgentDeps` 的旧调用方式可用
__all__ = [
    "AgentDeps",
    "AgentTask",
    "event_from_dict",
    "run_agent_stream",
    "run_agent_stream_legacy",
]

_EVENT_CLASSES: dict[str, type[AgentEvent]] = {
    "token": TokenEvent,
    "tool_call": ToolCallEvent,
    "tool_result": ToolResultEvent,
    "citation": CitationEvent,
    "plan": PlanEvent,
    "reflect": ReflectEvent,
    "final": FinalEvent,
    "error": ErrorEvent,
}


def event_from_dict(payload: dict[str, Any]) -> AgentEvent:
    """把节点写出的 dict 还原成事件对象。

    **为什么要绕这一圈。** `get_stream_writer()` 只接受可序列化的值
    （它要支持跨进程序列化），所以节点里写的是 `to_dict()` 的结果。
    但上层（`to_sse`、会话落库、测试断言）都按事件对象工作——
    在这里做一次还原，对外契约就完全不变。

    字段用 `dataclasses.fields` 过滤：节点可能多写一个字段
    （比如排查用的临时信息），不该因此抛 `TypeError` 打断整条流。
    """
    event_type = str(payload.get("type") or "")
    cls = _EVENT_CLASSES.get(event_type)
    if cls is None:
        raise ValueError(f"未知事件类型：{event_type!r}")

    allowed = {f.name for f in fields(cls)} - {"type"}
    kwargs = {k: v for k, v in payload.items() if k in allowed}
    return cls(**kwargs)


async def run_agent_stream(task: AgentTask, deps: AgentDeps) -> AsyncIterator[AgentEvent]:
    """执行 Agent 图并产出事件流。

    保证：最后一个事件必定是 FinalEvent 或 ErrorEvent
    （`answer` 节点两条路径都覆盖了，图外再兜一层异常转换）。
    """
    context = deps.context.build(
        system=SYSTEM_PROMPT,
        question=task.question,
        summary=task.summary,
        long_term_memory=task.long_term_memory,
        history=task.history,
    )

    initial: dict[str, Any] = {
        "messages": list(context.messages),
        "question": task.question,
        "plan": "",
        "need_retrieval": False,
        "retrieval_query": task.question,
        "citations": [],
        "iteration": 0,
        "reflection_round": 0,
        "need_more_retrieval": False,
        "reflect_reason": "",
        "final_answer": "",
    }

    graph = build_graph(deps)

    try:
        async for payload in graph.astream(initial, stream_mode="custom"):
            if not isinstance(payload, dict):
                continue
            try:
                yield event_from_dict(payload)
            except ValueError:
                # 未知事件类型：记日志但不中断。协议扩展期可能出现
                # 「节点先写了新事件、上层映射表还没跟上」的短暂不一致。
                logger.warning("跳过未知事件：%r", payload.get("type"))
    except Exception as exc:  # noqa: BLE001 — 图执行失败必须转成事件，不能裸抛
        logger.exception("Agent 图执行失败")
        yield ErrorEvent(
            code="agent_failed",
            message=f"{type(exc).__name__}: {exc}",
            recoverable=False,
        )


async def run_agent_stream_legacy(
    task: AgentTask, deps: AgentDeps
) -> AsyncIterator[AgentEvent]:
    """迁移前的实现：手写 `for` 循环。

    **保留一个版本周期做对照，之后删除。** 它和 `run_agent_stream`
    产出的事件序列在同一批输入下应当一致（除了新增的 plan / reflect）,
    这正是验证「迁移没有改变行为」的依据。

    不要在新代码里调用它。
    """
    context = deps.context.build(
        system=SYSTEM_PROMPT,
        question=task.question,
        summary=task.summary,
        long_term_memory=task.long_term_memory,
        history=task.history,
    )
    messages = list(context.messages)

    schemas = [spec.to_openai_schema() for spec in deps.tools.specs]
    bound = deps.model.bind_tools(schemas) if schemas else deps.model

    citations: list[CitationEvent] = []
    seen_doc_ids: set[str] = set()

    for _ in range(deps.max_iterations):
        accumulated: AIMessage | None = None

        async for chunk in bound.astream(messages):
            accumulated = chunk if accumulated is None else accumulated + chunk
            text = _as_text(chunk.content)
            if text:
                yield TokenEvent(text=text)

        if accumulated is None:
            yield ErrorEvent(code="empty_response", message="模型未返回任何内容", recoverable=False)
            return

        messages.append(accumulated)

        if not accumulated.tool_calls:
            for citation in citations:
                yield citation
            yield FinalEvent(answer=_as_text(accumulated.content))
            return

        for call in accumulated.tool_calls:
            yield ToolCallEvent(id=call["id"], name=call["name"], args=call["args"])

        results = await asyncio.gather(
            *(deps.tools.call(call["name"], call["args"]) for call in accumulated.tool_calls)
        )

        for call, result in zip(accumulated.tool_calls, results):
            yield ToolResultEvent(
                id=call["id"], ok=result.ok, data=result.data, ms=result.ms, error=result.error
            )
            messages.append(ToolMessage(
                content=_dump({"ok": result.ok, "data": result.data, "error": result.error}),
                tool_call_id=call["id"],
            ))

            if result.ok and call["name"] == "search_documents":
                for item in _citations_from(result.data, seen_doc_ids):
                    citations.append(CitationEvent(
                        doc_id=item["doc_id"],
                        chunk=item["chunk"],
                        source=item["source"],
                        score=item["score"],
                        page_title=item["page_title"],
                        section_path=item["section_path"],
                    ))
                    if len(citations) >= MAX_CITATIONS:
                        break

    for citation in citations:
        yield citation
    yield ErrorEvent(
        code="max_iterations",
        message=f"达到迭代上限 {deps.max_iterations}，未能得出结论",
        recoverable=True,
    )


def _dump(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, default=str)
