"""Agent 核心循环：ReAct + 并行工具调用 + 迭代上限。

用 model.astream() 累积 AIMessageChunk，同时获得：
  · token 级真流式（文本增量立即外发）
  · tool calling（tool_call_chunks 在流结束后已合并为完整 tool_calls）

已知取舍：若模型先输出一段文字再发起工具调用，那段文字已经作为
token 事件外发了。通过 system prompt 约束「调用工具时不要输出解释文字」来规避。
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.agent.protocol import (
    AgentEvent,
    ErrorEvent,
    FinalEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from app.context.engine import ContextEngine
from app.mcp.client import MCPToolClient

SYSTEM_PROMPT = (
    "你是智能问答助手，基于私有文档知识库与联网搜索回答用户问题。"
    "你可以调用工具检索私有文档、联网搜索、解析上传的文档。\n"
    "规则：\n"
    "1. 需要事实依据时先调用工具，不要凭记忆回答文档中的内容或数据。\n"
    "2. 调用工具时不要输出解释性文字。\n"
    "3. 资料不足时明确说明不确定，不要编造。\n"
    "4. 引用检索结果时说明来源。"
)


@dataclass
class AgentTask:
    question: str
    history: list[BaseMessage] = field(default_factory=list)
    summary: str = ""


@dataclass
class AgentDeps:
    model: BaseChatModel
    tools: MCPToolClient
    context: ContextEngine
    max_iterations: int = 8


async def run_agent_stream(task: AgentTask, deps: AgentDeps) -> AsyncIterator[AgentEvent]:
    """执行 Agent 循环并产出事件流。

    保证：最后一个事件必定是 FinalEvent 或 ErrorEvent。
    """
    context = deps.context.build(
        system=SYSTEM_PROMPT,
        question=task.question,
        summary=task.summary,
        history=task.history,
    )
    messages: list[BaseMessage] = list(context.messages)

    schemas = [spec.to_openai_schema() for spec in deps.tools.specs]
    bound = deps.model.bind_tools(schemas) if schemas else deps.model

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
            yield FinalEvent(answer=_as_text(accumulated.content))
            return

        # 先外发全部 tool_call，让前端立刻看到并发意图，再并行执行
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
                content=json.dumps(
                    {"ok": result.ok, "data": result.data, "error": result.error},
                    ensure_ascii=False,
                    default=str,
                ),
                tool_call_id=call["id"],
            ))

    yield ErrorEvent(
        code="max_iterations",
        message=f"达到迭代上限 {deps.max_iterations}，未能得出结论",
        recoverable=True,
    )


def _as_text(content: Any) -> str:
    """兼容 str 与 LangChain 的多模态 content block 列表。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)
