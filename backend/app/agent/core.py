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
    CitationEvent,
    ErrorEvent,
    FinalEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from app.context.engine import ContextEngine
from app.mcp.client import MCPToolClient

# 一次回答最多外发多少条引用。检索默认取 top_k=4 条，一般不会触顶；触顶时
# 保留最靠前的那些 —— 它们是精排/融合后的前排。
_MAX_CITATIONS = 8

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

            # 引用溯源：从检索工具的结果里提取引用片段。多轮里同一片段被反复召回
            # 只算一次，否则来源列表会出现重复条目。
            if result.ok and call["name"] == "search_documents":
                for citation in _citations_from(result.data, seen_doc_ids):
                    citations.append(citation)
                    if len(citations) >= _MAX_CITATIONS:
                        break

    for citation in citations:
        yield citation
    yield ErrorEvent(
        code="max_iterations",
        message=f"达到迭代上限 {deps.max_iterations}，未能得出结论",
        recoverable=True,
    )


def _citations_from(data: Any, seen: set[str]) -> list[CitationEvent]:
    """把 search_documents 的返回载荷转成引用事件。

    载荷形状是 MCP 信封 `{"ok", "data", "error"}`（见 app/mcp/client.py），
    真正的检索结果在 `data` 里，是一个 list[dict]，每项含
    doc_id / content / source / score。

    **不做严格校验。** 载荷形状变了就返回空列表（本次回答就没有来源），
    而不是让整个工具结果解析失败 —— 溯源是增强，不该拖垮问答本身。
    """
    if isinstance(data, dict) and set(data) == {"ok", "data", "error"}:
        data = data["data"]
    if not isinstance(data, list):
        return []

    citations: list[CitationEvent] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "")
        content = str(item.get("content") or "")
        if not doc_id or not content or doc_id in seen:
            continue
        seen.add(doc_id)
        citations.append(CitationEvent(
            doc_id=doc_id,
            chunk=content[:500],
            source=str(item.get("source") or ""),
            score=float(item.get("score") or 0.0),
        ))
    return citations


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
