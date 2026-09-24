"""评测服务：走 Agent Core，全程异步。

原实现是同步的，却被 async 端点直接调用（含 time.sleep(3)），会把整个事件循环冻住。
"""
from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.agent.core import AgentDeps, AgentTask, run_agent_stream
from app.agent.llm import get_chat_model
from app.agent.protocol import FinalEvent, ToolResultEvent
from app.mcp.client import MCPToolClient
from app.services.chat_service import build_context_engine

# 查询之间的人工间隔，用于规避限流。
# 并发执行 + 信号量限流属于评测体系的后续优化，本轮保持串行。
_QUERY_INTERVAL_SECONDS = 3


async def _run_query(
    question: str, tools: MCPToolClient, max_iterations: int = 8
) -> tuple[str, str]:
    """跑一次问答，返回 (答案, 检索到的上下文)。

    **上下文口径（重要）：** 忠实度判官本应吃「真正进入 prompt 的检索片段」。
    P0 的 Agent Core 只发出 5 种事件，其中不含引用溯源，因此这里用工具返回的
    data 近似。将来给 Core 加上引用溯源时，必须同步复核此处的口径，
    否则 faithfulness 分数的含义会悄悄漂移。
    """
    answer = ""
    retrieved: list[str] = []

    async for event in run_agent_stream(
        AgentTask(question=question),
        AgentDeps(
            model=get_chat_model(),
            tools=tools,
            context=build_context_engine(),
            max_iterations=max_iterations,
        ),
    ):
        if isinstance(event, ToolResultEvent) and event.ok and event.data:
            retrieved.append(_flatten(event.data))
        elif isinstance(event, FinalEvent):
            answer = event.answer

    return answer, "\n\n".join(retrieved)


def _flatten(data: Any) -> str:
    """把工具返回的结构化片段压成判官可读的文本。

    **先拆信封。** `ToolResultEvent.data` 是完整信封 `{"ok", "data", "error"}`
    （见 `app/mcp/client.py` 的 `MCPToolClient.call`），工具真正的载荷在
    `payload["data"]` 里。照着 `str(data)` 压平的话，判官拿到的会是带 ok/error
    元数据的信封 repr，而不是检索片段 —— 分数照样出得来，但含义已经不是
    「回答是否基于上下文」了。

    下面这个判别式与客户端的**逐字相同**，这是刻意的（那份判别式是线上契约），
    但它现在存在于两处，任何一侧改动都可能让另一侧静默漂移：该判别式是
    **失败开放**的，多一个键就不再命中，信封会被当成普通 dict 压平。
    """
    if isinstance(data, dict) and set(data) == {"ok", "data", "error"}:
        data = data["data"]

    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or item.get("snippet") or item))
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    if isinstance(data, dict):
        return str(data.get("content") or data)
    return str(data)


async def _score(chain, payload: dict) -> float:
    try:
        return float((await chain.ainvoke(payload)).strip())
    except (ValueError, TypeError):
        return 0.0


async def _evaluate_single(query: str, answer: str, context: str) -> dict:
    llm = get_chat_model()

    faith_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "评分0-1：回答是否完全基于给定上下文？1=完全基于上下文。只输出数字。"),
            ("human", "上下文: {context}\n\n回答: {answer}"),
        ])
        | llm
        | StrOutputParser()
    )
    relev_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "评分0-1：回答与问题的相关程度。1=高度相关。只输出数字。"),
            ("human", "问题: {query}\n\n回答: {answer}"),
        ])
        | llm
        | StrOutputParser()
    )

    return {
        "faithfulness": await _score(faith_chain, {"context": context, "answer": answer}),
        "answer_relevancy": await _score(relev_chain, {"query": query, "answer": answer}),
    }


async def run_evaluation(queries: list[str], tools: MCPToolClient) -> list[dict]:
    results = []
    for i, query in enumerate(queries):
        try:
            answer, context = await _run_query(query, tools)
            if i < len(queries) - 1:
                await asyncio.sleep(_QUERY_INTERVAL_SECONDS)
            scores = await _evaluate_single(query, answer, context)
            results.append({
                "query": query, "answer": answer, "context": context,
                "error": None, **scores,
            })
        except Exception as e:  # noqa: BLE001 —— 单条失败不应中断整批评测
            results.append({
                "query": query, "answer": "", "context": "",
                "faithfulness": None, "answer_relevancy": None,
                "error": str(e),
            })
    return results
