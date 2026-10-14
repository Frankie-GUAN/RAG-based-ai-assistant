"""评测服务：走 Agent Core，全程异步。

原实现是同步的，却被 async 端点直接调用（含 time.sleep(3)），会把整个事件循环冻住。
"""
from __future__ import annotations

import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.agent.core import AgentDeps, AgentTask, run_agent_stream
from app.agent.llm import get_chat_model
from app.agent.protocol import CitationEvent, FinalEvent
from app.mcp.client import MCPToolClient
from app.services.chat_service import build_context_engine

# 查询之间的人工间隔，用于规避限流。
# 并发执行 + 信号量限流属于评测体系的后续优化，本轮保持串行。
_QUERY_INTERVAL_SECONDS = 3


async def _run_query(
    question: str, tools: MCPToolClient, max_iterations: int = 8
) -> tuple[str, str]:
    """跑一次问答，返回 (答案, 检索到的上下文)。

    **上下文口径：** 忠实度判官吃的是 `CitationEvent` 的片段，即真正作为工具结果
    进入模型上下文的那些段落。此前这里用 `ToolResultEvent.data` 近似，而那份数据
    是整个检索工具的原始返回；Core 加入引用事件后，口径已经可以对齐，忠实度分数
    的含义不再需要打折扣。
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
        if isinstance(event, CitationEvent) and event.chunk:
            retrieved.append(event.chunk)
        elif isinstance(event, FinalEvent):
            answer = event.answer

    return answer, "\n\n".join(retrieved)


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
