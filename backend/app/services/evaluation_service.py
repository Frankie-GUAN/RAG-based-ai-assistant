"""评测服务：走 Agent Core，全程异步。

原实现是同步的，却被 async 端点直接调用（含 time.sleep(3)），会把整个事件循环冻住。
"""
from __future__ import annotations

import asyncio
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.agent.core import AgentDeps, AgentTask, run_agent_stream
from app.agent.llm import get_chat_model
from app.agent.protocol import CitationEvent, FinalEvent
from app.mcp.client import MCPToolClient
from app.services.chat_service import build_context_engine

logger = logging.getLogger(__name__)

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


async def _score(chain, payload: dict) -> float | None:
    """打分。**解析失败返回 None，不返回 0。**

    0 分是一个有含义的结论 ——「回答不忠于上下文」。把「判官这次没能给出数字」
    也记成 0，等于伪造出一条质量问题：统计里混进了本该缺席的数据点，
    平均分被拉低，而真正需要修的东西（判官输出格式不稳定）反而被藏起来。
    """
    try:
        return float((await chain.ainvoke(payload)).strip())
    except (ValueError, TypeError):
        logger.warning("判官输出无法解析为分数，本次记为缺失", exc_info=True)
        return None


async def _evaluate_single(query: str, answer: str, context: str) -> dict:
    llm = get_chat_model()

    # 评分锚点：不给标尺，LLM 会把「基本相关」也打成 0.9，所有 query 的分数挤在
    # 0.85~1.0 之间，看不出差别 —— 这样的分数不能用来判断改动是好是坏。
    faith_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "你是严格的评估员，给「回答是否忠于给定上下文」打分，范围 0~1。\n"
             "评分标准：\n"
             "1.0 = 回答的每个事实都能在上下文里逐句找到依据\n"
             "0.5 = 主要论断有依据，但有细节来自上下文之外\n"
             "0.2 = 只有部分论断有依据\n"
             "0.0 = 回答与上下文无关，或明显包含上下文里没有的结论\n"
             "只输出一个数字，不要任何解释、标点或空格。"),
            ("human", "上下文: {context}\n\n回答: {answer}"),
        ])
        | llm
        | StrOutputParser()
    )
    relev_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "你是严格的评估员，给「回答与问题的相关程度」打分，范围 0~1。\n"
             "评分标准：\n"
             "1.0 = 直接回答了问题本身\n"
             "0.5 = 回答了问题的一部分，或答非所问但有交集\n"
             "0.2 = 只是提到相关话题，没有给出答案\n"
             "0.0 = 与问题完全无关\n"
             "只输出一个数字，不要任何解释、标点或空格。"),
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
