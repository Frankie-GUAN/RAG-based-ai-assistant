"""对话服务：三层记忆 + Agent 图，产出事件流。

## 三层记忆在这里汇合

```
用户提问
  │
  ├─① 语义缓存查询（相似度 ≥ 0.90 直接返回，跳过检索与生成）
  │     命中 → cache_hit + citation + final，流程结束
  │
  ├─② 长期记忆召回（用户偏好 / 已确认事实，注入 system）
  │     与 ③ 一起进 ContextEngine 的 memory 层
  │
  ├─③ 短期记忆（本会话最近 N 轮 + 滚动摘要，调用方已备好）
  │
  → Agent 图（plan → retrieve → agent ⇄ tools → reflect → answer）
  │
  └─④ 回答完成后：写语义缓存 + 后台抽取长期记忆
```

**顺序不是随意的**：

- 缓存必须在最前——它的全部价值就是跳过后面所有步骤
- 记忆召回必须在构图前——它要进 system prompt
- 缓存写入必须在答案产出后——没有答案就没什么可缓存
- 记忆抽取必须异步——它要调 LLM，不能拖慢已经完成的回答
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agent.core import AgentDeps, AgentTask, run_agent_stream
from app.agent.llm import get_chat_model
from app.agent.protocol import (
    AgentEvent,
    CacheHitEvent,
    CitationEvent,
    FinalEvent,
)
from app.cache.semantic import CachedAnswer, get_semantic_cache
from app.config import settings
from app.context.engine import ContextBudget, ContextEngine
from app.context.tokens import HeuristicTokenCounter
from app.mcp.client import MCPToolClient
from app.memory.extractor import schedule_extraction
from app.memory.store import DEFAULT_USER_ID, MemoryItem, get_memory_store

logger = logging.getLogger(__name__)


def build_context_engine() -> ContextEngine:
    return ContextEngine(
        budget=ContextBudget.from_settings(settings),
        counter=HeuristicTokenCounter(),
    )


def to_langchain_messages(history: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    return messages


def format_memories(memories: list[MemoryItem]) -> str:
    """把召回的记忆渲染成给模型看的文本。

    **带 kind 标签是有意的**：模型看到「[偏好]」会知道这是个应当长期遵守的
    要求，看到「[事实]」会知道这是背景信息、必要时可以质疑。
    不标注的话，两者在模型眼里是一样的，容易把一条临时事实当成硬性要求。
    """
    if not memories:
        return ""
    label = {"preference": "偏好", "fact": "事实", "term": "术语"}
    lines = ["（以下是从以往对话中记住的信息，供参考）"]
    for memory in memories:
        lines.append(f"- [{label.get(memory.kind, '信息')}] {memory.content}")
    return "\n".join(lines)


def _recall_memory(question: str) -> str:
    """召回长期记忆。失败一律返回空字符串——记忆是增强，不该拖垮问答。"""
    if not settings.enable_long_term_memory:
        return ""
    try:
        memories = get_memory_store().recall(question, user_id=DEFAULT_USER_ID)
        return format_memories(memories)
    except Exception as exc:  # noqa: BLE001
        logger.warning("长期记忆召回失败：%s: %s", type(exc).__name__, exc)
        return ""


def _citations_from_cached(sources: list[dict]) -> list[CitationEvent]:
    """缓存里的引用（dict）转成事件。

    字段缺失时给空值而不是报错：缓存可能是旧版本写进去的，
    格式演进不该让命中缓存的用户拿不到答案。
    """
    events: list[CitationEvent] = []
    for item in sources or []:
        events.append(CitationEvent(
            doc_id=str(item.get("doc_id") or ""),
            chunk=str(item.get("chunk") or ""),
            source=str(item.get("source") or ""),
            score=float(item.get("score") or 0.0),
            page_title=str(item.get("page_title") or ""),
            section_path=list(item.get("section_path") or []),
        ))
    return events


def _citation_to_dict(event: CitationEvent) -> dict:
    return {
        "doc_id": event.doc_id,
        "chunk": event.chunk,
        "source": event.source,
        "score": event.score,
        "page_title": event.page_title,
        "section_path": event.section_path,
    }


async def stream_agent(
    *,
    question: str,
    history: list[dict],
    summary: str | None,
    tools: MCPToolClient,
    max_iterations: int = 8,
    conversation_id: int | None = None,
) -> AsyncIterator[AgentEvent]:
    # ------------------------------------------------------------------
    # ① 语义缓存：命中就跳过检索与生成
    # ------------------------------------------------------------------
    if settings.enable_semantic_cache:
        cached: CachedAnswer | None = None
        try:
            cached = await asyncio.to_thread(get_semantic_cache().get, question)
        except Exception as exc:  # noqa: BLE001 — 缓存故障退化为「没有缓存」
            logger.warning("语义缓存查询异常：%s: %s", type(exc).__name__, exc)

        if cached is not None:
            yield CacheHitEvent(
                matched_query=cached.matched_query,
                score=round(cached.score, 4),
                age_hours=cached.age_hours,
            )
            for event in _citations_from_cached(cached.sources):
                yield event
            yield FinalEvent(answer=cached.answer)
            return

    # ------------------------------------------------------------------
    # ② 长期记忆召回（③ 短期记忆由调用方通过 history/summary 传入）
    # ------------------------------------------------------------------
    memory_text = await asyncio.to_thread(_recall_memory, question)

    task = AgentTask(
        question=question,
        history=to_langchain_messages(history),
        summary=summary or "",
        long_term_memory=memory_text,
    )
    model = get_chat_model()
    deps = AgentDeps(
        model=model,
        tools=tools,
        context=build_context_engine(),
        max_iterations=max_iterations,
        # 规划复用同一个模型实例：规划只需要一次短输出（一个 JSON），
        # 没必要为它单独配置模型。关掉时传 None，走「不规划、不预检索」，
        # 行为与迁移前的自研循环一致。
        planner=model if settings.enable_planning else None,
    )

    answer = ""
    citations: list[dict] = []

    async for event in run_agent_stream(task, deps):
        # 顺手把答案与引用收集起来，供 ④ 写缓存用
        if isinstance(event, FinalEvent):
            answer = event.answer
        elif isinstance(event, CitationEvent):
            citations.append(_citation_to_dict(event))
        yield event

    # ------------------------------------------------------------------
    # ④ 写缓存 + 异步抽记忆
    # ------------------------------------------------------------------
    if not answer:
        # 出错或达到迭代上限：没有答案，不写缓存（半成品不该被复用）
        return

    if settings.enable_semantic_cache:
        try:
            await asyncio.to_thread(
                get_semantic_cache().put, question, answer, citations
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("写语义缓存失败：%s: %s", type(exc).__name__, exc)

    # **这一步不 await**：抽取要调 LLM（约 1 秒），
    # 放在这里会拖慢已经完成的回答。已知限制：任务依附于当前请求的事件循环，
    # 客户端断开时可能被取消——生产环境应交给 Celery（本项目暂无）。
    schedule_extraction(
        question,
        answer,
        model=model,
        conversation_id=conversation_id,
    )
