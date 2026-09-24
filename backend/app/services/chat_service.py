"""对话服务：把会话历史交给 Agent Core，产出事件流。"""
from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agent.core import AgentDeps, AgentTask, run_agent_stream
from app.agent.llm import get_chat_model
from app.agent.protocol import AgentEvent
from app.context.engine import ContextBudget, ContextEngine
from app.context.tokens import HeuristicTokenCounter
from app.config import settings
from app.mcp.client import MCPToolClient


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


async def stream_agent(
    *,
    question: str,
    history: list[dict],
    summary: str | None,
    tools: MCPToolClient,
    max_iterations: int = 8,
) -> AsyncIterator[AgentEvent]:
    task = AgentTask(
        question=question,
        history=to_langchain_messages(history),
        summary=summary or "",
    )
    deps = AgentDeps(
        model=get_chat_model(),
        tools=tools,
        context=build_context_engine(),
        max_iterations=max_iterations,
    )

    async for event in run_agent_stream(task, deps):
        yield event
