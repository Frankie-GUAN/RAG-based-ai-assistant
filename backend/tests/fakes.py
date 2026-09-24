"""测试替身：按脚本产出 AIMessageChunk 的假模型。

不用 LangChain 自带的 FakeListChatModel —— 它只支持纯文本，
无法模拟流式 tool_call_chunks 的增量合并。
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessageChunk, BaseMessage


class ScriptedChatModel:
    """script 是「每一轮 LLM 调用产出哪些 chunk」的列表。

    例如两轮：先调工具，再出答案 →
        ScriptedChatModel([
            tool_turn("search_documents", {"query": "x"}),
            text_turn("答案", "是……"),
        ])
    """

    def __init__(self, script: list[list[AIMessageChunk]]) -> None:
        self._script = [list(turn) for turn in script]
        self.calls: list[list[BaseMessage]] = []
        self.bound_tools: list[dict[str, Any]] = []

    def bind_tools(self, tools: list[dict[str, Any]]) -> "ScriptedChatModel":
        self.bound_tools = list(tools)
        return self

    async def astream(self, messages: list[BaseMessage]) -> AsyncIterator[AIMessageChunk]:
        self.calls.append(list(messages))

        if not self._script:
            raise AssertionError(
                f"脚本已耗尽：模型被调用了第 {len(self.calls)} 次，但没有预设响应"
            )

        for chunk in self._script.pop(0):
            yield chunk


def text_turn(*pieces: str) -> list[AIMessageChunk]:
    """一次纯文本回答，拆成多个 chunk 以模拟真实流式。"""
    return [AIMessageChunk(content=piece) for piece in pieces]


def tool_turn(name: str, args: dict[str, Any], call_id: str = "call_1") -> list[AIMessageChunk]:
    """一次工具调用。刻意拆成两个 chunk，复现真实的流式参数增量合并。"""
    return [
        AIMessageChunk(content="", tool_call_chunks=[
            {"name": name, "args": "", "id": call_id, "index": 0}
        ]),
        AIMessageChunk(content="", tool_call_chunks=[
            {"name": None, "args": json.dumps(args), "id": None, "index": 0}
        ]),
    ]


def two_tool_turn(
    first: tuple[str, dict[str, Any], str],
    second: tuple[str, dict[str, Any], str],
) -> list[AIMessageChunk]:
    """一轮内并发发起两个工具调用。"""
    return [
        AIMessageChunk(content="", tool_call_chunks=[
            {"name": first[0], "args": json.dumps(first[1]), "id": first[2], "index": 0},
            {"name": second[0], "args": json.dumps(second[1]), "id": second[2], "index": 1},
        ]),
    ]
