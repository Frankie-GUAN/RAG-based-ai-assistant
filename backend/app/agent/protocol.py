"""Agent 事件协议（Python 端）。

只定义 P0 真正会发出的事件 —— 不留「定义了但无生产者」的类型，
那正是本计划要消灭的那类问题。所有事件继承 AgentEvent，
to_dict() 序列化为 {"type": <事件名>, ...payload}。

新增事件类型时，必须同步：
  1. 本文件
  2. tests/test_protocol.py 的 EXPECTED_EVENT_TYPES —— 那条测试自动遍历
     AgentEvent 的子类，漏登记会直接失败
  3. 前端的事件分发（frontend/src/composables/useChat.ts；Task 8 才实现，
     在此之前该文件仍是旧的 content/done 行协议）
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True)
class AgentEvent:
    """所有事件的基类。子类用 ClassVar 声明 type 名（ClassVar 不参与 asdict）。

    禁止直接实例化：基类若能实例化，to_dict() 就会给出 {"type": "event"} ——
    又一个「可序列化但没有生产者」的类型，正是本模块要消灭的那类东西。
    """

    type: ClassVar[str] = "event"

    def __post_init__(self) -> None:
        if type(self) is AgentEvent:
            raise TypeError("AgentEvent 是基类，不应被实例化；请使用具体的子类事件")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = self.type
        return payload


@dataclass(frozen=True)
class ToolCallEvent(AgentEvent):
    type: ClassVar[str] = "tool_call"
    id: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResultEvent(AgentEvent):
    type: ClassVar[str] = "tool_result"
    id: str = ""
    ok: bool = True
    data: Any = None
    ms: int = 0
    error: str | None = None


@dataclass(frozen=True)
class TokenEvent(AgentEvent):
    type: ClassVar[str] = "token"
    text: str = ""


@dataclass(frozen=True)
class FinalEvent(AgentEvent):
    type: ClassVar[str] = "final"
    answer: str = ""


@dataclass(frozen=True)
class ErrorEvent(AgentEvent):
    type: ClassVar[str] = "error"
    code: str = ""
    message: str = ""
    recoverable: bool = False


def to_sse(event: AgentEvent) -> dict[str, str]:
    """转成 sse_starlette 接受的 dict 形式。ensure_ascii=False 保持中文可读。"""
    return {"event": event.type, "data": json.dumps(event.to_dict(), ensure_ascii=False, default=str)}
