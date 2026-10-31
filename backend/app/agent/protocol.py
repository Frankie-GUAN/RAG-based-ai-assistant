"""Agent 事件协议（Python 端）。

只定义 P0 真正会发出的事件 —— 不留「定义了但无生产者」的类型，
那正是本计划要消灭的那类问题。所有事件继承 AgentEvent，
to_dict() 序列化为 {"type": <事件名>, ...payload}。

新增事件类型时，必须同步：
  1. 本文件
  2. tests/test_protocol.py 的 EXPECTED_EVENT_TYPES —— 那条测试自动遍历
     AgentEvent 的子类，漏登记会直接失败
  3. 前端的事件分发（frontend/src/composables/useChat.ts，按 payload 的 type 分发）
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
class CitationEvent(AgentEvent):
    """一条检索引用，供前端展示来源、供会话落库。

    这是「回答可信」的唯一凭据：模型说「根据文档」，用户需要能点开看它依据的是
    哪一段。`chunk_id` 指向 `rag/loader.py` 打的稳定标识，因此重新入库之后旧引用
    依然指向同一段内容。

    `page_title` 与 `section_path` 指向「哪一节」——只有文件名时，
    用户拿到一句答案也不知道它出自哪一章；有了章节路径，
    引用才能真正用于核对。
    """

    type: ClassVar[str] = "citation"
    doc_id: str = ""
    chunk: str = ""
    source: str = ""
    score: float = 0.0
    page_title: str = ""
    section_path: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PlanEvent(AgentEvent):
    """规划节点的决策结果。

    **为什么要把规划过程外发。** 用户看到「正在规划」和「正在检索」是两种
    完全不同的体感：前者说明系统在理解问题，后者说明它在找资料。
    更重要的是——它能解释「为什么这次没检索」：
    用户问「你好」时直接回答是对的，但如果没有这个事件，
    用户会以为系统漏了检索。

    `reason` 只在需要时给一两句话，不写长篇大论——它是进度提示，不是日志。
    """

    type: ClassVar[str] = "plan"
    need_retrieval: bool = True
    query: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ReflectEvent(AgentEvent):
    """反思节点的自检结论。

    **这是 Agent 与「一次调用」的分水岭。** 只有 Reflection 才能真正发现
    「我检索了但没用上」「资料不足却硬答」这类失败，而不是把它们
    当成正常输出交出去。

    `need_more` 为真时会触发一次补充检索；为假时直接进入作答。
    `reason` 说明判断依据，让这条自检可追溯、可审计。
    """

    type: ClassVar[str] = "reflect"
    need_more: bool = False
    reason: str = ""


@dataclass(frozen=True)
class CacheHitEvent(AgentEvent):
    """语义缓存命中。

    **这个事件是缓存功能的安全阀，不是装饰。** 语义缓存按相似度复用答案，
    而相似度永远有误判的可能——实测「活跃变量分析怎么定义」与
    「活跃变量分析能用来做什么」相似度高达 0.899（见 verified-facts §11.3）。

    `matched_query` 给出被复用的**原问题原文**：用户看到措辞不同的问题被复用，
    可以自己判断这个答案靠不靠谱。把它藏起来，等于让用户承担误判的后果
    却不知道发生了什么。

    `score` 同时给出，让「哪些问题会命中缓存」这件事可观测、可调参。
    """

    type: ClassVar[str] = "cache_hit"
    matched_query: str = ""
    score: float = 0.0
    age_hours: float = 0.0


@dataclass(frozen=True)
class ErrorEvent(AgentEvent):
    type: ClassVar[str] = "error"
    code: str = ""
    message: str = ""
    recoverable: bool = False


def to_sse(event: AgentEvent) -> dict[str, str]:
    """转成 sse_starlette 接受的 dict 形式。ensure_ascii=False 保持中文可读。"""
    return {"event": event.type, "data": json.dumps(event.to_dict(), ensure_ascii=False, default=str)}
