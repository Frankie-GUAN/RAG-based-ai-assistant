import json

import pytest

from app.agent.protocol import (
    AgentEvent,
    ErrorEvent,
    FinalEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    to_sse,
)

# 与前端事件分发共享的契约（frontend/src/composables/useChat.ts；Task 8 才实现）。
# 任何一边增删事件类型，此集合必须同步更新 —— 下面的守卫测试会强制这一点。
EXPECTED_EVENT_TYPES = {
    "tool_call",
    "tool_result",
    "token",
    "final",
    "error",
}


def test_token_event_serializes_with_type_key():
    assert TokenEvent(text="你好").to_dict() == {"type": "token", "text": "你好"}


def test_tool_call_carries_args():
    payload = ToolCallEvent(id="call_1", name="search_documents", args={"query": "合同解除"}).to_dict()

    assert payload["type"] == "tool_call"
    assert payload["args"] == {"query": "合同解除"}


def test_tool_result_records_failure_without_raising():
    payload = ToolResultEvent(id="call_1", ok=False, error="TimeoutError: boom", ms=1200).to_dict()

    assert payload["ok"] is False
    assert payload["error"] == "TimeoutError: boom"
    assert payload["ms"] == 1200


def test_final_event_carries_the_full_answer():
    payload = FinalEvent(answer="劳动合同经双方协商一致可以解除").to_dict()

    assert payload["answer"] == "劳动合同经双方协商一致可以解除"


def test_error_event_is_machine_readable():
    payload = ErrorEvent(code="max_iterations", message="达到迭代上限", recoverable=True).to_dict()

    assert payload["type"] == "error"
    assert payload["code"] == "max_iterations"
    assert payload["recoverable"] is True


def test_every_declared_event_type_is_covered():
    """守护「新增事件类型后必须同步前端契约」。

    这里**遍历 AgentEvent 的子类**，而不是在测试里列一份手写清单 —— 手写清单只能
    证明「清单与 EXPECTED 一致」，证明不了「模块里的类型都已登记」。实测：新增第六种
    子类却两处都忘了同步时，手写清单版本会静默通过，自动发现版本直接失败。

    基类 AgentEvent 不参与：它已被禁止实例化，不会产出任何 type。
    断言用 == 而非 <=：多一个（后端加了未登记类型）或少一个（前端留了已删类型）
    都说明两边脱节，都该失败。
    """
    declared = {cls.type for cls in AgentEvent.__subclasses__()}

    assert declared == EXPECTED_EVENT_TYPES, (
        f"事件类型与前端契约不一致：后端多了 {declared - EXPECTED_EVENT_TYPES}，"
        f"契约多了 {EXPECTED_EVENT_TYPES - declared}"
    )


def test_base_class_cannot_be_instantiated():
    """基类可实例化的话，AgentEvent().to_dict() 会给出 {"type": "event"} ——
    又一个可序列化但没有生产者的类型。本模块的原则是这种类型不存在。"""
    with pytest.raises(TypeError, match="基类"):
        AgentEvent()


def test_to_sse_keeps_chinese_readable():
    frame = to_sse(TokenEvent(text="检索"))

    assert frame["event"] == "token"
    assert "检索" in frame["data"]  # ensure_ascii=False
    assert json.loads(frame["data"]) == {"type": "token", "text": "检索"}


def test_to_sse_data_is_single_line():
    """SSE 是行协议，data 里出现裸换行会破坏前端按行解析。"""
    frame = to_sse(FinalEvent(answer="第一行\n第二行"))

    assert "\n" not in frame["data"]
