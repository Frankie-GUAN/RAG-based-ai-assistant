import json

from app.agent.protocol import (
    ErrorEvent,
    FinalEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    to_sse,
)

# 与前端事件分发（frontend/src/composables/useChat.ts）共享的契约。
# 任何一边增删事件类型，此集合必须同步更新。
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
    """防止新增事件类型后忘记同步前端契约。"""
    declared = {
        ToolCallEvent.type,
        ToolResultEvent.type,
        TokenEvent.type,
        FinalEvent.type,
        ErrorEvent.type,
    }

    assert declared <= EXPECTED_EVENT_TYPES, f"出现了未登记的事件类型: {declared - EXPECTED_EVENT_TYPES}"


def test_to_sse_keeps_chinese_readable():
    frame = to_sse(TokenEvent(text="检索"))

    assert frame["event"] == "token"
    assert "检索" in frame["data"]  # ensure_ascii=False
    assert json.loads(frame["data"]) == {"type": "token", "text": "检索"}


def test_to_sse_data_is_single_line():
    """SSE 是行协议，data 里出现裸换行会破坏前端按行解析。"""
    frame = to_sse(FinalEvent(answer="第一行\n第二行"))

    assert "\n" not in frame["data"]
