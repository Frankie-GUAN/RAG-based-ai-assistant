"""SSE 编码契约测试。

锁的是「前端能依赖什么」：事件名与 payload 字段。这些断言在协议落地前就已成立，
它们的作用是把 `to_sse` 的输出钉住 —— 前端按 `event:` 名分发、按 payload 字段取值，
任何一侧漂移都会在这里响亮地失败，而不是在前端静默变成「什么都不显示」。
"""
import json

from app.agent.protocol import (
    ErrorEvent,
    FinalEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    to_sse,
)


def test_token_event_maps_to_sse_frame():
    frame = to_sse(TokenEvent(text="检索"))

    assert frame["event"] == "token"
    assert json.loads(frame["data"])["type"] == "token"


def test_frames_can_be_reassembled_into_the_full_answer():
    """前端按顺序拼接 token 事件必须还原出完整答案。"""
    pieces = ["根据", "文档", "内容", "……"]
    reconstructed = "".join(
        json.loads(to_sse(TokenEvent(text=p))["data"])["text"] for p in pieces
    )

    assert reconstructed == "根据文档内容……"


def test_tool_events_carry_enough_for_a_trace_view():
    call = json.loads(to_sse(ToolCallEvent(id="c1", name="search_documents", args={"query": "x"}))["data"])
    result = json.loads(to_sse(ToolResultEvent(id="c1", ok=True, data={"n": 1}, ms=42))["data"])

    assert call["name"] == "search_documents"
    assert call["args"] == {"query": "x"}
    assert result["ms"] == 42
    assert result["id"] == call["id"]


def test_error_event_is_terminal_and_machine_readable():
    payload = json.loads(to_sse(ErrorEvent(code="max_iterations", message="超限", recoverable=True))["data"])

    assert payload["type"] == "error"
    assert payload["code"] == "max_iterations"
    assert payload["recoverable"] is True


def test_final_event_carries_the_answer_without_chunking():
    """最终答案必须是完整字符串，前端不必再拼接 done 事件。"""
    payload = json.loads(to_sse(FinalEvent(answer="完整答案"))["data"])

    assert payload["answer"] == "完整答案"


def test_data_is_single_line_and_keeps_chinese_readable():
    """SSE 是行协议：data 里出现裸换行会把前端按行解析打乱。"""
    frame = to_sse(FinalEvent(answer="第一行\n第二行"))

    assert "\n" not in frame["data"]
    assert "第一行" in frame["data"]          # ensure_ascii=False
    assert json.loads(frame["data"])["answer"] == "第一行\n第二行"