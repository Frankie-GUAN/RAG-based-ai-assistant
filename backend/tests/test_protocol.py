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


def _event_subclasses(cls=None) -> list:
    """递归收集 cls 的全部后代，而不只是直接子类。

    只取直接子类的话 `class GrandChild(TokenEvent)` 会隐形 —— 它同样能被实例化、
    同样产出可序列化却无人登记的 type，正是本模块要排除的那类东西。
    """
    found = []
    for sub in (cls or AgentEvent).__subclasses__():
        found.append(sub)
        found.extend(_event_subclasses(sub))
    return found


def test_every_declared_event_type_is_covered():
    """守护「新增事件类型后必须同步前端契约」。

    遍历 AgentEvent 的**全部后代**，而不是在测试里列一份手写清单 —— 手写清单只能
    证明「清单与 EXPECTED 一致」，证明不了「模块里的类型都已登记」。实测：新增一种
    子类却两处都忘了同步时，手写清单版本会静默通过。

    注意 `__subclasses__()` 是**进程级**的：任何地方定义的 AgentEvent 后代都会进来。
    那正是这个守护想要的覆盖面，但也意味着失败信息必须说清这个类**来自哪个模块** ——
    否则报错落在 test_protocol.py 上，看上去像协议契约变了，实际是别的测试模块里
    定义了一个子类。

    基类 AgentEvent 不参与（它已被禁止实例化，不产出 type）。
    断言用 == 而非 <=：== 独有的价值是抓住 `EXPECTED_EVENT_TYPES` 里**预登记了一个
    没有子类产出的类型** —— 一个无生产者的契约条目，正是本模块针对的那种味道。
    """
    subs = _event_subclasses()
    declared = {cls.type for cls in subs}
    origin = {cls.type: f"{cls.__module__}.{cls.__qualname__}" for cls in subs}

    assert len(declared) == len(subs), (
        f"有事件类声明了重复的 type："
        f"{sorted(t for t in declared if [c.type for c in subs].count(t) > 1)}"
    )

    assert declared == EXPECTED_EVENT_TYPES, (
        "事件类型与前端契约不一致。\n"
        f"  后端多了 {[(t, origin[t]) for t in sorted(declared - EXPECTED_EVENT_TYPES)]}\n"
        f"  契约里没有子类产出的 {sorted(EXPECTED_EVENT_TYPES - declared)}"
    )


def test_base_class_cannot_be_instantiated():
    """基类可实例化的话，AgentEvent().to_dict() 会给出 {"type": "event"} ——
    又一个可序列化但没有生产者的类型。本模块的原则是这种类型不存在。

    匹配 "AgentEvent" 而不是中文措辞：改文案不该让用例碎掉。
    """
    with pytest.raises(TypeError, match="AgentEvent"):
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
