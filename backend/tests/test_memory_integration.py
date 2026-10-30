"""三层记忆接入 `stream_agent` 的测试。

**这一层的价值是验证「接线对了」。** 存储层和缓存层各自的逻辑已经单测过，
但如果 `chat_service` 忘了调用它们、或者调用顺序错了，前面的工作就全白费——
而且这种失败是静默的：问答照常能跑，只是缓存从不命中、记忆从不注入。

所以这里刻意用替身把三层都「探针化」，断言它们真的被调用、参数正确。
"""
from __future__ import annotations

import pytest

from app.agent.protocol import (
    AgentEvent,
    CacheHitEvent,
    CitationEvent,
    FinalEvent,
    TokenEvent,
)
from app.cache.semantic import CachedAnswer


class _StubTools:
    """MCP 工具客户端替身。"""

    specs: list = []

    async def call(self, name, args):
        raise AssertionError("本用例不该真的调工具")


async def _collect(agen):
    """把事件流收成列表。

    **必须是 async**：`stream_agent` 是异步生成器，
    同步的 `list()` 拿到的是 async_generator 对象本身。
    """
    return [event async for event in agen]


@pytest.fixture
def patched(monkeypatch):
    """把三层依赖全换成替身，并记录调用。"""
    from app.services import chat_service

    state = {
        "cache_get_calls": [],
        "cache_put_calls": [],
        "recall_calls": [],
        "extract_calls": [],
        "cached_value": None,
        "memory_text": "",
        "captured_task": None,
    }

    class _FakeCache:
        def get(self, query):
            state["cache_get_calls"].append(query)
            return state["cached_value"]

        def put(self, query, answer, sources=None):
            state["cache_put_calls"].append((query, answer, sources))

    class _FakeStore:
        def recall(self, query, **kwargs):
            state["recall_calls"].append(query)
            return []

    monkeypatch.setattr(
        chat_service, "get_semantic_cache", lambda: _FakeCache()
    )
    monkeypatch.setattr(chat_service, "get_memory_store", lambda: _FakeStore())

    # _recall_memory 返回受控文本
    monkeypatch.setattr(
        chat_service, "_recall_memory", lambda q: state["memory_text"]
    )

    # 捕获 AgentTask，并产出一条最小的正常事件流
    async def _fake_agent_stream(task, deps):
        state["captured_task"] = task
        yield TokenEvent(text="答案")
        yield CitationEvent(doc_id="d1", source="a.pdf", chunk="片段")
        yield FinalEvent(answer="完整答案")

    monkeypatch.setattr(chat_service, "run_agent_stream", _fake_agent_stream)

    # 记录记忆抽取调度
    def _fake_schedule(question, answer, *, model, conversation_id=None):
        state["extract_calls"].append((question, answer, conversation_id))
        return None

    monkeypatch.setattr(chat_service, "schedule_extraction", _fake_schedule)

    # 打开两个开关（conftest 默认关闭）
    from app.config import settings

    monkeypatch.setattr(settings, "enable_semantic_cache", True)
    monkeypatch.setattr(settings, "enable_long_term_memory", True)

    return chat_service, state


class TestCacheShortCircuit:
    @pytest.mark.anyio
    async def test_cache_hit_skips_agent_entirely(self, patched):
        """**缓存命中的全部意义就是跳过后续所有步骤。**

        断言 `captured_task is None` 是关键：它证明 Agent 图根本没被调用。
        只断言「返回了缓存答案」是不够的——那样即使走了完整链路再丢掉结果，
        测试也会通过。
        """
        chat_service, state = patched
        state["cached_value"] = CachedAnswer(
            answer="缓存的答案",
            sources=[{"doc_id": "d1", "source": "a.pdf", "chunk": "片段"}],
            matched_query="原来的问法",
            score=0.97,
            age_hours=2.5,
        )

        events = await _collect(
            e async for e in chat_service.stream_agent(
                question="相似的新问法", history=[], summary=None, tools=_StubTools()
            )
        )

        types = [e.type for e in events]
        assert types == ["cache_hit", "citation", "final"]
        assert state["captured_task"] is None, "命中缓存却还是跑了 Agent"
        assert state["cache_put_calls"] == [], "命中缓存不该再写一次缓存"

    @pytest.mark.anyio
    async def test_cache_hit_carries_provenance(self, patched):
        """命中事件必须带上原问题与分数。

        这是缓存的安全阀：用户看到被复用的问题措辞不同，可以自己判断
        这个答案靠不靠谱。把这些藏起来等于让用户承担误判后果却不知情。
        """
        chat_service, state = patched
        state["cached_value"] = CachedAnswer(
            answer="答案",
            matched_query="活跃变量分析怎么定义",
            score=0.93,
            age_hours=1.0,
        )

        events = await _collect(
            e async for e in chat_service.stream_agent(
                question="活跃变量分析有什么用", history=[], summary=None, tools=_StubTools()
            )
        )

        hit = next(e for e in events if isinstance(e, CacheHitEvent))
        assert hit.matched_query == "活跃变量分析怎么定义"
        assert hit.score == 0.93
        assert hit.age_hours == 1.0

    @pytest.mark.anyio
    async def test_miss_falls_through_to_agent(self, patched):
        chat_service, state = patched
        state["cached_value"] = None

        events = await _collect(
            e async for e in chat_service.stream_agent(
                question="新问题", history=[], summary=None, tools=_StubTools()
            )
        )

        assert state["captured_task"] is not None, "未命中缓存却没走 Agent"
        assert [e.type for e in events] == ["token", "citation", "final"]


class TestMemoryInjection:
    @pytest.mark.anyio
    async def test_recalled_memory_is_injected_into_task(self, patched):
        """召回到的记忆必须真的进 AgentTask。

        这是最容易静默失败的一环：`_recall_memory` 被调用了、
        但返回值没被使用 —— 那样记忆功能形同虚设，而所有测试都可能照常通过。
        """
        chat_service, state = patched
        state["memory_text"] = "【用户长期偏好与已确认事实】\n- [偏好] 回答要简短"

        await _collect(
            e async for e in chat_service.stream_agent(
                question="随便问点", history=[], summary=None, tools=_StubTools()
            )
        )

        task = state["captured_task"]
        assert task is not None
        assert "回答要简短" in task.long_term_memory

    @pytest.mark.anyio
    async def test_no_memory_still_runs(self, patched):
        chat_service, state = patched
        state["memory_text"] = ""

        events = await _collect(
            e async for e in chat_service.stream_agent(
                question="q", history=[], summary=None, tools=_StubTools()
            )
        )

        assert state["captured_task"].long_term_memory == ""
        assert events[-1].type == "final"


class TestWriteBack:
    @pytest.mark.anyio
    async def test_answer_is_cached_after_completion(self, patched):
        """答完之后要把答案写进缓存，且带上引用。

        引用必须一起缓存：命中缓存的答案若没有来源标签，
        可信度会掉一半——用户看到一段没有出处的断言。
        """
        chat_service, state = patched

        await _collect(
            e async for e in chat_service.stream_agent(
                question="问题", history=[], summary=None, tools=_StubTools()
            )
        )

        assert len(state["cache_put_calls"]) == 1
        query, answer, sources = state["cache_put_calls"][0]
        assert query == "问题"
        assert answer == "完整答案"
        assert sources and sources[0]["doc_id"] == "d1"

    @pytest.mark.anyio
    async def test_error_answer_is_not_cached(self, patched):
        """没有答案（出错 / 达到迭代上限）时不写缓存。

        半成品不该被复用——把一次失败的回答缓存下来，
        之后所有相似问题都会拿到同一个失败结果。
        """
        chat_service, state = patched

        async def _empty_stream(task, deps):
            from app.agent.protocol import ErrorEvent

            state["captured_task"] = task
            yield ErrorEvent(code="max_iterations", message="超限")

        import app.services.chat_service as cs

        patched_module = cs
        cs.run_agent_stream = _empty_stream

        await _collect(
            e async for e in patched_module.stream_agent(
                question="q", history=[], summary=None, tools=_StubTools()
            )
        )

        assert state["cache_put_calls"] == [], "失败的回答不该被缓存"

    @pytest.mark.anyio
    async def test_memory_extraction_is_scheduled_with_conversation_id(self, patched):
        """记忆抽取要被调度，且带上会话 id 以便溯源。"""
        chat_service, state = patched

        await _collect(
            e async for e in chat_service.stream_agent(
                question="记住我用 MySQL 8",
                history=[],
                summary=None,
                tools=_StubTools(),
                conversation_id=42,
            )
        )

        assert len(state["extract_calls"]) == 1
        question, answer, conv_id = state["extract_calls"][0]
        assert conv_id == 42
        assert answer == "完整答案"
