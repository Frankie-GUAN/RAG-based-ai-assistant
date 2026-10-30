"""长期记忆的测试。

分三层：
1. **纯函数**（规则判定、JSON 解析、渲染）—— 零依赖，覆盖核心判断逻辑
2. **存储层** —— 用内存 SQLite + 假 Chroma 替身，覆盖去重与召回
3. **接入层** —— 验证记忆确实被注入到 system prompt

为什么第 2 层要用替身而不是真 Chroma：真 Chroma 要加载 2GB 的 BGE-M3，
单测不该为此付几十秒；而且这里要验的是「去重与映射逻辑」，
不是「向量检索准不准」（那属于检索评测的范围）。
"""
from __future__ import annotations

import pytest

from app.memory.extractor import _parse_memories, detect_signal


# ---------------------------------------------------------------------------
# 1. 纯函数
# ---------------------------------------------------------------------------

class TestDetectSignal:
    def test_preference_signals(self):
        for text in [
            "以后都回答简short一点",
            "我偏好带代码示例的回答",
            "能不能简短些",
            "太长了",
            "我喜欢先给结论",
        ]:
            assert detect_signal(text) == "preference", f"未识别偏好信号：{text!r}"

    def test_fact_signals(self):
        for text in [
            "记住我们的知识库是编译器课件",
            "实际上我们用的是 MySQL 8",
            "不是这样，应该是常量传播",
            "我们的项目用 Python",
        ]:
            assert detect_signal(text) == "fact", f"未识别事实信号：{text!r}"

    def test_ordinary_question_does_not_trigger(self):
        """普通提问不该触发抽取——否则每轮都多一次 LLM 调用。

        **这条用例是成本控制的红线。** 规则一旦放宽，
        用户每问一个技术问题都会白发一次 LLM 请求。
        """
        for text in [
            "常量传播分析怎么做",
            "什么是格和偏序",
            "调用图怎么构建",
            "worklist 算法如何做指针分析",
            "谢谢",
        ]:
            assert detect_signal(text) is None, f"普通问题被误判为记忆信号：{text!r}"

    def test_empty_and_none(self):
        assert detect_signal("") is None
        assert detect_signal(None) is None  # type: ignore[arg-type]


class TestParseMemories:
    def test_plain_json_array(self):
        raw = '[{"kind": "preference", "content": "用户偏好简短回答", "confidence": 0.9}]'
        parsed = _parse_memories(raw)
        assert len(parsed) == 1
        assert parsed[0]["content"] == "用户偏好简短回答"

    def test_fenced_json_block(self):
        """模型经常把 JSON 包在代码块里 —— 解析必须能处理这一点。

        解析失败等于「这轮白抽了」：用户以为说过了，系统没记住。
        """
        raw = '```json\n[{"kind": "fact", "content": "知识库是编译器课件"}]\n```'
        parsed = _parse_memories(raw)
        assert len(parsed) == 1
        assert "编译器" in parsed[0]["content"]

    def test_json_with_leading_explanation(self):
        raw = '好的，我抽取到：\n[{"kind": "fact", "content": "用的是 MySQL"}]\n以上。'
        parsed = _parse_memories(raw)
        assert len(parsed) == 1

    def test_empty_array(self):
        assert _parse_memories("[]") == []

    def test_garbage_returns_empty(self):
        for raw in ["", "没有可记的内容", "{不是 JSON}", "[坏掉的"]:
            assert _parse_memories(raw) == [], f"垃圾输入未被兜住：{raw!r}"

    def test_single_object_is_accepted(self):
        """模型有时只给一个对象而不是数组，也要能接住。"""
        raw = '{"kind": "fact", "content": "用 MySQL 8"}'
        parsed = _parse_memories(raw)
        assert len(parsed) == 1
        assert "MySQL" in parsed[0]["content"]

    def test_filters_non_dict_items(self):
        raw = '[{"kind": "fact", "content": "a"}, "字符串", 123]'
        parsed = _parse_memories(raw)
        assert all(isinstance(x, dict) for x in parsed)
        assert len(parsed) == 1


class TestFormatMemories:
    def test_renders_with_kind_labels(self):
        """必须带 kind 标签。

        模型看到「[偏好]」才知道这是要长期遵守的要求；
        看到「[事实]」知道这是背景信息。不标注的话两者在模型眼里一样，
        容易把一条临时事实当成硬性要求。
        """
        from app.services.chat_service import format_memories

        class _M:
            def __init__(self, kind, content):
                self.kind = kind
                self.content = content

        text = format_memories([_M("preference", "偏好简短回答"), _M("fact", "用 MySQL")])
        assert "[偏好]" in text
        assert "[事实]" in text
        assert "偏好简短回答" in text

    def test_empty_returns_empty(self):
        from app.services.chat_service import format_memories

        assert format_memories([]) == ""


# ---------------------------------------------------------------------------
# 2. 存储层：用内存 SQLite + 假 Chroma 替身
# ---------------------------------------------------------------------------

class _FakeCollection:
    """Chroma collection 的最小替身。

    只实现 store.py 用到的三个方法。**按余弦模拟相似度**：
    这里不追求真实语义，只保证「相同文本得高分、不同文本得低分」，
    足够验证召回链路。
    """

    def __init__(self):
        self.docs: dict[str, str] = {}
        self.metas: dict[str, dict] = {}

    def add_texts(self, texts, metadatas=None, ids=None, **kwargs):
        for i, text in enumerate(texts):
            key = str(ids[i])
            self.docs[key] = text
            self.metas[key] = dict((metadatas or [])[i])

    def delete(self, ids=None, where=None, **kwargs):
        for key in list(ids or []):
            self.docs.pop(str(key), None)
            self.metas.pop(str(key), None)

    def similarity_search_with_relevance_scores(self, query, k=4, **kwargs):
        from langchain_core.documents import Document

        results = []
        for key, text in self.docs.items():
            # 极简相似度：字符重合比例。够用来区分「同一句话」和「别的话」
            overlap = len(set(query) & set(text))
            score = overlap / max(len(set(text)), 1)
            results.append((Document(page_content=text, metadata=self.metas[key]), score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


@pytest.fixture
def memory_db():
    """独立的内存库，只建 user_memories 表。"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.db.base import Base
    from app.models.conversation import Conversation
    from app.models.memory import UserMemory
    # Message 必须导入：Conversation.messages 的 relationship 引用了它，
    # 不导入会在 mapper 初始化时报 "failed to locate a name 'Message'"
    from app.models.message import Message  # noqa: F401

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    # 必须连 conversations 一起建：user_memories.source_conversation_id
    # 有指向它的外键，只建记忆表会报 NoReferencedTableError
    Base.metadata.create_all(
        engine, tables=[Conversation.__table__, UserMemory.__table__]
    )
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def store(monkeypatch, memory_db):
    """接上假 Chroma 的 MemoryStore。"""
    from app.memory.store import MemoryStore

    fake = _FakeCollection()
    s = MemoryStore()
    monkeypatch.setattr(s, "_collection", lambda: fake)
    monkeypatch.setattr(s, "_chroma", fake)
    # 让 store 用测试库而不是真实 MySQL
    monkeypatch.setattr(
        "app.memory.store.SessionLocal", lambda: memory_db
    )
    # memory_db 不该在 store 内部被 close
    monkeypatch.setattr(
        memory_db.__class__, "close", lambda self: None, raising=False
    )
    return s


class TestMemoryStore:
    def test_remember_then_list(self, store):
        assert store.remember("用户偏好简短回答", kind="preference") is True
        rows = store.list_all()
        assert len(rows) == 1
        assert rows[0].content == "用户偏好简短回答"
        assert rows[0].kind == "preference"

    def test_duplicate_is_rejected(self, store):
        """同一条内容只能存一次。

        **这是去重的核心。** 用户反复说「我喜欢简短回答」时，
        抽取器每轮都可能产出一条新记忆；不去重的话表里会积累十几条
        近乎重复的记录，召回时挤占上下文预算。
        """
        assert store.remember("用户偏好简短回答") is True
        assert store.remember("用户偏好简短回答") is False
        assert store.remember("  用户偏好简短回答  ") is False  # 前后空格也算同一条
        assert len(store.list_all()) == 1

    def test_empty_content_rejected(self, store):
        assert store.remember("") is False
        assert store.remember("   ") is False
        assert store.list_all() == []

    def test_recall_returns_matching(self, store):
        store.remember("用户偏好简短回答不带解释", kind="preference")
        store.remember("知识库是编译器课程课件", kind="fact")

        hits = store.recall("偏好简短回答", min_score=0.1)
        assert hits, "应召回与 query 更接近的那条"
        assert "简短回答" in hits[0].content

    def test_recall_respects_threshold(self, store):
        """阈值过高时召不回 —— 这条用来证明阈值确实生效。"""
        store.remember("用户偏好简短回答", kind="preference")
        assert store.recall("完全无关的话题内容", min_score=0.99) == []

    def test_recall_bumps_hit_count(self, store):
        """召回必须累加 hit_count。

        这是「记忆有没有用」的唯一证据：全为 0 说明抽取错了或阈值太高。
        """
        store.remember("用户偏好简短回答", kind="preference")
        store.recall("偏好简短回答", min_score=0.1)
        rows = store.list_all()
        assert rows[0].hit_count >= 1

    def test_forget_removes_row(self, store):
        store.remember("待删除的记忆", kind="fact")
        row_id = store.list_all()[0].id
        assert store.forget(row_id) is True
        assert store.list_all() == []

    def test_forget_unknown_returns_false(self, store):
        assert store.forget(99999) is False

    def test_list_filters_by_kind(self, store):
        store.remember("偏好一", kind="preference")
        store.remember("事实一", kind="fact")
        prefs = store.list_all(kind="preference")
        assert len(prefs) == 1
        assert prefs[0].kind == "preference"

    def test_clear_all(self, store):
        store.remember("记忆一", kind="fact")
        store.remember("记忆二", kind="fact")
        assert store.clear_all() == 2
        assert store.list_all() == []
