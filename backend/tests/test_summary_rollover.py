"""滚动增量摘要测试。SQLite 内存库 + 假模型，不依赖 MySQL / API key / 模型。

为什么这些用例值得存在：旧实现把摘要**生成一次**就永久缓存，窗口=10 时第 11 条
消息触发摘要（只覆盖第 1 条），此后第 2 条到第 N-10 条既不在摘要里也不在最近
窗口里 —— 静默丢失。要堵住这个洞，就必须盯住两件事：
  1. 水位线（summary_upto_message_id）是否只在该推进时推进；
  2. 每轮只把「新滑出窗口」的那批喂给模型，而不是整段重喂。
"""
from datetime import datetime

import pytest

from app.models.conversation import Conversation
from app.models.message import Message
from app.services import conversation_service as svc

WINDOW = 4


class _RecordingModel:
    """假模型：记录 prompt，可被指定失败一次。"""

    prompts: list[str] = []
    fail_next = False

    def invoke(self, prompt):
        type(self).prompts.append(prompt)
        if type(self).fail_next:
            type(self).fail_next = False
            raise RuntimeError("LLM 不可用")

        class _R:
            content = "合并后的摘要"

        return _R()


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    _RecordingModel.prompts = []
    _RecordingModel.fail_next = False
    monkeypatch.setattr(svc, "get_chat_model", lambda: _RecordingModel())
    monkeypatch.setattr(svc.settings, "history_window_size", WINDOW, raising=False)
    monkeypatch.setattr(svc.settings, "context_summary_max_chars", 500, raising=False)


def _seed(session, count):
    conv = Conversation(title="t")
    session.add(conv)
    session.flush()
    # created_at 全部相同：定序必须靠 Message.id，秒级时间戳区分不开
    base = datetime(2026, 9, 23, 10, 0, 0)
    for i in range(count):
        session.add(Message(
            conversation_id=conv.id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"消息{i}",
            created_at=base,
        ))
    session.flush()
    return conv


def _add(session, conv, role, content):
    session.add(Message(conversation_id=conv.id, role=role, content=content))
    session.flush()
    # 必须显式失效该关系的缓存：`conv.messages` 在同一 session 里已加载过，
    # 直接 `session.add(Message(...))`（而不是 conv.messages.append）**不会**
    # 让新消息出现在那个集合里，于是第二次调用看到的是过期快照、滑出窗口的那批
    # 算不出来，表现为「LLM 没被调用」。
    # 真实应用里每个请求都开新 SessionLocal、关系是新鲜加载的，所以这不是生产缺陷；
    # 但测试同 session 复用，就必须手动模拟那一步。
    session.expire(conv, ["messages"])


def test_summary_advances_watermark(sqlite_session):
    conv = _seed(sqlite_session, WINDOW + 2)          # 2 条在窗口外
    svc.get_conversation_with_context(sqlite_session, conv.id)

    reloaded = sqlite_session.get(Conversation, conv.id)
    assert reloaded.summary == "合并后的摘要"
    assert reloaded.summary_upto_message_id == 2


def test_only_newly_slid_out_messages_are_sent(sqlite_session):
    """第二次只喂「新滑出窗口」的那批，不重复喂已覆盖的。"""
    conv = _seed(sqlite_session, WINDOW + 2)
    svc.get_conversation_with_context(sqlite_session, conv.id)
    assert "消息0" in _RecordingModel.prompts[0]

    _add(sqlite_session, conv, "user", "消息6")
    _add(sqlite_session, conv, "assistant", "消息7")
    svc.get_conversation_with_context(sqlite_session, conv.id)

    second = _RecordingModel.prompts[1]
    assert "消息2" in second and "消息3" in second      # 新滑出的
    assert "消息0" not in second# 已覆盖，不再重复喂


def test_failure_does_not_advance_watermark(sqlite_session):
    """失败时不覆盖摘要、不推进水位线 —— 下个请求会重试（旧实现会永久钉死兜底文案）。"""
    conv = _seed(sqlite_session, WINDOW + 2)
    _RecordingModel.fail_next = True

    svc.get_conversation_with_context(sqlite_session, conv.id)
    reloaded = sqlite_session.get(Conversation, conv.id)
    assert reloaded.summary is None
    assert reloaded.summary_upto_message_id is None

    svc.get_conversation_with_context(sqlite_session, conv.id)      # 重试
    reloaded = sqlite_session.get(Conversation, conv.id)
    assert reloaded.summary == "合并后的摘要"
    assert reloaded.summary_upto_message_id == 2


def test_legacy_conversation_without_watermark_self_heals(sqlite_session):
    """存量会话有摘要但无水位线：一次合并全部窗口外消息并落上水位线。"""
    conv = _seed(sqlite_session, WINDOW + 2)
    conv.summary = "旧的摘要"
    sqlite_session.flush()

    svc.get_conversation_with_context(sqlite_session, conv.id)

    reloaded = sqlite_session.get(Conversation, conv.id)
    assert reloaded.summary_upto_message_id == 2
    assert "旧的摘要" in _RecordingModel.prompts[0]


def test_no_llm_call_when_nothing_new_slid_out(sqlite_session):
    """稳态下不再调用 LLM：否则每轮问答都要为「其实没有新东西」付一次调用。"""
    conv = _seed(sqlite_session, WINDOW + 2)
    svc.get_conversation_with_context(sqlite_session, conv.id)
    assert len(_RecordingModel.prompts) == 1

    svc.get_conversation_with_context(sqlite_session, conv.id)

    assert len(_RecordingModel.prompts) == 1


def test_messages_inside_the_window_are_not_summarized(sqlite_session):
    """窗口内的消息留给模型直接看，摘要只覆盖滑出去的那部分。"""
    conv = _seed(sqlite_session, WINDOW + 2)
    svc.get_conversation_with_context(sqlite_session, conv.id)

    result = svc.get_conversation_with_context(sqlite_session, conv.id)

    assert [m["content"] for m in result["messages"]] == [
        "消息2", "消息3", "消息4", "消息5",
    ]


# --- 幂等列检查 ---

def test_ensure_columns_adds_the_column_to_an_existing_table():
    """create_all 不 ALTER 已有表 —— 新增列对老库静默不生效，最难查。"""
    from sqlalchemy import create_engine, inspect, text

    from app.db.base import Base
    from app.db.migrate import ensure_columns
    from app.models.conversation import Conversation

    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as conn:
        # 造一张「老库」：有表，没那列
        conn.execute(text(
            "CREATE TABLE conversations (id INTEGER PRIMARY KEY, title VARCHAR(255), summary TEXT)"
        ))
    assert "summary_upto_message_id" not in {
        c["name"] for c in inspect(engine).get_columns("conversations")
    }

    ensure_columns(engine)

    assert "summary_upto_message_id" in {
        c["name"] for c in inspect(engine).get_columns("conversations")
    }
    # 对已有列是 no-op：连跑两次不报错
    ensure_columns(engine)
    engine.dispose()


def test_ensure_columns_ignores_missing_tables():
    """表还不存在时静默跳过 —— 新库由 create_all 负责。"""
    from sqlalchemy import create_engine

    from app.db.migrate import ensure_columns

    engine = create_engine("sqlite+pysqlite:///:memory:")
    ensure_columns(engine)          # 空库：没有 conversations 表，不该抛
    engine.dispose()


def test_ensure_columns_covers_every_declared_column():
    """REQUIRED_COLUMNS 与 ORM 模型不能漂移：模型加了列而清单没登记，老库照样坏。"""
    from app.db.migrate import REQUIRED_COLUMNS
    from app.models.conversation import Conversation

    model_columns = {c.name for c in Conversation.__table__.columns}
    for _table, column, _ddl in REQUIRED_COLUMNS:
        assert column in model_columns, f"{column} 在 ORM 里已不存在，清单该清理了"
