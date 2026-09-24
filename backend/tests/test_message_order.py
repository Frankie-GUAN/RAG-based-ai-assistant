"""消息时序回归测试。

真实触发场景是秒精度的 created_at 打平（实测同一次问答的两条消息时间戳完全相同）。
但「时间戳打平」在 SQLite 上恰好会按 rowid 返回，即碰巧等于正确顺序，测不出差异。

所以这里让 created_at 的顺序与插入顺序**相反** —— 这样「按时间排」与「按 id 排」
必然给出不同结果，测试才能确定性地把两者区分开。
"""
from datetime import datetime, timedelta

from app.models.conversation import Conversation
from app.models.message import Message


def _add(session, conversation_id, role, content, created_at):
    session.add(
        Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=created_at,
        )
    )
    session.flush()


def _conversation(session):
    conv = Conversation(title="t")
    session.add(conv)
    session.flush()
    return conv


def test_ordering_is_by_id_not_timestamp(sqlite_session):
    session = sqlite_session
    conv = _conversation(session)

    base = datetime(2026, 9, 23, 10, 0, 0)
    # 插入顺序：user → assistant；时间戳顺序相反
    _add(session, conv.id, "user", "问题", base + timedelta(seconds=5))
    _add(session, conv.id, "assistant", "回答", base)

    session.expire_all()
    reloaded = session.get(Conversation, conv.id)

    assert [m.role for m in reloaded.messages] == ["user", "assistant"]


def test_ordering_survives_identical_timestamps(sqlite_session):
    """回归的实际形态：秒精度下同一轮问答的两条消息时间戳相同，只有 id 能定序。"""
    session = sqlite_session
    conv = _conversation(session)

    same_second = datetime(2026, 9, 23, 10, 0, 0)
    _add(session, conv.id, "user", "问题", same_second)
    _add(session, conv.id, "assistant", "回答", same_second)

    session.expire_all()
    reloaded = session.get(Conversation, conv.id)

    assert [m.role for m in reloaded.messages] == ["user", "assistant"]
