import json
import logging

from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc, delete, update
from app.agent.llm import get_chat_model
from app.models.conversation import Conversation
from app.models.message import Message
from app.config import settings

logger = logging.getLogger(__name__)


def create_conversation(db: Session, title: str = "新对话") -> Conversation:
    conv = Conversation(title=title)
    db.add(conv)
    db.flush()
    return conv


def save_message(
    db: Session,
    conversation_id: int,
    role: str,
    content: str,
    source_type: str | None = None,
    source_data: str | None = None,
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        source_type=source_type,
        source_data=source_data,
    )
    db.add(msg)
    db.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(updated_at=func.now())
    )
    return msg


def list_conversations(db: Session) -> list[dict]:
    rows = db.execute(
        select(
            Conversation.id,
            Conversation.title,
            Conversation.created_at,
            Conversation.updated_at,
            func.count(Message.id).label("message_count"),
            func.max(Message.id).label("max_msg_id"),
        )
        .outerjoin(Message, Message.conversation_id == Conversation.id)
        .group_by(Conversation.id)
        .order_by(desc(Conversation.updated_at), desc(Conversation.id))
    ).all()

    # Batch-fetch last message previews
    msg_ids = [r.max_msg_id for r in rows if r.max_msg_id]
    preview_map: dict[int, str] = {}
    if msg_ids:
        msgs = db.execute(
            select(Message.id, Message.content).where(Message.id.in_(msg_ids))
        ).all()
        preview_map = {m.id: _truncate(m.content, 100) for m in msgs}

    return [
        {
            "id": r.id,
            "title": r.title,
            "message_count": r.message_count,
            "last_message_preview": preview_map.get(r.max_msg_id, ""),
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]


def get_conversation_with_context(db: Session, conversation_id: int) -> dict:
    conv = db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    ).scalar_one_or_none()
    if not conv:
        raise ValueError(f"Conversation {conversation_id} not found")

    messages = conv.messages  # 按 Message.id 排序，见 Conversation.messages 的 order_by

    # 滚动摘要：把「新滑出窗口」的那批合并进已有摘要
    window = settings.history_window_size
    if len(messages) > window:
        _roll_summary(db, conv, messages, window)
        recent = messages[-window:]
    else:
        recent = messages

    return {
        "conversation": {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
        },
        "summary": conv.summary,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "source_type": m.source_type,
                # 引用溯源：source_data 存的是本轮检索命中的片段（JSON 数组）。
                # 重载历史会话时若不返回它，来源只在本次会话里有，关掉页面就没了。
                "sources": _parse_sources(m.source_data),
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in recent
        ],
    }


def _parse_sources(raw: str | None) -> list[dict]:
    """把 source_data 解析成前端能直接渲染的来源列表。

    解析失败返回空列表而不是抛异常：一条历史消息的来源坏了，不该让整个会话
    打不开。
    """
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def rename_conversation(db: Session, conversation_id: int, title: str) -> Conversation:
    conv = db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    ).scalar_one_or_none()
    if not conv:
        raise ValueError(f"Conversation {conversation_id} not found")
    conv.title = title
    db.flush()
    return conv


def delete_conversation(db: Session, conversation_id: int) -> None:
    db.execute(delete(Message).where(Message.conversation_id == conversation_id))
    db.execute(delete(Conversation).where(Conversation.id == conversation_id))


def _roll_summary(
    db: Session, conv: Conversation, messages: list[Message], window: int
) -> None:
    """把新滑出窗口的消息合并进已有摘要。

    旧实现只在 `summary` 为空时摘要一次，其后永不更新 —— 窗口=10 时第 11 条消息
    触发摘要（只覆盖了第 1 条），此后第 2 条到第 N-10 条**既不在摘要里、也不在
    最近窗口里**，静默从上下文里消失。

    **失败时不推进水位线**，下个请求自然重试。旧实现把兜底文案
    `历史对话 (N 条消息)` 写进 `conv.summary`，一次 LLM 抖动就把这个空壳摘要永久
    钉死在该会话上。
    """
    cutoff = messages[-window].id                    # 窗口起点，之前的都属于「窗口外」
    upto = conv.summary_upto_message_id
    pending = [
        m for m in messages
        if m.id < cutoff and (upto is None or m.id > upto)
    ]
    if not pending:
        return

    merged = _merge_summary(conv.summary, pending)
    if merged is None:
        return                                        # 失败：交给下个请求重试

    conv.summary = merged
    conv.summary_upto_message_id = pending[-1].id
    db.flush()


def _merge_summary(previous: str | None, pending: list[Message]) -> str | None:
    """合并摘要。失败返回 None（调用方据此不推进水位线）。"""
    transcript = "\n".join(
        f"{'用户' if m.role == 'user' else '助手'}: {m.content}" for m in pending
    )
    prior = f"已有摘要：\n{previous}\n\n" if previous else ""
    prompt = (
        f"{prior}以下是新增的对话片段：\n{transcript}\n\n"
        "请输出一份更新后的对话摘要，保留关键事实、决策与用户意图，"
        f"总长度不超过 {settings.context_summary_max_chars} 字。只输出摘要本身。"
    )
    try:
        return get_chat_model().invoke(prompt).content
    except Exception:  # noqa: BLE001 —— 失败不缓存，交由下个请求重试
        logger.warning("摘要合并失败，本次不推进水位线", exc_info=True)
        return None


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
