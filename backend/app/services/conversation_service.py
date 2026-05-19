from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc, delete, update
from app.models.conversation import Conversation
from app.models.message import Message
from app.config import settings
from langchain_openai import ChatOpenAI


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
        .order_by(desc(Conversation.updated_at))
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

    messages = conv.messages  # ordered by created_at via relationship

    # Compression check
    if len(messages) > settings.history_window_size:
        if not conv.summary:
            old = messages[:-settings.history_window_size]
            conv.summary = _generate_summary(old)
            db.flush()

        recent = messages[-settings.history_window_size:]
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
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in recent
        ],
    }


def delete_conversation(db: Session, conversation_id: int) -> None:
    db.execute(delete(Message).where(Message.conversation_id == conversation_id))
    db.execute(delete(Conversation).where(Conversation.id == conversation_id))


def _generate_summary(messages: list[Message]) -> str:
    transcript = "\n".join(
        f"{'用户' if m.role == 'user' else '助手'}: {m.content}" for m in messages
    )
    try:
        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.2,
        )
        prompt = (
            "请用简洁的中文总结以下对话历史，保留关键事实、决策和用户意图，不超过5句话：\n\n"
            f"{transcript}\n\n摘要："
        )
        result = llm.invoke(prompt)
        return result.content
    except Exception:
        # If LLM is unavailable, return a simple fallback summary
        return f"历史对话 ({len(messages)} 条消息)"


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
