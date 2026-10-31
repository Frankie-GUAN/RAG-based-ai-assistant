from datetime import datetime
from sqlalchemy import String, Text, DateTime, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), default="新对话")
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 摘要已覆盖到的消息 id 上界（含）。NULL 表示「尚未记录」——
    # 下次遇到超窗会话时会自愈式地合并全部窗口外消息并落上水位线。
    #
    # 为什么必须有这一列：要「把新滑出窗口的那批追加进已有摘要」，就得知道上次
    # 摘要覆盖到第几条。消息是追加写的，上次的窗口位置无法反推。把水位线藏进
    # summary 文本做哨兵会让字段语义变脏，而 summary 本来就直接吐给前端。
    summary_upto_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    # 按自增 id 排序，不按 created_at：created_at 是秒精度，同一次问答的
    # user/assistant 两条消息时间戳完全相同（实测确认），无法定序。
    messages = relationship("Message", back_populates="conversation", order_by="Message.id")
