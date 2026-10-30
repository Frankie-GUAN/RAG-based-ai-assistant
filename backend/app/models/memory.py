"""长期记忆表。

**这是「三层记忆」的第二层。** 第一层（短期：会话内最近 N 轮 + 滚动摘要）
已经在 `Conversation.summary` 里做对了；这一层解决的是**跨会话**——
用户上个星期说过「我喜欢简短回答」，这个星期再问，系统应该还记得。

## 为什么记忆要分 kind

不同种类的记忆，召回策略与生命周期完全不同：

| kind | 例子 | 特点 |
|---|---|---|
| `preference` | 「我偏好带代码示例的回答」 | 长期有效，除非被明确推翻 |
| `fact` | 「我们的知识库是编译器课程课件」 | 稳定，但可能过时 |
| `term` | 「这个项目里『切片』指 chunk」 | 领域约定，召回时价值最高 |

分开存是为了将来能做「按 kind 加权召回」——术语类记忆应该比偏好类更容易被召回，
因为它直接影响对问题的理解。

## 为什么有 `content_hash`

它是**去重键**。记忆抽取是反复进行的：用户每次说「我喜欢简短回答」，
抽取器都可能产出一条新记忆。没有去重键的话，表里会积累十几条近乎重复的记录，
召回时挤占上下文预算。

用 hash 而不是直接对 `content` 建唯一索引：`content` 是 Text，
MySQL 对长文本建唯一索引需要前缀长度，容易出现「前缀相同但内容不同」的误判。
`content_hash` 是固定 64 字符，索引干净。

## 为什么有 `hit_count`

**它让「这条记忆有没有用」变成可测量的事实。** 一条从不被召回的记忆，
要么是抽取错了，要么是召回策略有问题。有了计数才能发现这类问题，
而不是凭感觉说「记忆功能正常」。

后续可以做：定期清理 `hit_count = 0` 且超过 30 天的记忆。
"""
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class UserMemory(Base):
    __tablename__ = "user_memories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # 多租户预留：当前没有登录系统，全部写 "default"。
    # 接入 JWT 后只需把写入值换成登录用户 id，表结构不用动。
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, default="default")

    # preference / fact / term，见模块 docstring
    kind: Mapped[str] = mapped_column(String(20), nullable=False, default="fact")

    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 去重键：sha256(content) 的十六进制
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # 抽取时的置信度。低置信度的记忆召回时可以降权或跳过。
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    # 这条记忆是从哪个会话抽出来的。会话被删除时置空（不级联删记忆）——
    # 记忆的价值在于跨会话，源会话消失不代表记忆失效。
    source_conversation_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )

    # 被召回次数，见模块 docstring
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    # 同 semantic_cache：不加 `onupdate=func.now()`。
    # 数据库函数的时区语义在 SQLite（UTC）与 MySQL（本地时间）之间不一致，
    # 而应用层显式写入可以保证跨环境一致。
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        # 同一用户下同一条内容只能有一条记忆
        Index("uq_user_memory_hash", "user_id", "content_hash", unique=True),
        # 按用户 + 类型查询（列表接口、按 kind 加权召回）
        Index("ix_user_memory_kind", "user_id", "kind"),
    )
