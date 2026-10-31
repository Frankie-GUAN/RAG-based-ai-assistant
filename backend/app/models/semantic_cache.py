"""语义缓存表：存「问题 → 答案」。

## 为什么答案要单独建表，而不是塞进 Chroma 的 metadata

实测（见 `verified-facts.md` §11.2）踩过一个坑：本想用
`add_texts(embeddings=[query 向量])` 把**答案**当 document 存、
向量用手算的 query 向量——结果 **Chroma 静默忽略了 `embeddings` 参数**，
实际存的是 `embed(答案文本)`。

后果是致命的：之后每次检索都变成「新 query vs 答案文本」的相似度，
所有分数都挤在 0.58 附近，缓存要么永不命中、要么全部错命中。

正确的做法只有一条：**document 必须是 query 文本**（让 Chroma 自己编码），
答案另存。而 Chroma 的 metadata 只适合短结构化字段，
几千字的答案塞进去既难用也不安全，所以单独建这张表。

## 为什么有 `query_hash`

同一个问题被问两次时，不该新增两条缓存，而应**更新答案**。
用 sha256(query) 做唯一键，写入时先查后 upsert。

## 为什么有 `hit_count` 与 `updated_at`

- `hit_count`：缓存到底有没有用，看这个。全为 0 说明阈值太高或缓存没接上。
- `updated_at`：TTL 判断依据。答案会过时（模型升级、文档更新），
  过期缓存必须失效，否则用户会拿到旧的错误答案。
"""
from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SemanticCacheEntry(Base):
    __tablename__ = "semantic_cache_entries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # 原始问题。存它是为了：① 它能被 Chroma 编码成向量 ② 命中时能展示
    # 「这次复用的是哪个问题」，便于用户判断缓存是否正确。
    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    answer: Mapped[str] = mapped_column(Text, nullable=False)

    # 引用来源（JSON 数组），命中缓存时必须原样带回去，
    # 否则「命中缓存」的答案就没有来源标签，可信度直接掉一半。
    sources_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    # **刻意不加 `onupdate=func.now()`。**
    #
    # 加了的话，SQLAlchemy 会在每次 UPDATE 时用数据库函数覆盖应用层写的值，
    # 而 `func.now()` 在两个环境语义不同：SQLite 返回 UTC、MySQL 返回本地时间。
    # 实测表现为「刚写入的缓存 age_hours = 8.0」（差一个 UTC+8）。
    # TTL 判断依赖这个字段，差 8 小时足以让「24 小时过期」变成「16 小时」或「32 小时」。
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("uq_semantic_cache_hash", "query_hash", unique=True),
        # TTL 清理按 updated_at 扫描
        Index("ix_semantic_cache_updated", "updated_at"),
    )
