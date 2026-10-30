"""长期记忆的存储层：MySQL 存档 + Chroma 索引。

## 为什么是双存储，而不是只用其中一个

**只用 MySQL**：召回要写成 `WHERE content LIKE '%关键词%'`。
用户说「回答短一点」，库里存的是「偏好简短回答」——词面对不上，召回不到。

**只用 Chroma**：能语义召回，但做不到「列出全部记忆」「按 kind 统计」
「记录召回次数」这类结构化操作，也没法跟会话表做外键关联。

所以分工：**MySQL 是唯一真相源（source of truth），Chroma 只是召回索引。**
召回流程是「Chroma 找出候选 id → MySQL 取完整记录」。

## 双写不一致怎么办

写入顺序是**先 MySQL，后 Chroma**。Chroma 写失败只记 warning，不回滚 MySQL。

理由：这条链路跑在后台异步任务里（见 `extractor.py`），失败不该影响用户问答。
最坏情况是「记忆存进了库但召回不到」——功能降级，数据没丢。
反过来「Chroma 有但 MySQL 没有」才是真问题（召回拿到 id 却查不到内容），
所以**不能先写 Chroma**。

删除时两边都删，且**先删 Chroma 再删 MySQL**——同样是「宁可有孤儿记录，
不能让索引指向不存在的行」。
"""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from sqlalchemy import select

from app.config import settings
from app.db.session import SessionLocal

# 通过统一出口导入：确保 conversations 等外键目标表已注册到 mapper，
# 否则写记忆时会抛 NoReferencedTableError（实测踩过）
import app.models  # noqa: F401
from app.models.memory import UserMemory

logger = logging.getLogger(__name__)

# 长期记忆的默认用户。接入 JWT 后替换成登录用户 id，表结构不用动。
DEFAULT_USER_ID = "default"

# Chroma collection 名。与文档库（默认 collection）完全隔离。
MEMORY_COLLECTION = "memories"

_memory_store: "MemoryStore | None" = None
_store_lock = threading.Lock()


def content_hash(content: str) -> str:
    """记忆内容的去重键。"""
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryItem:
    """对外的记忆视图。

    **为什么不直接返回 ORM 对象。** 会话一关，ORM 对象就 detach，
    之后访问任何属性都会尝试 refresh 并抛 `DetachedInstanceError`。
    更麻烦的是它只在**某些**调用路径上出现：`_bump_hits` 里的 commit
    会让对象 expire，于是「召回过的记忆」读属性必崩，而「没召回过的」
    反而正常——这种 bug 极难复现。

    返回不可变 DTO 从根上解决：仓储层负责把 ORM 对象转成值对象，
    调用方永远不持有会话资源。
    """

    id: int
    kind: str
    content: str
    confidence: float
    hit_count: int
    created_at: datetime | None = None


def _to_item(row: UserMemory) -> MemoryItem:
    return MemoryItem(
        id=row.id,
        kind=row.kind,
        content=row.content,
        confidence=float(row.confidence or 0.0),
        hit_count=int(row.hit_count or 0),
        created_at=row.created_at,
    )


class MemoryStore:
    """长期记忆的读写。

    **Chroma 句柄惰性创建**：它要加载 embedding 模型（约 2GB，首次约 26 秒）。
    没启用记忆功能的环境不该为此买单。
    """

    def __init__(self) -> None:
        self._chroma = None
        self._chroma_lock = threading.Lock()

    def _collection(self):
        if self._chroma is not None:
            return self._chroma
        with self._chroma_lock:
            if self._chroma is None:
                from langchain_community.vectorstores import Chroma

                from app.rag.vector_store import get_embeddings

                self._chroma = Chroma(
                    persist_directory=str(settings.chroma_dir),
                    embedding_function=get_embeddings(),
                    collection_name=MEMORY_COLLECTION,
                )
        return self._chroma

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def remember(
        self,
        content: str,
        *,
        kind: str = "fact",
        user_id: str = DEFAULT_USER_ID,
        confidence: float = 1.0,
        conversation_id: int | None = None,
        db=None,
    ) -> bool:
        """存一条记忆。已存在则跳过，返回 False。

        **去重发生在 MySQL 层**（`uq_user_memory_hash` 唯一索引），
        不是先查后插——「先查再插」在并发下会漏。这里用「插入失败就当作重复」
        的方式，靠数据库约束兜底。
        """
        content = content.strip()
        if not content:
            return False

        own_session = db is None
        session = db or SessionLocal()
        memory_id: int | None = None
        try:
            digest = content_hash(content)
            exists = session.execute(
                select(UserMemory.id).where(
                    UserMemory.user_id == user_id,
                    UserMemory.content_hash == digest,
                )
            ).scalar_one_or_none()
            if exists is not None:
                return False

            row = UserMemory(
                user_id=user_id,
                kind=kind,
                content=content,
                content_hash=digest,
                confidence=confidence,
                source_conversation_id=conversation_id,
            )
            session.add(row)
            session.commit()
            memory_id = row.id
        except Exception as exc:  # noqa: BLE001 — 唯一索引冲突等
            session.rollback()
            logger.warning("记忆入库失败：%s: %s", type(exc).__name__, exc)
            return False
        finally:
            if own_session:
                session.close()

        # 索引更新失败不影响「已存进库」这个事实
        try:
            self._collection().add_texts(
                [content],
                # **id 必须写进 metadata。** Chroma 的 `similarity_search_*`
                # 返回的 Document 只带 metadata，**不带文档 id** ——
                # 想在召回时把 Chroma 的命中映射回 MySQL 行，就得自己存一份。
                metadatas=[{"user_id": user_id, "kind": kind, "id": memory_id}],
                ids=[str(memory_id)],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "记忆已入库但索引失败，该条将无法被语义召回：id=%s %s: %s",
                memory_id, type(exc).__name__, exc,
            )

        return True

    # ------------------------------------------------------------------
    # 召回
    # ------------------------------------------------------------------
    def recall(
        self,
        query: str,
        *,
        k: int = 3,
        user_id: str = DEFAULT_USER_ID,
        min_score: float | None = None,
        db=None,
    ) -> list[MemoryItem]:
        """按语义相似度召回记忆。

        **阈值不设死值。** 实测「回答要短一点」对「我偏好简短回答」的
        相似度只有 0.669 —— 如果照抄常见的 0.95 阈值，几乎永远召不回。
        默认阈值由配置控制（`memory_recall_min_score`），可实测校准。
        """
        threshold = (
            settings.memory_recall_min_score if min_score is None else min_score
        )
        try:
            hits = self._collection().similarity_search_with_relevance_scores(
                query, k=k
            )
        except Exception as exc:  # noqa: BLE001 — 召回失败退化为「没有记忆」
            logger.warning("记忆召回失败：%s: %s", type(exc).__name__, exc)
            return []

        wanted: list[int] = []
        for doc, score in hits:
            # 跨用户隔离：索引里存了 user_id，不在召回结果里混用别人的记忆
            if (doc.metadata or {}).get("user_id") != user_id:
                continue
            if score < threshold:
                continue
            try:
                wanted.append(int(doc.metadata.get("id") or 0))
            except (TypeError, ValueError):
                continue

        if not wanted:
            return []

        own_session = db is None
        session = db or SessionLocal()
        try:
            rows = session.execute(
                select(UserMemory).where(UserMemory.id.in_(wanted))
            ).scalars().all()
            order = {mid: i for i, mid in enumerate(wanted)}
            ordered = sorted(rows, key=lambda r: order.get(r.id, 999))
            self._bump_hits(session, [r.id for r in ordered])
            # 在会话还活着时就转成 DTO —— 见 MemoryItem 的说明
            return [_to_item(r) for r in ordered]
        finally:
            if own_session:
                session.close()

    def _bump_hits(self, session, ids: Sequence[int]) -> None:
        """累加召回次数。

        **这条数据是「记忆有没有用」的唯一证据。**
        一条 `hit_count` 长期为 0 的记忆，要么抽取错了，要么阈值太高。
        没有它就只能凭感觉判断记忆功能是否有效。
        """
        if not ids:
            return
        try:
            rows = session.execute(
                select(UserMemory).where(UserMemory.id.in_(ids))
            ).scalars().all()
            for row in rows:
                row.hit_count = (row.hit_count or 0) + 1
            session.commit()
        except Exception as exc:  # noqa: BLE001 — 统计失败不影响召回
            session.rollback()
            logger.warning("记忆命中计数失败：%s: %s", type(exc).__name__, exc)

    # ------------------------------------------------------------------
    # 管理（列表 / 删除）
    # ------------------------------------------------------------------
    def list_all(
        self, *, user_id: str = DEFAULT_USER_ID, kind: str | None = None, db=None
    ) -> list[MemoryItem]:
        own_session = db is None
        session = db or SessionLocal()
        try:
            stmt = select(UserMemory).where(UserMemory.user_id == user_id)
            if kind:
                stmt = stmt.where(UserMemory.kind == kind)
            stmt = stmt.order_by(UserMemory.created_at.desc())
            rows = session.execute(stmt).scalars().all()
            return [_to_item(r) for r in rows]
        finally:
            if own_session:
                session.close()

    def forget(
        self, memory_id: int, *, user_id: str = DEFAULT_USER_ID, db=None
    ) -> bool:
        """删一条记忆。**先删索引再删行**，理由见模块 docstring。"""
        own_session = db is None
        session = db or SessionLocal()
        try:
            row = session.execute(
                select(UserMemory).where(
                    UserMemory.id == memory_id, UserMemory.user_id == user_id
                )
            ).scalar_one_or_none()
            if row is None:
                return False

            try:
                self._collection().delete(ids=[str(memory_id)])
            except Exception as exc:  # noqa: BLE001
                logger.warning("记忆索引删除失败（将留下孤儿向量）：%s: %s",
                               type(exc).__name__, exc)

            session.delete(row)
            session.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            logger.warning("删除记忆失败：%s: %s", type(exc).__name__, exc)
            return False
        finally:
            if own_session:
                session.close()

    def clear_all(self, *, user_id: str = DEFAULT_USER_ID, db=None) -> int:
        """清空某用户的全部记忆。返回删除条数。"""
        own_session = db is None
        session = db or SessionLocal()
        try:
            rows = session.execute(
                select(UserMemory).where(UserMemory.user_id == user_id)
            ).scalars().all()
            ids = [str(r.id) for r in rows]
            if ids:
                try:
                    self._collection().delete(ids=ids)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("记忆索引清空失败：%s: %s", type(exc).__name__, exc)
            for row in rows:
                session.delete(row)
            session.commit()
            return len(rows)
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            logger.warning("清空记忆失败：%s: %s", type(exc).__name__, exc)
            return 0
        finally:
            if own_session:
                session.close()


def get_memory_store() -> MemoryStore:
    """进程内单例。"""
    global _memory_store
    if _memory_store is not None:
        return _memory_store
    with _store_lock:
        if _memory_store is None:
            _memory_store = MemoryStore()
    return _memory_store
