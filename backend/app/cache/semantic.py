"""三层记忆的第三层：语义缓存。

## 它解决什么

同一个问题被反复问是常态：用户追问、同事问同一件事、演示时连问几遍同一个。
每一次都跑完「术语改写 → 向量+BM25 混合检索 → 精排 → LLM 生成」，
成本是几千 token + 十几秒——而这些答案**上次已经算过并引用了同一批文档**。

命中缓存时：**跳过检索与生成，直接返回上次的答案**，延迟从十几秒降到亚秒级。

## 阈值是实测校准的，不是抄的

方案文档原本写 0.95。实测（BGE-M3，见 `verified-facts.md` §11.3）：

| 样本类型 | 相似度范围 |
|---|---|
| 同一问题的不同问法 | 0.884 ~ 0.994（均值 0.959） |
| 明显不同的问题 | 0.239 ~ 0.614 |
| **相关但答案不同**（难样本） | 0.551 ~ **0.899** |

0.95 会漏掉近一半同义改写（均值才 0.959，最低 0.884）。
0.85 能全中正样本，但会把「活跃变量分析**怎么定义**」和
「活跃变量分析**能用来做什么**」（相似度 0.899）判成同一个问题——**这是给错答案**。

**选 0.90 的理由**：缓存错命中的代价远高于漏命中。
漏了只是多等十几秒；错了是立刻给出一个看似权威的错误答案，
而且用户不知道它是缓存来的。

## 三道保险，缺一不可

1. **阈值 0.90**（宁可漏，不可错）
2. **命中时在事件里显式标注**「命中语义缓存」，让用户知道这是复用答案
3. **TTL**：答案会过时（模型升级、文档更新），过期缓存必须失效

## 可替换后端

Redis 是语义缓存的常见选择，但：

- 普通 Redis 只能存精确 key，做语义匹配要自己线性扫描所有 embedding
- 要做向量检索得上 Redis Stack / RediSearch，不是 `redis:7` 自带的
- 而 Chroma 天生有 HNSW 向量索引，正是干这个的

所以默认实现是 Chroma。`SemanticCacheBackend` 是协议，
将来有 Redis Stack 时加一个实现类即可，业务代码不用改。
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import select

from app.config import settings
from app.db.session import SessionLocal

import app.models  # noqa: F401  —— 同 store.py：保证 mapper 注册完整
from app.models.semantic_cache import SemanticCacheEntry

logger = logging.getLogger(__name__)

CACHE_COLLECTION = "qa_cache"


@dataclass(frozen=True)
class CachedAnswer:
    """命中缓存时返回的内容。

    `matched_query` 是**缓存里存的那个问题**，不是用户这次问的。
    带上它是为了让前端能显示「复用的是『XX』的答案」——
    用户看到措辞略有不同的问题被复用，可以自己判断这个答案靠不靠谱。
    """

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    matched_query: str = ""
    score: float = 0.0
    age_hours: float = 0.0
    """缓存写入距今多少小时。让「这条缓存有多旧」成为可见信息——
    一条 7 天前的答案和一条 10 分钟前的答案，可信度是不同的。"""


class SemanticCacheBackend(Protocol):
    """语义缓存后端协议。

    换存储（Chroma → Redis Stack → pgvector）只需要实现这三个方法，
    业务代码一行不用改。
    """

    def get(self, query: str) -> CachedAnswer | None: ...

    def put(
        self, query: str, answer: str, sources: list[dict[str, Any]] | None = None
    ) -> None: ...

    def clear(self) -> int: ...


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.strip().encode("utf-8")).hexdigest()


class ChromaSemanticCache:
    """Chroma 索引 + MySQL 存档的语义缓存实现。"""

    def __init__(self) -> None:
        self._chroma = None
        self._lock = threading.Lock()

    def _collection(self):
        if self._chroma is not None:
            return self._chroma
        with self._lock:
            if self._chroma is None:
                from langchain_community.vectorstores import Chroma

                from app.rag.vector_store import get_embeddings

                self._chroma = Chroma(
                    persist_directory=str(settings.chroma_dir),
                    embedding_function=get_embeddings(),
                    collection_name=CACHE_COLLECTION,
                    # **必须显式配 cosine。** 默认是 l2，
                    # 而 relevance = 1 - l2距离 与余弦不是同一量纲：
                    # 实测余弦 0.98 会显示成 relevance 0.88，
                    # 于是基于余弦校准的 0.90 阈值全部失效。
                    collection_metadata={"hnsw:space": "cosine"},
                )
        return self._chroma

    # ------------------------------------------------------------------
    def get(self, query: str) -> CachedAnswer | None:
        try:
            hits = self._collection().similarity_search_with_relevance_scores(
                query, k=1
            )
        except Exception as exc:  # noqa: BLE001 — 缓存失效退化为「没命中」
            logger.warning("语义缓存查询失败：%s: %s", type(exc).__name__, exc)
            return None

        if not hits:
            return None

        doc, score = hits[0]
        if score < settings.semantic_cache_threshold:
            return None

        entry_id = (doc.metadata or {}).get("id")
        if entry_id is None:
            return None

        session = SessionLocal()
        try:
            row = session.get(SemanticCacheEntry, int(entry_id))
            if row is None:
                # 索引有、库里没有：孤儿向量。清掉它避免反复命中。
                self._drop_orphan(entry_id)
                return None

            if self._is_expired(row):
                self._delete_row(row)
                self._drop_orphan(row.id)
                return None

            sources = self._parse_sources(row.sources_json)
            try:
                row.hit_count = (row.hit_count or 0) + 1
                session.commit()
            except Exception:  # noqa: BLE001 — 计数失败不影响返回
                session.rollback()

            base = row.updated_at or row.created_at
            age_hours = (
                (datetime.now() - base).total_seconds() / 3600.0 if base else 0.0
            )

            return CachedAnswer(
                answer=row.answer,
                sources=sources,
                matched_query=row.query,
                score=float(score),
                age_hours=round(age_hours, 2),
            )
        finally:
            session.close()

    # ------------------------------------------------------------------
    def put(
        self, query: str, answer: str, sources: list[dict[str, Any]] | None = None
    ) -> None:
        query = (query or "").strip()
        answer = (answer or "").strip()
        if not query or not answer:
            return

        digest = _query_hash(query)
        sources_json = json.dumps(sources or [], ensure_ascii=False, default=str)

        session = SessionLocal()
        entry_id: int | None = None
        try:
            row = session.execute(
                select(SemanticCacheEntry).where(
                    SemanticCacheEntry.query_hash == digest
                )
            ).scalar_one_or_none()

            if row is not None:
                # 同一个问题再问一次：更新答案与时间戳，不新增条目
                row.answer = answer
                row.sources_json = sources_json
                row.updated_at = datetime.now()
                session.commit()
                entry_id = row.id
            else:
                now = datetime.now()
                row = SemanticCacheEntry(
                    query=query,
                    query_hash=digest,
                    answer=answer,
                    sources_json=sources_json,
                    # **时间戳在应用层显式写，不依赖 server_default。**
                    # 实测：`func.now()` 在 SQLite 下返回 UTC、在 MySQL 下返回
                    # 服务器本地时间——同一份代码在两个环境里 TTL 判断会差 8 小时
                    # （UTC+8）。显式写入让两边语义一致。
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
                session.commit()
                entry_id = row.id
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            logger.warning("写语义缓存失败：%s: %s", type(exc).__name__, exc)
            return
        finally:
            session.close()

        # 索引更新失败不影响「已存进库」
        try:
            collection = self._collection()
            # 先删旧向量再写新的：同一 entry 可能对应旧问题了
            collection.delete(ids=[str(entry_id)])
            collection.add_texts(
                [query],
                metadatas=[{"id": entry_id}],
                ids=[str(entry_id)],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "语义缓存已入库但索引失败，该条无法被命中：id=%s %s: %s",
                entry_id, type(exc).__name__, exc,
            )

    # ------------------------------------------------------------------
    def clear(self) -> int:
        """清空缓存。文档增删后应当调用——旧答案可能引用了已经不存在的文档。"""
        session = SessionLocal()
        try:
            rows = session.execute(select(SemanticCacheEntry)).scalars().all()
            count = len(rows)
            for row in rows:
                session.delete(row)
            session.commit()
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            logger.warning("清空语义缓存失败：%s: %s", type(exc).__name__, exc)
            return 0
        finally:
            session.close()

        try:
            # **不能用 `delete(where={})`**：Chroma 会抛
            # `ValueError: Expected where to have exactly one operator, got {}`。
            # 必须先取出全部 id 再按 id 删。
            collection = self._collection()
            got = collection.get(include=[])
            ids = list(got.get("ids") or [])
            if ids:
                collection.delete(ids=ids)
        except Exception as exc:  # noqa: BLE001
            # 索引没清干净会留下孤儿向量，但 get() 有孤儿检测会兜住
            logger.warning("清空语义缓存索引失败：%s: %s", type(exc).__name__, exc)

        return count

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _is_expired(self, row: SemanticCacheEntry) -> bool:
        ttl_hours = settings.semantic_cache_ttl_hours
        if ttl_hours <= 0:
            return False
        base = row.updated_at or row.created_at
        if base is None:
            return True
        return datetime.now() - base > timedelta(hours=ttl_hours)

    def _delete_row(self, row: SemanticCacheEntry) -> None:
        session = SessionLocal()
        try:
            session.delete(row)
            session.commit()
        except Exception:  # noqa: BLE001
            session.rollback()
        finally:
            session.close()

    def _drop_orphan(self, entry_id: Any) -> None:
        try:
            self._collection().delete(ids=[str(entry_id)])
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _parse_sources(raw: str | None) -> list[dict[str, Any]]:
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return []
        return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


_cache_backend: SemanticCacheBackend | None = None
_backend_lock = threading.Lock()


def get_semantic_cache() -> SemanticCacheBackend:
    """进程内单例。换后端只改这里。"""
    global _cache_backend
    if _cache_backend is not None:
        return _cache_backend
    with _backend_lock:
        if _cache_backend is None:
            _cache_backend = ChromaSemanticCache()
    return _cache_backend


def reset_semantic_cache() -> None:
    """清掉后端实例（测试用）。"""
    global _cache_backend
    with _backend_lock:
        _cache_backend = None
