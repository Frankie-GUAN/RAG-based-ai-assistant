"""语义缓存的测试。

**这个模块最需要测的是「不该命中的时候一定不命中」。**

缓存错命中的代价远高于漏命中：漏了只是多等十几秒；错了是立刻给出
一个看似权威的错误答案，而且用户不知道它是复用来的。
所以下面的用例里，反例比正例更重要。
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document


class _FakeCollection:
    """按预设返回检索结果的假 collection。

    允许每个用例直接指定「查询时的相似度」，这样阈值行为可以被精确断言，
    而不依赖真实 embedding 的浮点波动。
    """

    def __init__(self):
        self.docs: dict[str, tuple[str, dict]] = {}
        self.next_score: float | None = None

    def add_texts(self, texts, metadatas=None, ids=None, **kwargs):
        for i, text in enumerate(texts):
            self.docs[str(ids[i])] = (text, dict((metadatas or [])[i]))

    def get(self, include=None, **kwargs):
        """生产代码用 `get(include=[])` 取全部 id 再删——
        `delete(where={})` 会被 Chroma 拒绝。"""
        return {"ids": list(self.docs.keys())}

    def delete(self, ids=None, where=None, **kwargs):
        for key in list(ids or []):
            self.docs.pop(str(key), None)

    def similarity_search_with_relevance_scores(self, query, k=1, **kwargs):
        if not self.docs:
            return []
        key, (text, meta) = next(iter(self.docs.items()))
        score = self.next_score if self.next_score is not None else 1.0
        return [(Document(page_content=text, metadata=meta), score)][:k]


@pytest.fixture
def cache_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.db.base import Base
    from app.models.semantic_cache import SemanticCacheEntry

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine, tables=[SemanticCacheEntry.__table__])
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def cache(monkeypatch, cache_db):
    from app.cache import semantic

    fake = _FakeCollection()
    backend = semantic.ChromaSemanticCache()
    monkeypatch.setattr(backend, "_collection", lambda: fake)
    monkeypatch.setattr(backend, "_chroma", fake)
    monkeypatch.setattr("app.cache.semantic.SessionLocal", lambda: cache_db)
    monkeypatch.setattr(
        cache_db.__class__, "close", lambda self: None, raising=False
    )
    return backend, fake


class TestThreshold:
    def test_hit_above_threshold(self, cache, monkeypatch):
        """相似度高于阈值时命中。"""
        from app.config import settings

        monkeypatch.setattr(settings, "semantic_cache_threshold", 0.90)
        backend, fake = cache

        backend.put("调用图怎么构建", "调用图有三种构建方式……")
        fake.next_score = 0.95

        hit = backend.get("如何构建调用图")
        assert hit is not None
        assert hit.answer == "调用图有三种构建方式……"
        assert hit.matched_query == "调用图怎么构建"

    def test_miss_below_threshold(self, cache, monkeypatch):
        """**低于阈值必须不命中。**

        这是缓存的安全底线：0.89 的相似度意味着「话题相关但问题不同」，
        复用答案会给用户错的内容。
        """
        from app.config import settings

        monkeypatch.setattr(settings, "semantic_cache_threshold", 0.90)
        backend, fake = cache

        backend.put("活跃变量分析怎么定义", "活跃变量分析的定义是……")
        fake.next_score = 0.89

        assert backend.get("活跃变量分析能用来做什么") is None

    def test_exactly_at_threshold_hits(self, cache, monkeypatch):
        """正好等于阈值算命中（>= 而不是 >）。"""
        from app.config import settings

        monkeypatch.setattr(settings, "semantic_cache_threshold", 0.90)
        backend, fake = cache

        backend.put("q", "a")
        fake.next_score = 0.90
        assert backend.get("q2") is not None


class TestStorage:
    def test_put_then_get_roundtrip(self, cache):
        backend, fake = cache
        backend.put(
            "常量传播分析怎么做",
            "常量传播的关键是构造格……",
            sources=[{"doc_id": "d1", "source": "a.pdf", "chunk": "片段"}],
        )
        fake.next_score = 1.0

        hit = backend.get("常量传播分析怎么做")
        assert hit is not None
        assert "构造格" in hit.answer
        # 引用必须原样带回：命中缓存的答案没有来源标签，可信度会掉一半
        assert hit.sources and hit.sources[0]["doc_id"] == "d1"

    def test_same_query_updates_instead_of_duplicating(self, cache, cache_db):
        """同一个问题再写一次，应更新而不是新增一条。"""
        from sqlalchemy import select

        from app.models.semantic_cache import SemanticCacheEntry

        backend, fake = cache
        backend.put("q", "旧答案")
        backend.put("q", "新答案")

        rows = cache_db.execute(select(SemanticCacheEntry)).scalars().all()
        assert len(rows) == 1, "同一 query 不该产生两条缓存"
        assert rows[0].answer == "新答案"

    def test_empty_input_ignored(self, cache):
        backend, fake = cache
        backend.put("", "答案")
        backend.put("问题", "")
        assert fake.docs == {}

    def test_clear(self, cache, cache_db):
        from sqlalchemy import select

        from app.models.semantic_cache import SemanticCacheEntry

        backend, fake = cache
        backend.put("q1", "a1")
        backend.put("q2", "a2")
        assert len(fake.docs) == 2

        removed = backend.clear()
        assert removed == 2
        assert fake.docs == {}
        assert cache_db.execute(select(SemanticCacheEntry)).scalars().all() == []


class TestTTL:
    def test_expired_entry_is_not_returned(self, cache, cache_db, monkeypatch):
        """过期的缓存必须失效。

        答案会过时：模型升级、文档更新后，旧答案就可能是错的。
        没有 TTL 的语义缓存是个定时炸弹。
        """
        from datetime import datetime, timedelta

        from sqlalchemy import select

        from app.config import settings
        from app.models.semantic_cache import SemanticCacheEntry

        monkeypatch.setattr(settings, "semantic_cache_ttl_hours", 24)
        backend, fake = cache
        backend.put("q", "旧答案")

        # 手动把时间调到 25 小时前
        row = cache_db.execute(select(SemanticCacheEntry)).scalars().first()
        row.updated_at = datetime.now() - timedelta(hours=25)
        cache_db.commit()

        fake.next_score = 1.0
        assert backend.get("q") is None, "过期缓存不该命中"

    def test_fresh_entry_is_returned(self, cache, cache_db, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "semantic_cache_ttl_hours", 24)
        backend, fake = cache
        backend.put("q", "新答案")
        fake.next_score = 1.0
        hit = backend.get("q")
        assert hit is not None
        assert hit.age_hours < 1.0

    def test_ttl_zero_disables_expiry(self, cache, cache_db, monkeypatch):
        from datetime import datetime, timedelta

        from sqlalchemy import select

        from app.config import settings
        from app.models.semantic_cache import SemanticCacheEntry

        monkeypatch.setattr(settings, "semantic_cache_ttl_hours", 0)
        backend, fake = cache
        backend.put("q", "答案")

        row = cache_db.execute(select(SemanticCacheEntry)).scalars().first()
        row.updated_at = datetime.now() - timedelta(days=365)
        cache_db.commit()

        fake.next_score = 1.0
        assert backend.get("q") is not None, "TTL=0 表示永不过期"


class TestOrphanHandling:
    def test_index_without_row_returns_none(self, cache, cache_db):
        """索引里有、库里没有（孤儿向量）→ 不命中。

        这种情况会在「库里删了但索引没删干净」时出现，
        命中它等于返回一个不存在的答案。
        """
        backend, fake = cache
        # 只往索引里塞，不写库
        fake.add_texts(["孤儿"], metadatas=[{"id": 99999}], ids=["99999"])
        fake.next_score = 1.0

        assert backend.get("孤儿") is None
