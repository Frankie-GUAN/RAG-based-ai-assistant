"""Query 向量缓存的单元测试。

**为什么这个缓存值得单独测。** 它是「让检索快 60 倍」的关键那一环，
但它是**纯优化**——一旦它悄悄改变了检索结果，就是最坏的情况：
性能提升掩盖了语义变化。所以这里要钉住三件事：

1. 相同 query 得到相同向量（缓存的基本语义）
2. 不同 query 得到不同向量（缓存不能把所有查询都变成同一个）
3. 换嵌入模型后缓存必须能清掉（否则会返回旧模型的向量）
"""
import pytest

from app.rag import query_cache


@pytest.fixture(autouse=True)
def _clean_cache():
    """每个用例前后都清缓存，避免用例间互相污染。"""
    query_cache.clear_query_cache()
    yield
    query_cache.clear_query_cache()


class TestBasicBehavior:
    def test_miss_returns_none(self):
        assert query_cache.get_cached_query_vector("没缓存过的 query") is None

    def test_put_then_get(self):
        query_cache.put_query_vector("q1", [0.1, 0.2, 0.3])
        assert query_cache.get_cached_query_vector("q1") == [0.1, 0.2, 0.3]

    def test_same_query_same_vector(self):
        query_cache.put_query_vector("q1", [0.1, 0.2])
        first = query_cache.get_cached_query_vector("q1")
        second = query_cache.get_cached_query_vector("q1")
        assert first == second

    def test_whitespace_is_normalized(self):
        """前后空格不应产生两个缓存条目 —— 那是同一个 query。"""
        query_cache.put_query_vector("  q1  ", [0.1])
        assert query_cache.get_cached_query_vector("q1") == [0.1]
        assert query_cache.get_cached_query_vector("q1 ") == [0.1]

    def test_clear_empties_cache(self):
        query_cache.put_query_vector("q1", [0.1])
        query_cache.clear_query_cache()
        assert query_cache.get_cached_query_vector("q1") is None


class TestEviction:
    def test_lru_evicts_oldest_when_full(self):
        """超过上限时淘汰最久未用的。"""
        limit = query_cache._MAX_ENTRIES
        for i in range(limit + 10):
            query_cache.put_query_vector(f"q{i}", [float(i)])
        stats = query_cache.cache_stats()
        assert stats["size"] <= limit, f"缓存未按上限淘汰：{stats}"

    def test_recently_used_is_kept(self):
        """刚访问过的条目不该被淘汰（LRU 而不是 FIFO）。

        构造顺序：
        1. 填满缓存（q0 最老）
        2. 写入 mid → 触发淘汰，q0 没了
        3. 访问 mid → 它变成最近使用
        4. 再写入 newcomer → 此时被淘汰的应该是 q1，而不是 mid
        """
        limit = query_cache._MAX_ENTRIES
        for i in range(limit):
            query_cache.put_query_vector(f"q{i}", [float(i)])

        query_cache.put_query_vector("mid", [2.0])       # 触发淘汰，q0 没了
        assert query_cache.get_cached_query_vector("q0") is None, "最老的条目应该被淘汰"

        query_cache.get_cached_query_vector("mid")        # mid 变热
        query_cache.put_query_vector("newcomer", [3.0])  # 再淘汰一次

        assert query_cache.get_cached_query_vector("mid") is not None, "LRU 淘汰了最近用过的条目"
        assert query_cache.get_cached_query_vector("q1") is None, "次老的条目应该被淘汰"

    def test_cache_stats(self):
        query_cache.put_query_vector("q1", [0.1])
        stats = query_cache.cache_stats()
        assert stats["size"] == 1
        assert stats["max_entries"] == query_cache._MAX_ENTRIES


class TestCachedEmbedQuery:
    def test_uses_cache_on_second_call(self, monkeypatch):
        """第二次调用不应该再碰模型。"""
        calls = []

        class _FakeModel:
            def embed_query(self, text):
                calls.append(text)
                return [0.1, 0.2]

        import app.rag.vector_store as vs_mod
        monkeypatch.setattr(vs_mod, "get_embeddings", lambda: _FakeModel())

        first = query_cache.cached_embed_query("q1")
        second = query_cache.cached_embed_query("q1")

        assert first == second
        assert len(calls) == 1, f"模型被调用了 {len(calls)} 次，缓存没生效"

    def test_different_queries_hit_model_separately(self, monkeypatch):
        calls = []

        class _FakeModel:
            def embed_query(self, text):
                calls.append(text)
                return [float(len(text))]

        import app.rag.vector_store as vs_mod
        monkeypatch.setattr(vs_mod, "get_embeddings", lambda: _FakeModel())

        query_cache.cached_embed_query("aaa")
        query_cache.cached_embed_query("bbb")

        assert len(calls) == 2, "不同 query 不该共用缓存"
