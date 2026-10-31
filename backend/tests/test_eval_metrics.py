"""检索指标的单元测试。

**为什么指标也要测。** 评测器是「用数字下结论」的工具，
而指标一旦算错，整张对照表就是错的——而且这种错误极难发现：
数字照样生成，只是全都偏了。所以指标必须有已知答案的单测。

每个用例都用手算得出的期望值，不依赖实现细节。
"""
from app.eval.metrics import (
    average,
    evaluate_query,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecall:
    def test_all_relevant_found(self):
        # 相关 {a, b}，前 3 个都命中 → 1.0
        assert recall_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0

    def test_half_found(self):
        # 相关 {a, b}，只命中 a → 0.5
        assert recall_at_k(["a", "x", "y"], ["a", "b"], 3) == 0.5

    def test_cutoff_at_k_excludes_beyond(self):
        """k 之外的结果不计入。相关 {a}，a 在第 4 位，k=3 → 0.0"""
        assert recall_at_k(["x", "y", "z", "a"], ["a"], 3) == 0.0
        assert recall_at_k(["x", "y", "z", "a"], ["a"], 4) == 1.0

    def test_no_relevant_returns_zero(self):
        """标注为空时返回 0.0 而不是 1.0——避免把空标注粉饰成满分。"""
        assert recall_at_k(["a", "b"], [], 3) == 0.0

    def test_k_zero(self):
        assert recall_at_k(["a"], ["a"], 0) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], ["a"], 10) == 1.0

    def test_second_position(self):
        # 1/2 = 0.5
        assert reciprocal_rank(["x", "a"], ["a"], 10) == 0.5

    def test_third_position(self):
        # 1/3
        assert reciprocal_rank(["x", "y", "a"], ["a"], 10) == 1 / 3

    def test_not_found(self):
        assert reciprocal_rank(["x", "y"], ["a"], 10) == 0.0

    def test_uses_first_relevant_only(self):
        """多个相关项时只看第一个命中的位置。"""
        assert reciprocal_rank(["x", "a", "b"], ["a", "b"], 10) == 0.5

    def test_beyond_k_is_zero(self):
        assert reciprocal_rank(["x", "y", "z", "a"], ["a"], 3) == 0.0


class TestNdcg:
    def test_perfect_order_is_one(self):
        """相关项全部排在最前 → nDCG = 1.0"""
        assert ndcg_at_k(["a", "b", "x"], ["a", "b"], 3) == 1.0

    def test_relevant_at_second_position(self):
        # DCG = 1/log2(3)，IDCG = 1/log2(2) = 1
        import math
        expected = (1 / math.log2(3)) / 1.0
        assert abs(ndcg_at_k(["x", "a"], ["a"], 2) - expected) < 1e-9

    def test_zero_when_nothing_found(self):
        assert ndcg_at_k(["x", "y"], ["a"], 3) == 0.0

    def test_empty_relevant_is_zero(self):
        assert ndcg_at_k(["a"], [], 3) == 0.0

    def test_partial_discount(self):
        """两个相关项分别在第 1、2 位 → IDCG = 1 + 1/log2(3)"""
        import math
        dcg = 1 / math.log2(2) + 1 / math.log2(3)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert abs(ndcg_at_k(["a", "b"], ["a", "b"], 2) - dcg / idcg) < 1e-9


class TestEvaluateQuery:
    def test_returns_three_metrics(self):
        m = evaluate_query(["a", "b"], ["a"], 10)
        assert set(m.keys()) == {"recall@10", "mrr@10", "ndcg@10"}

    def test_k_is_reflected_in_key(self):
        m = evaluate_query(["a"], ["a"], 4)
        assert "recall@4" in m and "mrr@4" in m and "ndcg@4" in m


class TestAverage:
    def test_basic(self):
        assert average([1.0, 2.0, 3.0]) == 2.0

    def test_empty(self):
        assert average([]) == 0.0
