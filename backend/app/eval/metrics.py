"""检索质量指标：Recall@k、MRR@k、nDCG@k。

**为什么要有这一层。** 端到端的 LLM-as-Judge（见 services/evaluation_service.py）
只能回答「答案好不好」，回答不了「检索准不准」——而 RAG 系统的效果瓶颈
通常在检索而不在生成。一个忠实度 0.5 的答案，可能是判官不严，也可能是
根本没召回正确片段，两者的改进方向完全不同。

所以这里提供**检索层**的指标，把「召回」与「排序」分开看：

- **Recall@k** —— 相关片段被召回的比例。回答「找没找到」。
- **MRR@k**（Mean Reciprocal Rank）—— 首个相关片段排名倒数的均值。
  回答「排得靠不靠前」。它对「第一个结果就不对」最敏感，
  而这正是用户体感最强的失败模式。
- **nDCG@k**（Normalized Discounted Cumulative Gain）—— 带位置折损的增益，
  兼顾「找到」与「排在前面」。回答「整体排序质量」。

三个指标互补：只看 Recall 会掩盖「都找到了但排在第 10 位」；
只看 MRR 会掩盖「第一个对了但后面全错」。

**相关性标注是二元的**（相关 / 不相关），不做分级打分。
这样标注成本低、可复核，且与「这一段能否支撑答案」这个实际问题对齐。
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int
) -> float:
    """Recall@k：前 k 个结果里命中的相关片段占全部相关片段的比例。

    没有相关片段时返回 0.0（而不是 1.0）——那意味着标注为空，
    这种情况应该在数据校验阶段被拦住，而不是在这里被粉饰成满分。
    """
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hit = len(set(retrieved_ids[:k]) & relevant)
    return hit / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int
) -> float:
    """Reciprocal Rank@k：首个相关片段排名倒数的均值（单个 query 时即该值）。

    找不到时返回 0.0。
    """
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int
) -> float:
    """nDCG@k：归一化的折损累积增益。

    增益取二值（相关=1），折损取 `1/log2(rank+1)`（rank 从 1 起）。
    归一化因子是「理想排序」的 DCG，即把全部相关片段按最有利顺序排在前 k 位。

    之所以不直接用 `dcg/ideal_dcg` 而要先判 ideal_dcg==0：
    标注里没有相关片段时，理想排序的增益也是 0，除法会得到 0/0。
    """
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    # 理想排序：前 k 位放满相关片段（相关片段数不足 k 时就是全部）
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def average(values: Iterable[float]) -> float:
    """算术平均。空序列返回 0.0。"""
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def evaluate_query(
    retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 10
) -> dict[str, float]:
    """单条 query 的三个指标。

    `retrieved_ids` 必须是**已按相关性排序**的结果标识序列（通常是 chunk_id）。
    """
    return {
        f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k),
        f"mrr@{k}": reciprocal_rank(retrieved_ids, relevant_ids, k),
        f"ndcg@{k}": ndcg_at_k(retrieved_ids, relevant_ids, k),
    }
