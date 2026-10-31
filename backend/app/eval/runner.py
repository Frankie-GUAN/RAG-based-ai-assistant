"""检索评测器：跑金标集，对比不同检索策略。

**它能回答的问题。** 「我的混合检索到底比单一向量检索好多少？」
——这是任何 RAG 项目在面试时最该拿数字回答的问题，也是绝大多数个人项目
只能含糊其辞的地方。

**支持的策略。**

| 策略 | 说明 |
|---|---|
| `vector` | 仅 BGE-M3 向量召回 |
| `bm25` | 仅 BM25 稀疏召回（jieba 分词） |
| `rrf` | 两路 RRF 融合，**不精排** |
| `rrf_rerank` | RRF 融合后 Cross-Encoder 精排（生产默认路径） |

**默认跑 `rrf_rerank` 还是 `rrf`？** 都跑。差别本身就是结论：
如果两者指标一致，说明精排没带来收益（要么模型没就位，要么候选太同质）；
如果 `rrf_rerank` 在 MRR 上明显更好，那就是「精排有效」的直接证据。

**命中判定。** 按 (文档前缀, 页码) 判定，而非精确 chunk_id。
理由见 golden_set.py 的说明：切分参数一变，精确 chunk_id 就全失效，
标注会失去跨版本可比性——而可比性是评测集的第一属性。

**为什么不用 LLM 判官做检索评测。** 判官有随机性、有位置偏好，
而检索指标是确定性的。同一个问题问两次，判官可能给不同分；
但 Recall@10 每次跑都一样。检索层用确定性指标，生成层才用判官。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from langchain_core.documents import Document

from app.eval.metrics import average, evaluate_query
from app.eval.golden_set import GOLDEN_SET, GoldenCase

logger = logging.getLogger(__name__)

# 策略名 → 检索函数
Strategy = Callable[[str, int], list[tuple[Document, float]]]


@dataclass
class QueryOutcome:
    """单条 query 的评测结果。"""

    query: str
    category: str
    metrics: dict[str, float]
    retrieved: list[str] = field(default_factory=list)
    latency_ms: int = 0
    hits: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)


@dataclass
class StrategyReport:
    """一种策略跑完整个金标集的结果。"""

    strategy: str
    outcomes: list[QueryOutcome] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        """汇总指标。空结果集返回全 0。"""
        if not self.outcomes:
            return {"recall@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0}
        keys = self.outcomes[0].metrics.keys()
        return {k: average(o.metrics[k] for o in self.outcomes) for k in keys}

    def mean_latency_ms(self) -> float:
        if not self.outcomes:
            return 0.0
        return average(o.latency_ms for o in self.outcomes)

    def by_category(self) -> dict[str, dict[str, float]]:
        """分类别指标。用于定位「哪类问题检索不行」。"""
        groups: dict[str, list[QueryOutcome]] = {}
        for o in self.outcomes:
            groups.setdefault(o.category, []).append(o)
        return {
            cat: {k: average(o.metrics[k] for o in outs) for k in outs[0].metrics}
            for cat, outs in groups.items()
        }


def _matches(case: GoldenCase, doc: Document) -> bool:
    """该文档是否算这条 query 的相关命中。

    **页码口径**：`metadata["page"]` 是 0 起始的（PyPDF 的页序号），
    而 `metadata["page_label"]` 是文档里显示的页码（多数 PDF 从 1 起）。
    标注是按人看到的页码写的，所以**优先用 `page_label`**；
    它缺失时才回退到 `page`（此时语义是「0 起始页序号」，与标注差 1）。
    """
    metadata = doc.metadata or {}
    source = str(metadata.get("source") or "")

    # 文档名用前缀匹配：source 带 uuid 后缀
    if not any(source.startswith(prefix) for prefix in case.relevant_sources):
        return False

    label = metadata.get("page_label")
    if label is not None:
        try:
            return int(label) in case.relevant_pages
        except (TypeError, ValueError):
            return False

    page = metadata.get("page")
    if page is None:
        return False
    try:
        return int(page) in case.relevant_pages
    except (TypeError, ValueError):
        return False


def _hits_and_misses(
    case: GoldenCase, retrieved: Sequence[Document]
) -> tuple[list[str], list[str]]:
    """把检索结果拆成命中与未命中（用于人工复核）。"""
    hits: list[str] = []
    misses: list[str] = []
    for doc in retrieved:
        metadata = doc.metadata or {}
        label = f"{metadata.get('source', '?')}#p{metadata.get('page', '?')}"
        (hits if _matches(case, doc) else misses).append(label)
    return hits, misses


def evaluate_strategy(
    strategy_name: str,
    retrieve: Strategy,
    cases: Sequence[GoldenCase] = GOLDEN_SET,
    k: int = 10,
) -> StrategyReport:
    """跑一种策略，返回报告。

    `retrieve(query, k)` 返回按相关性降序的 (Document, score) 列表。
    """
    report = StrategyReport(strategy=strategy_name)

    for case in cases:
        started = time.perf_counter()
        try:
            results = retrieve(case.query, k)
        except Exception as exc:  # noqa: BLE001 — 单条失败不该中断整批评测
            logger.warning("检索失败 query=%r: %s: %s", case.query, type(exc).__name__, exc)
            report.outcomes.append(
                QueryOutcome(
                    query=case.query,
                    category=case.category,
                    metrics=evaluate_query([], [], k),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            )
            continue

        latency_ms = int((time.perf_counter() - started) * 1000)
        docs = [doc for doc, _ in results]

        # 把「命中的结果」当作相关项，指标因此反映**真实排序**下的召回与排名。
        # 做法：给每个结果一个序号，相关项 = 命中标注的那些序号，
        # 检索列表 = 实际顺序的序号列表。
        # 于是 MRR 量的是「第一个命中实际排在第几位」，
        # nDCG 的折损也作用在真实位置上——精排有没有用，这三个指标能直接分辨。
        relevant_ids = [str(i) for i, doc in enumerate(docs) if _matches(case, doc)]
        retrieved_ids = [str(i) for i in range(len(docs))]

        hits, misses = _hits_and_misses(case, docs)
        report.outcomes.append(
            QueryOutcome(
                query=case.query,
                category=case.category,
                metrics=evaluate_query(retrieved_ids, relevant_ids, k),
                retrieved=[f"{d.metadata.get('source', '?')}#p{d.metadata.get('page', '?')}"
                           for d in docs],
                latency_ms=latency_ms,
                hits=hits,
                misses=misses,
            )
        )

    return report
