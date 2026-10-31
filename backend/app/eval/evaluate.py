"""评测 CLI：跑金标集，输出策略对照表。

    python -m app.eval.evaluate                      # 跑全部策略
    python -m app.eval.evaluate --strategy rrf_rerank # 只跑一种
    python -m app.eval.evaluate --show-failures      # 列出未命中的 query
    python -m app.eval.evaluate --by-category        # 分类别指标

**为什么要有这个命令。** 检索质量的判断必须基于**实际检索结果**，
而人的直觉判断在「排序好不好」这件事上非常不可靠——
看前几条结果觉得挺像，就容易忽略真正相关的那段排在第 8 位。
这张表是唯一的依据。

**注意每次跑都会重新加载模型**（BGE-M3 约 26s，精排模型约 26s），
所以一轮全策略评测大约需要 1~2 分钟。
"""
from __future__ import annotations

import argparse
import logging

from app.eval.golden_set import GOLDEN_SET
from app.eval.runner import evaluate_strategy
from app.eval.strategies import STRATEGIES, build_strategy, corpus_stats


def _fmt_table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    """等宽对齐的文本表格。"""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description="检索质量评测（对照金标集）")
    parser.add_argument(
        "--strategy",
        action="append",
        choices=list(STRATEGIES),
        help="要评测的策略，可重复；默认全部",
    )
    parser.add_argument("--k", type=int, default=10, help="截断位置，默认 10")
    parser.add_argument("--show-failures", action="store_true", help="列出未命中的 query")
    parser.add_argument("--by-category", action="store_true", help="输出分类别指标")
    args = parser.parse_args()

    # 这条命令只读，但「打开索引目录」本身就足以损坏它（见 exclusive.py）
    from app.rag.exclusive import ensure_exclusive_access
    ensure_exclusive_access("app.eval.evaluate")

    names = args.strategy or list(STRATEGIES)
    stats = corpus_stats()

    print("=" * 72)
    print(f"检索质量评测  |  金标集 {len(GOLDEN_SET)} 条  |  "
          f"向量 {stats['vector']} · BM25 {stats['bm25']}")
    print("=" * 72)

    reports = []
    for name in names:
        retrieve = build_strategy(name)
        report = evaluate_strategy(name, retrieve, k=args.k)
        reports.append(report)

    rows = []
    for report in reports:
        s = report.summary()
        rows.append((
            report.strategy,
            f"{s.get(f'recall@{args.k}', 0):.3f}",
            f"{s.get(f'mrr@{args.k}', 0):.3f}",
            f"{s.get(f'ndcg@{args.k}', 0):.3f}",
            f"{report.mean_latency_ms():.0f}ms",
        ))
    print()
    print(_fmt_table(
        rows,
        ("策略", f"Recall@{args.k}", f"MRR@{args.k}", f"nDCG@{args.k}", "平均延迟"),
    ))

    if args.by_category:
        print()
        categories = sorted({c.category for c in GOLDEN_SET})
        header = ("策略",) + tuple(categories)
        rows2 = []
        for report in reports:
            by_cat = report.by_category()
            row = [report.strategy]
            for cat in categories:
                m = by_cat.get(cat, {})
                row.append(f"{m.get(f'recall@{args.k}', 0):.2f}")
            rows2.append(tuple(row))
        print("分类别 Recall@%d" % args.k)
        print(_fmt_table(rows2, header))

    if args.show_failures:
        for report in reports:
            failures = [o for o in report.outcomes if o.metrics[f"recall@{args.k}"] < 1.0]
            if not failures:
                continue
            print()
            print(f"--- {report.strategy}：{len(failures)} 条未完全命中 ---")
            for o in failures:
                print(f"  [{o.category}] {o.query}")
                print(f"      命中: {o.hits[:3] or '（无）'}")

    print()
    print("提示：Recall 反映「找没找到」，MRR 反映「排得靠不靠前」，nDCG 综合两者。")


if __name__ == "__main__":
    main()
