"""命令行检索：不开服务、不开浏览器，直接看检索结果。

    python -m app.rag.search "如何解除劳动合同"
    python -m app.rag.search --no-rerank "指针分析的作用域" --top-k 8

为什么需要它：检索质量的问题（分词、切分、精排、召回）几乎都要靠「实际搜一把」
来判断，而最快的方式不是起前端。这条命令会加载 BGE-M3（首次约十几秒），
打印每个候选的分数、来源与 chunk_id。
"""
from __future__ import annotations

import argparse
import logging

from app.config import settings
from app.rag.bm25_index import get_bm25_index
from app.rag.hybrid_search import hybrid_search
from app.rag.vector_store import get_vectorstore


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description="命令行检索（用于人工核对检索质量）")
    parser.add_argument("query", help="查询语句")
    parser.add_argument("--top-k", type=int, default=None, help="返回条数，默认取配置 top_k")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="跳过精排，直接看 RRF 融合顺序（用于对比精排是否真的有用）",
    )
    args = parser.parse_args()

    if args.no_rerank:
        settings.enable_reranker = False

    vectorstore = get_vectorstore()
    count = vectorstore._collection.count() if vectorstore else 0
    index = get_bm25_index()
    print(f"向量 {count} · BM25 {index.document_count} · 精排 {'开' if settings.enable_reranker else '关'}")
    print(f"查询：{args.query}\n" + "-" * 68)

    results = hybrid_search(
        args.query,
        index,
        final_top_k=args.top_k if args.top_k is not None else settings.top_k,
    )
    if not results:
        print("（无结果。注意：BM25 在语料小于 2 篇时按数学性质必然全空，见 bm25_index.search）")
        return

    for i, (doc, score) in enumerate(results, 1):
        metadata = doc.metadata or {}
        print(f"[{i}] score={score:.4f}  {metadata.get('source', '?')}")
        print(f"    chunk_id={metadata.get('chunk_id', '（无 —— 该索引未经 loader 建立）')}")
        print(f"    {doc.page_content[:100].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()