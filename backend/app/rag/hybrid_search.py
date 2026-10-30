import logging
from typing import List, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.rag.bm25_index import BM25Index
from app.rag.query_cache import cached_embed_query
from app.rag.vector_store import get_vectorstore

logger = logging.getLogger(__name__)


def _doc_key(doc: Document) -> str:
    """跨路去重用的片段标识。

    **不要用 `page_content[:200]`。** 同一文档相邻分片的开头高度相似
    （「第三章 劳动合同的订立……」），截断后会被判成同一片段而互相覆盖；
    反过来，两份内容相同但来源不同的文档也会被合并，于是检索结果里少一条，
    而 `doc_map[key] = doc` 会让后写入的那条顶掉先写入的那条 ——
    丢的是**先召回的那一路的证据**。

    优先用入库时写进 metadata 的 chunk_id（见 rag/loader.py）；没有它时
    退回到 (source, 页码, 内容摘要) 组合，仍然比纯内容前缀稳。
    """
    metadata = doc.metadata or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"
    source = metadata.get("source", "")
    page = metadata.get("page")
    digest = hash(doc.page_content)  # 内容相同的两片视为同一片段
    return f"meta:{source}:{page}:{digest}"


def _vector_search(query: str, top_k: int) -> List[Tuple[Document, float]]:
    if top_k <= 0:
        # 0 是合法取值（「向量这一路不要候选」），所以在这里短路，而不是构造一个
        # k=0 的 retriever —— chromadb 接受构造，但 invoke 时会抛
        # TypeError: Number of requested results 0, cannot be negative, or zero.
        return []
    vectorstore = get_vectorstore()
    if vectorstore is None:
        raise ValueError("No persisted vector store found")

    # 用带分数的检索，而不是 as_retriever().invoke()：后者只给 Document，
    # 相似度得由调用方自己编 —— 旧实现就编了一个 `1.0 - i * 0.05` 的名次占位分，
    # 然后让它一路流到 search_documents 的输出里展示成「相关度」。模型看到
    # 「相关度 0.03」会以为资料不相关而拒答，这是会主动降低回答质量的 bug。
    #
    # **query 向量走缓存**：BGE-M3 在 CPU 上编码一条 query 要数秒，
    # 而同一个 query 被反复问是常态（评测会把同一批跑四遍、用户会追问）。
    # 缓存命中后这一步从「编码 + 检索」变成「检索」。
    # 详见 app/rag/query_cache.py。
    query_vector = cached_embed_query(query)
    pairs: List[Tuple[Document, float]] = _search_by_vector(
        vectorstore, query_vector, top_k
    )
    # chroma 的 relevance score 理论上是 [0, 1]，但不同版本/距离度量下可能越界或为负
    # （余弦距离转相似度时尤其如此）。夹到 [0, 1] 只为让它可读，不影响排序 ——
    # 排序完全由 similarity_search 返回的顺序决定。
    return [(doc, max(0.0, min(1.0, float(score)))) for doc, score in pairs]


def _search_by_vector(
    vectorstore, query_vector: List[float], top_k: int
) -> List[Tuple[Document, float]]:
    """用预计算的 query 向量做检索，并**保持与原实现一致的分数语义**。

    **为什么不能直接用 `similarity_search_by_vector_with_relevance_scores`。**
    实测它的分数与 `similarity_search_with_relevance_scores` 差得很远
    （同一批结果：0.92 vs 0.35）—— 排序一样，但分数含义完全不同。

    原因：后者用 `_select_relevance_score_fn()` 把距离转成 relevance
    （默认 cosine：`1 - distance`），而 by_vector 版本**直接返回原始距离**。
    分数会一路流到 `search_documents` 的输出里展示成「相关度」，
    也参与 RRF 融合的语义判断——所以必须对齐。

    这里显式做同样的转换，保证**换用缓存前后检索结果与分数完全一致**
    （否则这次优化就成了「悄悄改了检索语义」）。
    """
    raw = vectorstore.similarity_search_by_vector_with_relevance_scores(
        query_vector, k=top_k
    )
    # 默认集合的距离度量是 l2（Chroma 的默认），relevance = 1 - 距离。
    # 这里的转换与 langchain 的 `_select_relevance_score_fn()` 默认行为一致。
    relevance_fn = getattr(vectorstore, "_select_relevance_score_fn", None)
    if callable(relevance_fn):
        fn = relevance_fn()
        return [(doc, fn(score)) for doc, score in raw]

    # 拿不到就按 cosine 兜底：向量已归一化，余弦相似度 = 1 - 距离
    return [(doc, 1.0 - float(score)) for doc, score in raw]


def _rrf_fuse(
    vector_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    k: int = 60,
    top_k: int = 10,
) -> List[Tuple[Document, float]]:
    if top_k <= 0:
        # 与 _vector_search / BM25Index.search 统一：非正数即不要结果。负数尤其不能
        # 靠切片表达 —— [:-1] 会返回「除最后一个之外的全部」，既不空也不报错。
        return []
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(vector_results):
        key = _doc_key(doc)
        doc_map[key] = doc
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)

    for rank, (doc, _) in enumerate(bm25_results):
        key = _doc_key(doc)
        doc_map[key] = doc
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_map[key], score) for key, score in ranked]


def _rerank(
    query: str,
    candidates: List[Tuple[Document, float]],
    top_k: int,
) -> List[Tuple[Document, float]]:
    """用 Cross-Encoder 重排候选顺序；任何失败都退回 RRF 原序。

    **精排是增强，不是依赖。** 模型没下载、内存不够、CPU 上加载超时——任何一种
    都不该让整条检索路径失败。所以这里捕获全部异常并原样返回 RRF 结果：最坏情况
    是「精排没生效」，而不是「搜不出来」。

    **只改顺序，不改分数语义。** CrossEncoder 打的是 logit，量纲随模型而异
    （常见 -10 ~ +10），把它当「相关度」展示给 LLM 会让模型误判资料是否相关 ——
    那正是当年伪造相似度带来的问题。返回的仍是 RRF 融合分。
    """
    from app.rag.reranker import get_reranker

    if not candidates:
        return []

    try:
        reranked = get_reranker().rerank(query, candidates, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 — 见上方docstring：降级而非失败
        logger.warning(
            "精排失败，本次退回 RRF 融合顺序：%s: %s",
            type(exc).__name__,
            exc,
        )
        return candidates[:top_k]

    if not reranked:
        # 精排返回空不该静默变成「没有结果」——那比不精排糟得多。
        return candidates[:top_k]

    # 按文档标识回填 RRF 分，保持分数语义不变
    fusion_scores = {_doc_key(doc): score for doc, score in candidates}
    return [(doc, fusion_scores.get(_doc_key(doc), 0.0)) for doc, _ in reranked]


def hybrid_search(
    query: str,
    bm25_index: BM25Index,
    vector_top_k: int | None = None,
    bm25_top_k: int | None = None,
    final_top_k: int | None = None,
) -> List[Tuple[Document, float]]:
    """两路各召回 hybrid_top_k 个候选，RRF 融合后按需精排，返回 top_k 个。

    精排开着时，融合阶段取`rerank_candidates` 个（默认 10）送进 Cross-Encoder，
    再截到 `top_k`（默认 4）；关掉时融合阶段直接产出 `top_k` 个，少一次重排开销。

    **跨语言改写只作用于 BM25 一路。** 见 `_expand_for_sparse` 的说明。
    """
    # 用 is None 而不是 `or`：0 是合法取值（"这一路不要候选"），不该被当成"没传"
    if vector_top_k is None:
        vector_top_k = settings.hybrid_top_k
    if bm25_top_k is None:
        bm25_top_k = settings.hybrid_top_k
    if final_top_k is None:
        final_top_k = settings.top_k
    if final_top_k <= 0:
        return []

    use_reranker = settings.enable_reranker
    fusion_top_k = max(settings.rerank_candidates, final_top_k) if use_reranker else final_top_k

    vector_results = _vector_search(query, top_k=vector_top_k)
    bm25_results = bm25_index.search(
        _expand_for_sparse(query), top_k=bm25_top_k
    )
    fused = _rrf_fuse(vector_results, bm25_results, top_k=fusion_top_k)

    if not use_reranker:
        return fused[:final_top_k]

    return _rerank(query, fused, top_k=final_top_k)


def _expand_for_sparse(query: str) -> str:
    """给稀疏检索（BM25）追加英文术语候选。

    **为什么只给 BM25 用，向量侧用原 query。**

    实测（见 docs/tech-notes/verified-facts.md §9）：
    - **BM25 靠词面匹配**。中文 query 遇到英文语料，词面完全对不上；
      补上 "Constant Propagation" 之后，BM25 立刻把它排到第一。
    - **向量靠语义匹配**。BGE-M3 的多语言对齐已经能处理中英对应关系，
      给它追加英文术语反而**稀释了中文语义信号**——实测对部分 query
      会让向量侧的 top1 从命中变成不命中。

    两路各用各的最优输入，这正是混合检索该有的样子。

    **改写失败时原样返回** —— 术语映射是增强，不该让检索失败。
    """
    if not settings.enable_query_rewrite:
        return query
    try:
        from app.rag.terminology import rewrite_query
        return rewrite_query(query)
    except Exception as exc:  # noqa: BLE001 — 增强失败不影响主链路
        logger.warning("query 术语改写失败，按原 query 检索：%s: %s", type(exc).__name__, exc)
        return query
