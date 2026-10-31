from app.rag.bm25_index import get_bm25_index
from app.rag.hybrid_search import hybrid_search

# 过滤后仍不够 top_k 时，候选池最多放宽到几倍。
# 上限的作用是防止「语料里绝大多数都被软删除」时无限取候选——
# 那会拖慢每一次检索，而这种情况说明用户该重建索引了。
_MAX_POOL_MULTIPLE = 8


def search_documents(query: str, top_k: int = 4) -> list[dict]:
    """混合检索私有文档库，返回结构化片段列表（相关度降序）。

    返回结构化 dict 而不是拼好的字符串：字符串管道会丢掉 doc_id 与 metadata，
    引用溯源、token 预算、去重全都无从谈起。

    **已软删除的文档会被过滤掉。** 删除只标记数据库、不动向量索引（原因见
    `services/knowledge_service.delete_document`：就地删向量会毁掉整个 HNSW 段），
    因此向量这一路仍可能返回它的片段；BM25 那一份是物理移除的。

    **候选池要渐进放宽，不能一次定死。** 软删除名单可能很长（用户删过很多文档），
    而向量索引里的已删片段也仍然占位。若只按 `top_k * 3` 取一次候选，
    过滤完可能不足 top_k —— 表现为「明明有相关文档却检索不到」，
    而且删除的文档越多越明显。旧实现正是栽在这里：
    固定池宽 9 时，过滤掉 6 条后只剩 2 条，凑不满 top_k=3。

    所以这里逐轮放宽重试：每次把池子翻倍，直到凑满 top_k 或到达上限。
    代价是最多多跑几次检索，换来「结果条数不因删除而缩水」。
    """
    # 局部导入：knowledge_service 依赖 DB 与 rag 层，放在顶层会把导入顺序绑死。
    from app.services.knowledge_service import deleted_sources

    # 必须用共享单例：此处若写 BM25Index()，每次检索都会重建索引并重读 pickle，
    # 把 Task 2 对 BM25 的修复整体作废。
    index = get_bm25_index()
    hidden = deleted_sources()

    if not hidden:
        results = hybrid_search(query, index, final_top_k=top_k)
    else:
        # 初值取 max(top_k*3, top_k+隐藏数)：至少覆盖「被删文档数 + 要保留数」
        # 的最坏情况，再靠下面的翻倍兜底。
        pool_size = max(top_k * 3, top_k + len(hidden))
        limit = top_k * _MAX_POOL_MULTIPLE

        results = []
        while True:
            pool = hybrid_search(query, index, final_top_k=pool_size)
            results = [
                (doc, score) for doc, score in pool
                if (doc.metadata or {}).get("source") not in hidden
            ][:top_k]

            if len(results) >= top_k or pool_size >= limit:
                break
            # 没凑满就把池子翻倍再试。池子已到上限还凑不满，
            # 说明剩下的候选确实都被删了 —— 如实返回已有的，不硬凑。
            pool_size = min(pool_size * 2, limit)

    return [
        {
            # 用 loader 打的稳定 chunk_id，不用「第几个」——序号在下一次入库后会变，
            # 而引用一旦存进会话消息就必须还能指回同一段内容。
            "doc_id": str(doc.metadata.get("chunk_id") or f"{doc.metadata.get('source', 'unknown')}#{i}"),
            "content": doc.page_content,
            "source": doc.metadata.get("source", "未知"),
            "score": round(float(score), 4),
            # 章节信息：让引用能指到「哪一节」而不仅是「哪个文件」。
            # 幻灯片标题由 rag/slide_title.py 从正文首行提取，
            # 覆盖率约 65%（其余页的首行是公式或图形残留，规则有意放过）。
            "page_title": str(doc.metadata.get("page_title") or ""),
            "section_path": list(doc.metadata.get("section_path") or []),
        }
        for i, (doc, score) in enumerate(results)
    ]