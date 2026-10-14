from app.rag.bm25_index import get_bm25_index
from app.rag.hybrid_search import hybrid_search


def search_documents(query: str, top_k: int = 4) -> list[dict]:
    """混合检索私有文档库，返回结构化片段列表（相关度降序）。

    返回结构化dict 而不是拼好的字符串：字符串管道会丢掉 doc_id 与 metadata，
    引用溯源、token预算、去重全都无从谈起。
    """
    # 必须用共享单例：此处若写 BM25Index()，每次检索都会重建索引并重读 pickle，
    # 把 Task 2 对 BM25 的修复整体作废。
    results = hybrid_search(query, get_bm25_index(), final_top_k=top_k)

    return [
        {
            # 用 loader 打的稳定 chunk_id，不用「第几个」——序号在下一次入库后会变，
            # 而引用一旦存进会话消息就必须还能指回同一段内容。
            "doc_id": str(doc.metadata.get("chunk_id") or f"{doc.metadata.get('source', 'unknown')}#{i}"),
            "content": doc.page_content,
            "source": doc.metadata.get("source", "未知"),
            "score": round(float(score), 4),
        }
        for i, (doc, score) in enumerate(results)
    ]