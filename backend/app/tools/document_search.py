from app.rag.bm25_index import get_bm25_index
from app.rag.hybrid_search import hybrid_search


def search_documents(query: str, top_k: int = 4) -> list[dict]:
    """混合检索私有文档库，返回结构化片段列表（相关度降序）。"""
    # 必须用共享单例：此处若写 BM25Index()，每次检索都会重建索引并重读 pickle，
    # 把 Task 2 对 BM25 的修复整体作废。
    results = hybrid_search(query, get_bm25_index(), final_top_k=top_k)

    return [
        {
            "doc_id": f"{doc.metadata.get('source', 'unknown')}#{i}",
            "content": doc.page_content,
            "source": doc.metadata.get("source", "未知"),
            "score": round(float(score), 4),
        }
        for i, (doc, score) in enumerate(results)
    ]
