from app.rag.hybrid_search import hybrid_search
from app.rag.bm25_index import BM25Index
from app.tools.registry import tool_registry


def search_documents(query: str, top_k: int = 4) -> str:
    bm25 = BM25Index()
    if bm25.document_count == 0:
        bm25.load()

    results = hybrid_search(query, bm25, final_top_k=top_k)
    if not results:
        return "未找到相关文档"

    lines = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "未知")
        lines.append(f"[{i}] (相关度: {score:.2f}) 来源: {source}\n{doc.page_content[:500]}")
    return "\n\n".join(lines)


tool_registry.register("search_documents", search_documents)
