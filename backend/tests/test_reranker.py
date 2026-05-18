from langchain_core.documents import Document
from app.rag.reranker import Reranker


def test_reranker_returns_top_k():
    reranker = Reranker()
    docs = [
        (Document(page_content="Python编程语言入门教程"), 0.5),
        (Document(page_content="劳动法劳动合同解除"), 0.5),
        (Document(page_content="Django和FastAPI对比"), 0.5),
        (Document(page_content="劳动争议仲裁流程"), 0.5),
        (Document(page_content="机器学习基础"), 0.5),
    ]
    query = "劳动合同如何解除"
    result = reranker.rerank(query, docs, top_k=3)
    assert len(result) == 3
    # Labor-law related documents should rank first
    assert "劳动法" in result[0][0].page_content or "劳动合同" in result[0][0].page_content
