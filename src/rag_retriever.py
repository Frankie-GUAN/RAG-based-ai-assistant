from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from .config import TOP_K


def build_retriever(vectorstore):
    # 构建检索器
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def retrieve_documents(retriever, question: str) -> List[Document]:
    # 兼容新版 LangChain Retriever 的 Runnable 接口
    if hasattr(retriever, "invoke"):
        return retriever.invoke(question)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(question)
    raise AttributeError("Retriever does not support document retrieval")


def format_documents(documents: List[Document]) -> str:
    # 拼接文档内容供模型生成
    return "\n\n".join(doc.page_content for doc in documents)
