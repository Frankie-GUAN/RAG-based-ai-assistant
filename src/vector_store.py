from __future__ import annotations

import os
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from .config import CHROMA_DIR, EMBED_MODEL, GEMINI_API_KEY_ENV


def _require_gemini_key() -> str:
    # 仅从环境变量读取 Gemini 密钥
    api_key = os.getenv(GEMINI_API_KEY_ENV)
    if not api_key:
        raise ValueError("未检测到 GEMINI_API_KEY 环境变量")
    return api_key


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    # 初始化 Gemini Embedding 模型
    api_key = _require_gemini_key()
    model_name = os.getenv("GEMINI_EMBED_MODEL") or EMBED_MODEL
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    try:
        # 预热校验模型可用性
        embeddings.embed_query("ping")
        return embeddings
    except Exception as exc:
        message = (
            "Embedding 模型不可用或当前地区不支持该 API。"
            "请在支持的区域创建密钥，或设置环境变量 GEMINI_EMBED_MODEL "
            "为你账号可用的 embedding 模型名称后重试。"
        )
        raise ValueError(message) from exc


def has_persisted_index() -> bool:
    # 判断是否存在本地持久化索引
    return (CHROMA_DIR / "chroma.sqlite3").exists()


def load_vectorstore(embeddings: GoogleGenerativeAIEmbeddings) -> Optional[Chroma]:
    # 读取已存在的 Chroma 索引
    if not has_persisted_index():
        return None
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)


def add_documents(documents: List[Document], embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    # 将文档写入或追加到本地 Chroma 向量库
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if has_persisted_index():
        vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
        vectorstore.add_documents(documents)
    else:
        vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=str(CHROMA_DIR),
        )
    vectorstore.persist()
    return vectorstore
