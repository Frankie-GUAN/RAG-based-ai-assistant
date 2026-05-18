import os
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    model_name = settings.embed_model_path
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    embeddings.embed_query("ping")
    return embeddings


def has_persisted_index() -> bool:
    return (settings.chroma_dir / "chroma.sqlite3").exists()


def load_vectorstore() -> Optional[Chroma]:
    if not has_persisted_index():
        return None
    return Chroma(persist_directory=str(settings.chroma_dir), embedding_function=get_embeddings())


def add_documents(documents: List[Document]) -> Chroma:
    embeddings = get_embeddings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    if has_persisted_index():
        vectorstore = Chroma(persist_directory=str(settings.chroma_dir), embedding_function=embeddings)
        vectorstore.add_documents(documents)
    else:
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=str(settings.chroma_dir))
    return vectorstore


def get_retriever():
    vectorstore = load_vectorstore()
    if vectorstore is None:
        raise ValueError("No persisted vector store found")
    return vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
