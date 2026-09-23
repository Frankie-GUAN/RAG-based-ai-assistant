import threading
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

_embeddings: HuggingFaceEmbeddings | None = None
_embeddings_lock = threading.Lock()

_vectorstore: Chroma | None = None
_vectorstore_lock = threading.Lock()


def get_embeddings() -> HuggingFaceEmbeddings:
    """BGE-M3 的进程内单例。

    不用 functools.lru_cache：cache miss 时它允许多个线程同时进入被包装函数，
    对约 2GB 的模型等于偶发双份常驻内存。双检锁保证只构造一次。
    """
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                model = HuggingFaceEmbeddings(
                    model_name=settings.embed_model_path,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                model.embed_query("ping")  # 预热，确保权重真正就绪
                _embeddings = model
    return _embeddings


def has_persisted_index() -> bool:
    return (settings.chroma_dir / "chroma.sqlite3").exists()


def get_vectorstore(create: bool = False) -> Optional[Chroma]:
    """Chroma 句柄的进程内单例。

    读写必须共用同一句柄 —— 旧实现每次 `Chroma(persist_directory=...)` 都新建客户端，
    导致「上传后立刻检索」是碰运气成立的。
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    if not create and not has_persisted_index():
        return None

    with _vectorstore_lock:
        if _vectorstore is None:
            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            _vectorstore = Chroma(
                persist_directory=str(settings.chroma_dir),
                embedding_function=get_embeddings(),
            )
    return _vectorstore


def add_documents(documents: List[Document]) -> Chroma:
    vectorstore = get_vectorstore(create=True)
    if vectorstore is None:  # create=True 之下不会发生，只为收窄 Optional
        raise RuntimeError("Failed to create the vector store")
    vectorstore.add_documents(documents)
    return vectorstore


def get_retriever(k: int | None = None) -> VectorStoreRetriever:
    """k 默认取 hybrid_top_k（每路召回的候选数），**不是** top_k（最终返回数）。

    此前这里硬编码 settings.top_k=4，导致 hybrid_search 的 vector_top_k 参数
    被无声忽略、settings.hybrid_top_k 成为死配置 —— RRF 实际只融合了 4 路
    向量结果，而非 10+10。
    """
    vectorstore = get_vectorstore()
    if vectorstore is None:
        raise ValueError("No persisted vector store found")
    return vectorstore.as_retriever(search_kwargs={"k": k or settings.hybrid_top_k})
