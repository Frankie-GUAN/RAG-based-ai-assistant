import threading
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

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


def delete_by_source(source: str) -> int:
    """删除某个来源的全部向量，返回删除条数。

    按 metadata 的 `source` 过滤，而不是按 id 列表：`add_documents` 不保留
    「哪些 id 属于哪个文档」的映射（id 由 Chroma 内部生成），而 source 是入库时
    打在每个 chunk metadata 上的字段，是唯一可靠的关联键。

    先取 id 再删（而不是直接 `delete(where=...)`）是为了拿到真实删除条数：
    「删了 0 条」与「删了 12 条」在响应上必须能区分，否则删错文件也表现为成功。
    """
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return 0

    got = vectorstore.get(where={"source": source}, include=[])   # 只要 id
    ids = list(got.get("ids") or [])
    if not ids:
        return 0

    vectorstore.delete(ids=ids)
    return len(ids)
