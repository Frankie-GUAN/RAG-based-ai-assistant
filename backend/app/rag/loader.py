from pathlib import Path
from typing import List
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from app.config import settings


def save_uploaded_file(uploaded_bytes: bytes, filename: str) -> Path:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    safe_name = f"{Path(filename).stem}_{uuid4().hex}{suffix}"
    file_path = settings.upload_dir / safe_name
    file_path.write_bytes(uploaded_bytes)
    return file_path


def load_file(file_path: Path) -> List[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".txt":
        loader = TextLoader(str(file_path), autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    return _stamp_chunk_ids(chunks)


def _stamp_chunk_ids(chunks: List[Document]) -> List[Document]:
    """给每个片段打上稳定的 `chunk_id`。

    这个 id 有两处消费者：

    1. `hybrid_search._doc_key` —— 跨路（向量 / BM25）去重，此前靠
       `page_content[:200]`，会把同一文档里开头相似的相邻分片判成同一片段。
    2. 引用溯源 —— 前端要能指回「答案里的这句话来自哪个文件的哪一段」。

    刻意用 `来源 + 页码 + 序号` 而不是随机 uuid：重新入库同一份文件应当得到
    同一套 id，否则每次恢复索引都会让历史引用全部失效。
    """
    counters: dict[tuple[str, object], int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", "")
        key = (source, page)
        index = counters.get(key, 0)
        counters[key] = index + 1
        chunk.metadata["chunk_id"] = f"{source}#p{page}#{index}"
    return chunks
