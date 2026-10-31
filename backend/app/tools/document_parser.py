from pathlib import Path

from app.rag.bm25_index import get_bm25_index
from app.rag.loader import load_file, split_documents
from app.rag.vector_store import add_documents


def parse_document(file_path: str) -> dict:
    """解析上传文档并建立索引。文件不存在时抛 FileNotFoundError。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    chunks = split_documents(load_file(path))
    add_documents(chunks)
    get_bm25_index().add(chunks)          # Task 2 引入，取代原先的 load/extend/build/save

    return {"file": path.name, "chunks": len(chunks)}
