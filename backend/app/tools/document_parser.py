from app.rag.loader import load_file, split_documents
from app.rag.vector_store import add_documents
from app.rag.bm25_index import BM25Index
from app.tools.registry import tool_registry


def parse_document(file_path: str) -> str:
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        return f"文件不存在: {file_path}"

    documents = load_file(path)
    chunks = split_documents(documents)

    # Add to vector store
    add_documents(chunks)

    # Update BM25 index
    bm25 = BM25Index()
    bm25.load()
    all_docs = list(bm25._documents) if bm25._documents else []
    all_docs.extend(chunks)
    bm25.build(all_docs)
    bm25.save()

    return f"成功解析文档 {path.name}，共生成 {len(chunks)} 个片段"


tool_registry.register("parse_document", parse_document)
