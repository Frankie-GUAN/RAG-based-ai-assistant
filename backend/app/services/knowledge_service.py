from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.document import Document as DocumentModel
from app.rag.loader import load_file, split_documents
from app.rag.vector_store import add_documents, delete_by_source
from app.rag.bm25_index import get_bm25_index


def ingest_saved_file(
    db: Session,
    file_path: Path,
    filename: str,
    content_hash: str,
) -> tuple[DocumentModel, bool]:
    """把已落盘的文件入库。返回 (文档记录, 是否为重复上传)。

    **去重看内容，不看文件名。** 同一份文件改个名字再传一次，检索结果里就会多出
    一份完全相同的片段 —— Chroma 与 BM25 都不按内容去重。
    """
    existing = _find_by_hash(db, content_hash)
    if existing is not None:
        # 刚落的这份是多余的，删掉：否则 data/uploads 里会堆一份没人引用的副本，
        # 而 `app.rag.ingest --fresh` 会把它当成独立文档再入库一次。
        file_path.unlink(missing_ok=True)
        return existing, True

    documents = load_file(file_path)
    chunks = split_documents(documents)
    if not chunks:
        # 空文件 / 纯扫描版 PDF：不入库，也不留一条 0 片段的记录让用户困惑
        file_path.unlink(missing_ok=True)
        raise ValueError(f"没有从 {filename} 解析出任何内容（可能是扫描版 PDF 或空文件）")

    add_documents(chunks)
    get_bm25_index().add(chunks)

    doc_record = DocumentModel(
        filename=filename,
        file_type=Path(filename).suffix.lower().lstrip("."),
        chunk_count=len(chunks),
        file_path=str(file_path),
        content_hash=content_hash,
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)
    return doc_record, False


def delete_document(db: Session, document_id: int) -> Optional[DocumentModel]:
    """删除文档，并把它的片段从 Chroma、BM25 与磁盘上一并清掉。

    **四处必须一起清。** 只删数据库行是最常见也最糟的做法：向量与 BM25 里的片段
    还在，照样会被检索命中 —— 用户删掉一份文档，模型依然会引用它，唯一的证据是
    一个他再也点不开的来源名。

    删除顺序刻意是「索引 → 磁盘 → 数据库行」：索引是唯一无法从其它地方重建的部分
    （BM25 那一份只能从 Chroma 重建，而 Chroma 一旦删了就没了）。中途失败时留下
    的孤儿索引片段，下次上传同名文件会重新入库覆盖；而反过来（先删行）就彻底
    失去了对账依据。
    """
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).one_or_none()
    if doc is None:
        return None

    # 入库时 source 用的是**存储文件名**（带 uuid），而不是用户上传时的原始名 ——
    # 两个不同名的上传可能内容相同，同名不同内容的两个文件则无法用原始名区分。
    # file_path 才是与 Chroma/BM25 里的 metadata 对得上的那个键。
    source = Path(doc.file_path).name

    delete_by_source(source)
    get_bm25_index().remove_source(source)

    Path(doc.file_path).unlink(missing_ok=True)

    db.delete(doc)
    db.commit()
    return doc


def _find_by_hash(db: Session, content_hash: str) -> Optional[DocumentModel]:
    return (
        db.query(DocumentModel)
        .filter(DocumentModel.content_hash == content_hash)
        .order_by(DocumentModel.id)
        .first()
    )


def list_documents(db: Session) -> List[DocumentModel]:
    # 补 id 作次级键：created_at 是秒精度，同一秒上传的多份文档排序不稳定，
    # 列表顺序会在每次刷新之间抖动
    return (
        db.query(DocumentModel)
        .order_by(DocumentModel.created_at.desc(), DocumentModel.id.desc())
        .all()
    )
