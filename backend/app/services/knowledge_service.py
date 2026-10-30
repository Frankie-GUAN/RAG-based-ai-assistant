from datetime import datetime
from pathlib import Path
import time
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.document import Document as DocumentModel
from app.rag.loader import load_file, split_documents
from app.rag.vector_store import add_documents
from app.rag.bm25_index import get_bm25_index

# 已删除来源名单的缓存与有效期；删除操作会主动失效它。
_DELETED_CACHE_TTL = 30.0
_deleted_cache: Optional[set[str]] = None
_deleted_cache_at: float = 0.0


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
    """删除文档：对用户立即生效，但**不就地改动向量索引**。

    **为什么不能直接删向量。** 2026-10-14 三轮真实验证：只要执行过一次向量删除，
    磁盘上的 Chroma HNSW 段就会只剩 `index_metadata.pickle`，此后任何新进程打开
    索引一律报 `Error loading hnsw index`（sqlite 里向量行数不变，但检索彻底不可用，
    只能靠 `app.rag.ingest --fresh` 从 uploads 整库重建，约 6 分钟）。
    对照组：只做上传、优雅停止服务后新进程读取正常（906 条可读）。
    隔离环境里用四种组合都没能复现删除本身（原生删除、langchain 包装层、
    20/40/60/903 条各规模、服务持有句柄时并发读），因此无法归因到具体机制，
    也就无法保证「再删一次没事」——一次删除换整库不可读，这个交换不成立。

    现在的语义：

    - **立刻**：DB 行标记 `deleted_at`、界面不再显示、BM25 片段移除、磁盘文件删除、
      检索结果过滤掉该来源（`tools/document_search.py` 读这份名单）；
    - **之后**：向量里的残留由 `python -m app.rag.ingest --fresh` 自然清除 —— 文件已
      从 uploads 删除，重建时不会再入库。

    BM25 与磁盘文件之所以敢直接删：前者是可重建的自包含 pickle（2MB），后者的存在
    本身就是「还要不要它」的答案；只有向量索引是唯一不能就地重建的（BM25 那一份
    正是从 Chroma 派生的）。
    """
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).one_or_none()
    if doc is None:
        return None

    # 入库时 source 用的是**存储文件名**（带 uuid），而不是用户上传时的原始名 ——
    # 两个不同名的上传可能内容相同，同名不同内容的两个文件则无法用原始名区分。
    # file_path 才是与 Chroma/BM25 里的 metadata 对得上的那个键。
    source = Path(doc.file_path).name

    get_bm25_index().remove_source(source)
    Path(doc.file_path).unlink(missing_ok=True)

    doc.deleted_at = datetime.now()
    db.commit()
    invalidate_deleted_sources_cache()
    return doc


def deleted_sources(db: Session | None = None) -> set[str]:
    """已软删除文档的**存储文件名**集合，检索时用来过滤。

    带 TTL 缓存：这是每次检索要付的代价，不该是每个问题一条 SQL。删除操作会主动
    失效缓存，所以「刚删完立刻提问」不会读到旧名单。
    """
    global _deleted_cache, _deleted_cache_at

    now = time.monotonic()
    if _deleted_cache is not None and now - _deleted_cache_at < _DELETED_CACHE_TTL:
        return _deleted_cache

    own_session = db is None
    session = db or SessionLocal()
    try:
        rows = session.execute(
            select(DocumentModel.file_path).where(DocumentModel.deleted_at.is_not(None))
        ).all()
        sources = {Path(row[0]).name for row in rows if row[0]}
    finally:
        if own_session:
            session.close()

    _deleted_cache = sources
    _deleted_cache_at = now
    return sources


def invalidate_deleted_sources_cache() -> None:
    global _deleted_cache, _deleted_cache_at
    _deleted_cache = None
    _deleted_cache_at = 0.0


def _find_by_hash(db: Session, content_hash: str) -> Optional[DocumentModel]:
    return (
        db.query(DocumentModel)
        .filter(DocumentModel.content_hash == content_hash)
        .filter(DocumentModel.deleted_at.is_(None))
        .order_by(DocumentModel.id)
        .first()
    )


def list_documents(db: Session, include_deleted: bool = False) -> List[DocumentModel]:
    # 补 id 作次级键：created_at 是秒精度，同一秒上传的多份文档排序不稳定，
    # 列表顺序会在每次刷新之间抖动
    query = db.query(DocumentModel)
    if not include_deleted:
        query = query.filter(DocumentModel.deleted_at.is_(None))
    return (
        query.order_by(DocumentModel.created_at.desc(), DocumentModel.id.desc()).all()
    )
