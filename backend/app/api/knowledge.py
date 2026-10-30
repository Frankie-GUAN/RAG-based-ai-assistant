import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.rag.loader import UploadTooLargeError, save_uploaded_stream
from app.services.knowledge_service import delete_document, ingest_saved_file, list_documents

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """上传文档并入库。

    落盘与入库都在线程池里跑：`save_uploaded_stream` 与解析/嵌入都是同步且重的，
    直接在 async 端点里调用会冻结整个事件循环 —— 上传期间其他请求全部排队。
    """
    loop = asyncio.get_event_loop()
    filename = file.filename or "未命名文件"

    try:
        file_path, content_hash = await loop.run_in_executor(
            None, save_uploaded_stream, file.file, filename
        )
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        doc, duplicate = await loop.run_in_executor(
            None, ingest_saved_file, db, file_path, filename, content_hash
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "id": doc.id,
        "filename": doc.filename,
        "chunk_count": doc.chunk_count,
        # 前端据此提示「这份文件已在库里」，而不是让用户以为白传了一次
        "duplicate": duplicate,
    }


@router.get("/documents")
async def get_documents(db: Session = Depends(get_db)):
    docs = list_documents(db)
    return [
        {"id": d.id, "filename": d.filename, "file_type": d.file_type, "chunk_count": d.chunk_count, "created_at": d.created_at.isoformat()}
        for d in docs
    ]


@router.delete("/documents/{document_id}")
async def delete_document_by_id(document_id: int, db: Session = Depends(get_db)):
    """删除文档：对用户立即生效（界面消失、检索不再命中、磁盘文件清除）。

    向量索引里的片段保留到下一次 `app.rag.ingest --fresh` 重建为止 —— 就地删向量
    会毁掉整个 HNSW 段（实测 chromadb 1.5.9），代价是整库不可读，不值。
    """
    loop = asyncio.get_event_loop()
    doc = await loop.run_in_executor(None, delete_document, db, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"ok": True, "id": document_id, "filename": doc.filename}