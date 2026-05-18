from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.knowledge_service import process_document, list_documents

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    doc = process_document(db, file.filename, content)
    return {"id": doc.id, "filename": doc.filename, "chunk_count": doc.chunk_count}


@router.get("/documents")
async def get_documents(db: Session = Depends(get_db)):
    docs = list_documents(db)
    return [
        {"id": d.id, "filename": d.filename, "file_type": d.file_type, "chunk_count": d.chunk_count, "created_at": d.created_at.isoformat()}
        for d in docs
    ]
