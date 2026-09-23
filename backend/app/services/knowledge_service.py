from pathlib import Path
from typing import List

from sqlalchemy.orm import Session

from app.models.document import Document as DocumentModel
from app.rag.loader import save_uploaded_file, load_file, split_documents
from app.rag.vector_store import add_documents
from app.rag.bm25_index import get_bm25_index


def process_document(db: Session, filename: str, content: bytes) -> DocumentModel:
    file_path = save_uploaded_file(content, filename)
    documents = load_file(file_path)
    chunks = split_documents(documents)

    add_documents(chunks)

    get_bm25_index().add(chunks)

    doc_record = DocumentModel(
        filename=filename,
        file_type=Path(filename).suffix.lower().lstrip("."),
        chunk_count=len(chunks),
        file_path=str(file_path),
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)
    return doc_record


def list_documents(db: Session) -> List[DocumentModel]:
    return db.query(DocumentModel).order_by(DocumentModel.created_at.desc()).all()
