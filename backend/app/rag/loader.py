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
    return splitter.split_documents(documents)
