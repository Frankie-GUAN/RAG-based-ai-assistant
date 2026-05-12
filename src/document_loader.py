from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
from uuid import uuid4

try:
    # 兼容 LangChain 新旧版本的文本切分器位置
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from .config import CHUNK_OVERLAP, CHUNK_SIZE, UPLOAD_DIR


def ensure_upload_dir() -> None:
    # 确保上传目录存在
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_files(uploaded_files: Iterable) -> List[Path]:
    # 保存上传文件到本地，返回可读路径列表
    ensure_upload_dir()
    saved_paths: List[Path] = []
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in {".pdf", ".txt"}:
            continue
        safe_name = f"{Path(uploaded_file.name).stem}_{uuid4().hex}{suffix}"
        file_path = UPLOAD_DIR / safe_name
        with file_path.open("wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths


def load_documents(file_paths: Iterable[Path]) -> List[Document]:
    # 根据文件类型加载为 LangChain Document
    documents: List[Document] = []
    for file_path in file_paths:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix == ".txt":
            loader = TextLoader(str(file_path), autodetect_encoding=True)
        else:
            continue
        documents.extend(loader.load())
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    # 使用递归字符切分器进行切片
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)
