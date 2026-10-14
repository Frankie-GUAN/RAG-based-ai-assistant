import hashlib
from pathlib import Path
from typing import BinaryIO, List, Tuple
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from app.config import settings

# 流式读写的块大小。1MB 是经验值：再大对磁盘吞吐没明显帮助，再小会增加 Python 层循环开销。
_READ_BLOCK = 1024 * 1024


class UploadTooLargeError(ValueError):
    """上传文件超过 `settings.max_upload_mb`。"""


def save_uploaded_stream(stream: BinaryIO, filename: str) -> Tuple[Path, str]:
    """流式落盘，同时算 sha256，返回 (路径, 内容摘要)。

    **不要改回 `await file.read()`。** 那会把整个文件读进内存：一个大 PDF 就能把
    进程撑爆，而正确行为只是拒绝它（413）。流式写还顺带在**写入过程中**就知道
    文件是否超限，可以在超限的那一刻就停手并删掉半成品，而不是先读完再判断。

    摘要用于去重：同一份文件重复上传时直接返回既有记录，不再重复入库 ——
    Chroma 与 BM25 都没有按内容去重的能力，重复上传等于检索结果里出现两份
    同样的片段。
    """
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    safe_name = f"{Path(filename).stem}_{uuid4().hex}{suffix}"
    file_path = settings.upload_dir / safe_name

    limit = settings.max_upload_mb * 1024 * 1024
    digest = hashlib.sha256()
    total = 0

    try:
        with open(file_path, "wb") as out:
            while True:
                block = stream.read(_READ_BLOCK)
                if not block:
                    break
                total += len(block)
                if total > limit:
                    raise UploadTooLargeError(
                        f"文件超过上限 {settings.max_upload_mb}MB（已读 {total // 1024 // 1024}MB）"
                    )
                digest.update(block)
                out.write(block)
    except BaseException:
        # 半成品必须删掉：它会被 `app.rag.ingest --fresh` 当成一份真实文档重新入库
        file_path.unlink(missing_ok=True)
        raise

    return file_path, digest.hexdigest()


def load_file(file_path: Path) -> List[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".txt":
        loader = TextLoader(str(file_path), autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    documents = loader.load()
    # PyPDFLoader / TextLoader 把 metadata["source"] 设成**绝对路径**。它会一路流到
    # 两个不该看到的地方：前端把source 当标题渲染，以及检索结果作为上下文发给
    # LLM —— 等于把自己的目录结构交给模型。统一降级成文件名。
    for document in documents:
        document.metadata["source"] = file_path.name
    return documents


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
