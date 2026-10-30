import hashlib
from pathlib import Path
from typing import BinaryIO, Callable, List, Tuple
from uuid import uuid4

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from app.config import settings
from app.rag.slide_title import (
    build_section_path,
    extract_page_title,
    render_title_prefix,
)

# 流式读写的块大小。1MB 是经验值：再大对磁盘吞吐没明显帮助，再小会增加 Python 层循环开销。
_READ_BLOCK = 1024 * 1024

class _Docx2txtLoader:
    """docx 的最小加载器。

    不走 langchain 的文档加载器：`langchain-community 0.4` 已经不再提供
    `Docx2txtLoader`（拆分到了 langchain-classic），而为一个格式引入
    unstructured 那一整套依赖并不划算。

    **docx2txt 延迟导入**：它是可选依赖，缺它应该只让 .docx 上传失败，
    而不是让整个应用在 import 阶段就崩掉。
    """

    def __init__(self, file_path: str) -> None:
        self._path = file_path

    def load(self) -> List[Document]:
        import docx2txt

        text = docx2txt.process(self._path) or ""
        return [Document(page_content=text, metadata={"source": self._path})]


# 支持的类型 → 加载器。抽成表而不是 if 链：新增类型只改这一处，
# 扩展名判断也就有了单一可测的入口（loader_class_for）。
_LOADERS: dict[str, Callable[[str], object]] = {
    ".pdf": PyPDFLoader,
    ".txt": lambda path: TextLoader(path, autodetect_encoding=True),
    ".md": lambda path: TextLoader(path, encoding="utf-8"),
    ".markdown": lambda path: TextLoader(path, encoding="utf-8"),
    ".docx": _Docx2txtLoader,
}

# Markdown 的语义切分：先按标题层级拆，再按长度细分。
# 元数据键沿用 markdown 标题本身（"Header 1"），便于回溯片段属于哪一节。
_MARKDOWN_HEADERS: dict[str, str] = {
    "#": "Header 1",
    "##": "Header 2",
    "###": "Header 3",
    "####": "Header 4",
    "#####": "Header 5",
    "######": "Header 6",
}


class UploadTooLargeError(ValueError):
    """上传文件超过 `settings.max_upload_mb`。"""


def supported_suffixes() -> list[str]:
    return sorted(_LOADERS)


def loader_class_for(suffix: str) -> Callable[[str], object]:
    """按扩展名返回加载器；不支持的类型抛 ValueError（信息里带上支持清单）。"""
    factory = _LOADERS.get(suffix.lower())
    if factory is None:
        raise ValueError(
            f"不支持的文件类型：{suffix or '（无扩展名）'}；"
            f"当前支持 {' '.join(supported_suffixes())}"
        )
    return factory


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
    loader = loader_class_for(file_path.suffix)(str(file_path))
    documents = loader.load()
    # 各 loader 把 metadata["source"] 设成**绝对路径**。它会一路流到两个不该看到
    # 的地方：前端把 source 当标题渲染，以及检索结果作为上下文发给 LLM ——
    # 等于把自己的目录结构交给模型。统一降级成文件名。
    for document in documents:
        document.metadata["source"] = file_path.name
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """切分并打上稳定 chunk_id。

    **Markdown 先按标题切。** 个人笔记的主力格式就是 Markdown，而纯按 800/120 的
    长度切会把「## 安装依赖」和它下面的正文拦腰截断：片段既不包含标题（检索时
    不知道自己在讲什么），又横跨两节。先按标题层级拆、再对超长小节按长度细分，
    片段才能带上「我是哪一节」这个语义。

    **PDF 补全幻灯片标题。** 课件类 PDF 一页一张幻灯片，正文第一行就是标题。
    标题写进 chunk 文本后，query 的词面能直接对上——「检索得到」与「检索得准」
    之间的差别就在这里。判定规则见 `rag/slide_title.py`，那里的原则是
    **宁可没有标题，也不能有坏标题**（约 40% 的页首行是公式或图形残留）。
    """
    chunks: List[Document] = []

    markdown_docs = [d for d in documents if _is_markdown(d)]
    other_docs = [d for d in documents if not _is_markdown(d)]

    if markdown_docs:
        # 本版本 langchain-text-splitters 的两处API 与旧教程不同：
        # 参数是 headers_to_split_on 且必须是 (分隔符, 名称) 的**列表**（传 dict
        # 会在解包时报 "not enough values to unpack"），方法只有 split_text
        # （没有 split_documents）。照抄旧写法会直接 TypeError/ValueError。
        by_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=list(_MARKDOWN_HEADERS.items()),
            # 保留标题行：否则片段正文里只剩指令（「执行下面这条命令…」），
            # 少了主语，光看片段不知道它在讲什么
            strip_headers=False,
        )
        for doc in markdown_docs:
            for section in by_splitter.split_text(doc.page_content):
                # split_text 产出的 metadata 只有标题层级；原文档的 source 等要带上，
                # 否则 chunk_id 与删除/去重都会认不出这个片段属于哪个文件
                section.metadata = {**doc.metadata, **section.metadata}
                chunks.append(section)

    if other_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks.extend(splitter.split_documents(other_docs))

    return _stamp_chunk_ids(_enrich_with_titles(chunks))


def _enrich_with_titles(chunks: List[Document]) -> List[Document]:
    """把幻灯片标题补进片段文本与元数据。

    **只在 PDF 片段上做。** Markdown 的片段本来就带标题行（MarkdownHeaderTextSplitter
    保留了它），再补一遍会造成重复。

    **标题写进文本，不只是元数据。** 元数据不参与相似度计算，
    放进文本才能让「worklist 算法」这类 query 在词面上直接命中。

    **没抽到标题就原样放过。** 不写空字符串、不写占位符——
    一个假的标题比没有标题更糟：它会稀释向量，还会让模型误以为
    这一片真的属于某个章节。
    """
    for chunk in chunks:
        metadata = chunk.metadata or {}

        # 只处理 PDF 片段：loader 给 PDF 打了 page 字段，MD 没有
        if metadata.get("page") is None:
            continue

        title = extract_page_title(chunk.page_content)
        if not title:
            continue

        document_title = str(metadata.get("title") or "")
        section_path = build_section_path(title, document_title)

        # 标题已经在正文首行（提取成功即意味着首行就是它）→ 正文里已经含标题，
        # 只补 metadata 即可，不必重复写进文本。
        # 判定：正文首行去掉标题后还剩实质内容 → 说明标题不在正文里
        body = chunk.page_content
        first_line = body.splitlines()[0].strip() if body.strip() else ""
        already_present = first_line == title

        metadata["page_title"] = title
        metadata["section_path"] = section_path

        if not already_present:
            prefix = render_title_prefix(title, section_path)
            if prefix:
                chunk.page_content = f"{prefix}\n\n{body}"

    return chunks


def _is_markdown(doc: Document) -> bool:
    """判断是不是 Markdown：看扩展名，其次看 source 字段。"""
    path = str(doc.metadata.get("source") or "")
    return Path(path).suffix.lower() in (".md", ".markdown")


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
