"""文档解析：支持的类型、Markdown 的语义切分。

不加载 BGE-M3/Chroma：这里只测loader 与 splitter 本身。
"""
import pytest
from langchain_core.documents import Document

from app.rag import loader as loader_mod


# --- 支持的类型 ---

def test_supported_suffixes_cover_the_note_formats():
    """.md 是个人笔记的主力格式，.docx 是别人发来的格式 —— 两者曾经都被拒绝。"""
    suffixes = loader_mod.supported_suffixes()

    assert {".pdf", ".txt", ".md", ".markdown", ".docx"} <= set(suffixes)


def test_unsupported_type_error_lists_what_is_supported():
    with pytest.raises(ValueError) as exc:
        loader_mod.loader_class_for(".exe")

    assert ".pdf" in str(exc.value)


def test_loader_class_is_case_insensitive():
    assert loader_mod.loader_class_for(".PDF") is loader_mod.loader_class_for(".pdf")


# --- Markdown 切分 ---

MARKDOWN = """# 项目说明

这是顶层简介，够短不会被切开。

## 安装依赖

先装 Python 3.11 以上版本，然后执行下面这条命令来安装全部依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

启动服务之后打开浏览器就能用了。

### 常见问题

如果端口被占用，就换一个端口启动。
"""


def _md_doc(tmp_path, name="note.md", text=MARKDOWN):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return loader_mod.load_file(path)


def test_markdown_is_split_on_headers_not_length(tmp_path):
    """每一节各自成片段，且带上标题元数据 —— 否则检索命中后不知道片段在讲什么。"""
    documents = _md_doc(tmp_path)

    chunks = loader_mod.split_documents(documents)

    sections = {
        tuple(sorted((k, v) for k, v in c.metadata.items() if k.startswith("Header")))
        for c in chunks
    }
    assert any("安装依赖" in dict(s).values() for s in sections), sections
    assert any("使用方法" in dict(s).values() for s in sections), sections


def test_markdown_chunk_content_still_contains_its_header(tmp_path):
    """标题行必须留在片段里：只有元数据的话，光看片段正文仍是「一条没有主语的指令」。"""
    documents = _md_doc(tmp_path)

    chunks = loader_mod.split_documents(documents)

    install = [c for c in chunks if "安装依赖" in c.page_content]
    assert install, "找不到包含「安装依赖」的片段"
    assert "pip install -r requirements.txt" in install[0].page_content


def test_markdown_sections_are_not_merged_across_headers(tmp_path):
    """标题是硬边界：不同小节的内容不该被拼进同一个片段。"""
    documents = _md_doc(tmp_path)

    chunks = loader_mod.split_documents(documents)

    for chunk in chunks:
        headers = {v for k, v in chunk.metadata.items() if k.startswith("Header")}
        if "安装依赖" in headers:
            assert "常见问题" not in chunk.page_content, "片段跨越了两个小节"


def test_chunk_ids_are_unique_for_markdown(tmp_path):
    """Markdown 没有 page 字段，计数必须落在 (source, page) 二元组上，否则 id 会重复。"""
    documents = _md_doc(tmp_path)

    chunks = loader_mod.split_documents(documents)

    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id 出现重复"
    assert all("#p" in i for i in ids)


def test_non_markdown_keeps_using_length_based_splitting(tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("普通文本。" * 500, encoding="utf-8")

    chunks = loader_mod.split_documents(loader_mod.load_file(path))

    assert len(chunks) > 1, "长文本仍应按长度切"
    assert all("Header" not in c.metadata for c in chunks)


def test_mixed_corpus_splits_each_document_by_its_own_rule(tmp_path):
    """同一批文档里混着 md 与 txt 时，两边各自按自己的规则处理。"""
    md_path = tmp_path / "a.md"
    md_path.write_text(MARKDOWN, encoding="utf-8")
    txt_path = tmp_path / "b.txt"
    txt_path.write_text("纯文本内容。" * 300, encoding="utf-8")

    chunks = loader_mod.split_documents(
        loader_mod.load_file(md_path) + loader_mod.load_file(txt_path)
    )

    md_chunks = [c for c in chunks if c.metadata["source"] == "a.md"]
    txt_chunks = [c for c in chunks if c.metadata["source"] == "b.txt"]
    assert md_chunks and txt_chunks
    assert any(k.startswith("Header") for c in md_chunks for k in c.metadata)
    assert not any(k.startswith("Header") for c in txt_chunks for k in c.metadata)


def test_source_is_the_file_name_not_the_absolute_path(tmp_path):
    """source 会展示给用户、也会发给模型，不能带绝对路径。"""
    documents = _md_doc(tmp_path)

    assert all(d.metadata["source"] == "note.md" for d in documents)
