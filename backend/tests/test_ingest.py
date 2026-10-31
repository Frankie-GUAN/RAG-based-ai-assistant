"""重新入库命令的测试。

索引目录指向 tmp_path，绝不碰真实的 data/。BGE-M3 用假的 embedding function 顶替，
因此不需要加载 2GB 模型，也不需要打标记。
"""
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag import ingest as ingest_mod
from app.rag.bm25_index import BM25Index


class _FakeEmbeddings(Embeddings):
    """确定性的假 embedding：按文本长度给向量，够 Chroma 建索引即可。"""

    def embed_documents(self, texts):
        return [[float(len(t) % 7)] * 8 for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7)] * 8


@pytest.fixture
def fake_embeddings(monkeypatch):
    """让 Chroma 句柄用假 embedding，避免加载 BGE-M3。"""
    import app.rag.vector_store as vs

    monkeypatch.setattr(vs, "get_embeddings", lambda: _FakeEmbeddings())
    # 句柄是进程内单例：清掉它，让下一行用刚 patch 的 embedding 重新构造
    monkeypatch.setattr(vs, "_vectorstore", None, raising=False)


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(ingest_mod.settings, "upload_dir", upload_dir)
    monkeypatch.setattr(ingest_mod.settings, "data_dir", tmp_path)
    monkeypatch.setattr(ingest_mod.settings, "chroma_dir", tmp_path / "chroma")
    return upload_dir


def _write_txt(path, text):
    path.write_text(text, encoding="utf-8")


def test_reingest_writes_both_indexes(uploads, fake_embeddings, monkeypatch):
    monkeypatch.setattr(
        ingest_mod, "load_file",
        lambda p: [Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})],
    )
    _write_txt(uploads / "a.txt", "劳动合同法第三十六条规定协商一致可以解除劳动合同")
    _write_txt(uploads / "b.txt", "FastAPI 是现代 Python Web 框架，支持异步")

    report = ingest_mod.reingest_uploads()

    assert report.files_ok == 2
    assert report.files_failed == 0
    assert report.chunks > 0

    # BM25 落盘且能被重新加载
    index = BM25Index()
    index._index_path = uploads.parent / "bm25_index.pkl"
    assert index.load() is True
    assert index.document_count == report.chunks


def test_reingest_continues_past_one_bad_file(uploads, fake_embeddings, monkeypatch):
    def _load(p):
        if p.name == "broken.txt":
            raise ValueError("模拟解析失败")
        return [Document(page_content="正常内容", metadata={"source": p.name})]

    monkeypatch.setattr(ingest_mod, "load_file", _load)
    _write_txt(uploads / "broken.txt", "坏的")
    _write_txt(uploads / "good.txt", "好的")

    report = ingest_mod.reingest_uploads()

    assert report.files_ok == 1
    assert report.files_failed == 1
    assert report.failures[0][0] == "broken.txt"


def test_fresh_moves_the_old_chroma_dir_instead_of_deleting(uploads, fake_embeddings, monkeypatch):
    monkeypatch.setattr(
        ingest_mod, "load_file",
        lambda p: [Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})],
    )
    _write_txt(uploads / "a.txt", "内容")
    chroma_dir = uploads.parent / "chroma"
    chroma_dir.mkdir()
    (chroma_dir / "old.bin").write_bytes(b"x")

    report = ingest_mod.reingest_uploads(fresh=True)

    assert report.backup_dir is not None
    assert report.backup_dir.exists()          # 原数据还在
    assert (report.backup_dir / "old.bin").read_bytes() == b"x"
    assert not (report.backup_dir.parent / "chroma" / "old.bin").exists()


def test_refuses_to_run_with_no_uploadable_files(uploads):
    # 注意用 .exe 而不是 .md：Markdown 已经是支持格式了
    (uploads / "tool.exe").write_bytes(b"MZ")

    with pytest.raises(RuntimeError, match="没有可入库的文件"):
        ingest_mod.reingest_uploads()