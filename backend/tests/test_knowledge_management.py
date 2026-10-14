"""知识库管理：上传去重、大小上限、四处删除。

索引写入走假实现（不加载 BGE-M3、不碰真实 Chroma/BM25），因此这些用例不需要
模型也不需要 MySQL；SQLite 内存库承担数据库部分。
"""
import io

import pytest
from langchain_core.documents import Document

from app.models.document import Document as DocumentModel
from app.rag import loader as loader_mod
from app.services import knowledge_service as svc


class _FakeVectorStore:
    def __init__(self):
        self.docs = []

    def add_documents(self, documents):
        self.docs.extend(documents)
        return [i for i, _ in enumerate(documents)]

    def delete(self, ids=None):
        keep = {}
        for i, doc in enumerate(self.docs):
            chunk_id = (doc.metadata or {}).get("chunk_id", "")
            keep[chunk_id] = doc
        self.docs = [d for cid, d in keep.items() if cid not in set(ids or [])]


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    """把四处副作用全部替换成内存实现。"""
    store = _FakeVectorStore()
    deleted_sources = []

    class _FakeBM25:
        def __init__(self):
            self.chunks = []

        def add(self, chunks):
            self.chunks.extend(chunks)

        def remove_source(self, source):
            before = len(self.chunks)
            self.chunks = [c for c in self.chunks if (c.metadata or {}).get("source") != source]
            return before - len(self.chunks)

    bm25 = _FakeBM25()

    monkeypatch.setattr(svc, "add_documents", store.add_documents)
    monkeypatch.setattr(svc, "get_bm25_index", lambda: bm25)
    monkeypatch.setattr(svc, "delete_by_source", lambda source: deleted_sources.append(source) or 0)
    monkeypatch.setattr(loader_mod.settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(loader_mod.settings, "data_dir", tmp_path)
    # load_file 需要真实文件；用假实现只关心 metadata 的 source
    monkeypatch.setattr(
        svc, "load_file",
        lambda p: [Document(page_content="内容", metadata={"source": p.name})],
    )
    return store, bm25, deleted_sources, tmp_path


@pytest.fixture
def doc_table(sqlite_session):
    """sqlite_session 只建了 conversations/messages，这里补上 documents。"""
    from app.db.base import Base

    Base.metadata.create_all(sqlite_session.get_bind(), tables=[DocumentModel.__table__])
    return sqlite_session


# --- 上传落盘 ---

def test_stream_save_writes_file_and_returns_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_mod.settings, "upload_dir", tmp_path / "uploads")
    payload = b"hello knowledge base"

    path, digest = loader_mod.save_uploaded_stream(io.BytesIO(payload), "note.txt")

    assert path.read_bytes() == payload
    assert digest == __import__("hashlib").sha256(payload).hexdigest()
    assert path.name.endswith(".txt") and path.name != "note.txt"   # 存储名带 uuid


def test_stream_save_refuses_oversized_file_and_leaves_nothing(tmp_path, monkeypatch):
    """超限时必须删掉半成品：它会被 ingest --fresh 当成真实文档重新入库。"""
    monkeypatch.setattr(loader_mod.settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(loader_mod.settings, "max_upload_mb", 1)

    with pytest.raises(loader_mod.UploadTooLargeError):
        loader_mod.save_uploaded_stream(io.BytesIO(b"x" * (2 * 1024 * 1024)), "big.pdf")

    assert list((tmp_path / "uploads").iterdir()) == []


# --- 重复上传 ---

def test_same_content_under_a_different_name_is_deduplicated(fakes, doc_table):
    store, bm25, _deleted, tmp_path = fakes
    content = b"identical bytes"

    def _write(name):
        p = tmp_path / "uploads" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return p

    first, dup1 = svc.ingest_saved_file(doc_table, _write("a.pdf"), "a.pdf", "hash-1")
    second, dup2 = svc.ingest_saved_file(doc_table, _write("b.pdf"), "b.pdf", "hash-1")

    assert dup1 is False and dup2 is True
    assert second.id == first.id, "重复上传应返回既有记录"
    assert len(store.docs) == 1, "重复上传不得再入库一份"
    assert len(bm25.chunks) == 1
    # 重复那份的磁盘文件也要清掉，否则 ingest --fresh 会把它当独立文档再入库
    assert sorted(p.name for p in (tmp_path / "uploads").iterdir()) == ["a.pdf"]


def test_legacy_rows_without_hash_are_ignored_by_dedup(fakes, doc_table):
    """老库补列后 content_hash 是 NULL：不能因为它是空的就拒收所有上传。"""
    store, _bm25, _deleted, tmp_path = fakes
    legacy = DocumentModel(filename="old.pdf", file_type="pdf", chunk_count=1, file_path="x", content_hash="")
    doc_table.add(legacy)
    doc_table.commit()

    p = tmp_path / "uploads" / "new.pdf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"brand new")

    doc, duplicate = svc.ingest_saved_file(doc_table, p, "new.pdf", "hash-new")

    assert duplicate is False
    assert len(store.docs) == 1


def test_empty_document_is_not_recorded(fakes, doc_table, monkeypatch):
    _store, _bm25, _deleted, tmp_path = fakes
    monkeypatch.setattr(svc, "load_file", lambda p: [])

    p = tmp_path / "uploads" / "scan.pdf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"%PDF-1.4 empty")

    with pytest.raises(ValueError, match="没有从"):
        svc.ingest_saved_file(doc_table, p, "scan.pdf", "hash-empty")

    assert not p.exists(), "解析不出内容的文件不该留在 uploads 里"


# --- 删除 ---

def test_delete_clears_both_indexes_the_file_and_the_row(fakes, doc_table):
    store, bm25, deleted_sources, tmp_path = fakes
    p = tmp_path / "uploads" / "doc.pdf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"content")

    doc, _ = svc.ingest_saved_file(doc_table, p, "doc.pdf", "hash-x")
    assert len(store.docs) == 1 and len(bm25.chunks) == 1

    removed = svc.delete_document(doc_table, doc.id)

    assert removed is not None
    assert bm25.chunks == [], "BM25 片段必须一起清掉"
    assert deleted_sources == [p.name], "Chroma 必须按同一个 source 键删除"
    assert not p.exists(), "磁盘文件必须删除"
    assert doc_table.query(DocumentModel).filter(DocumentModel.id == doc.id).first() is None


def test_delete_of_unknown_id_returns_none(fakes, doc_table):
    _store, _bm25, _deleted, _tmp = fakes
    assert svc.delete_document(doc_table, 99999) is None


def test_delete_touches_the_stored_name_not_the_upload_name(fakes, doc_table):
    """入库时 source 是带 uuid 的存储名；用原始名去删会一条都没删掉。"""
    _store, bm25, deleted_sources, tmp_path = fakes
    stored = tmp_path / "uploads" / "原始文档_abc123.pdf"
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_bytes(b"content")

    doc, _ = svc.ingest_saved_file(doc_table, stored, "原始文档.pdf", "hash-y")
    svc.delete_document(doc_table, doc.id)

    assert deleted_sources == ["原始文档_abc123.pdf"], "必须用存储名（file_path 的 basename）"
