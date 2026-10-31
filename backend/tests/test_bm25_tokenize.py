"""BM25 分词与重建路径测试。

不加载 BGE-M3 / Chroma：Chroma 句柄在这里是假的，索引写到 tmp_path，
因此不会碰磁盘上那份真实的 data/bm25_index.pkl。
"""
import pytest
from langchain_core.documents import Document

from app.rag import bm25_index as mod
from app.rag.bm25_index import BM25Index


# --- 分词 ---

def test_keeps_compound_terms_not_just_single_chars():
    """中文必须保留多字词。单字切分会让「劳动合同」永远匹配不上。"""
    tokens = BM25Index._tokenize("劳动合同如何解除")

    assert "劳动合同" in tokens
    assert "劳动" in tokens and "合同" in tokens


def test_drops_stopwords_and_punctuation():
    tokens = BM25Index._tokenize("这个劳动合同的条款是什么？")

    assert "的" not in tokens
    assert "这个" not in tokens
    assert "？" not in tokens
    assert all(any(ch.isalnum() for ch in t) for t in tokens)


def test_latin_and_numbers_lowercased():
    tokens = BM25Index._tokenize("Python 3.11 AsyncIO")

    assert "python" in tokens
    assert "3.11" in tokens
    assert "asyncio" in tokens


def test_query_and_document_share_the_same_tokenizer():
    """查询侧与语料侧必须走同一个分词器，否则两边词形对不上等于没索引。"""
    query = BM25Index._tokenize("解除劳动合同")
    doc = BM25Index._tokenize("用人单位与劳动者协商一致可以解除劳动合同")

    assert set(query) & set(doc)


def test_search_finds_a_document_by_its_compound_term():
    """端到端（纯内存）：用复合词查询应当命中含该词的片段。

    语料必须够大：`idf = log((N-df+0.5)/(df+0.5))` 只在 `df < N/2` 时为正，
    而 search() 会过滤掉 `score <= 0` 的结果。N=2 时任何词的 idf 都不为正，
    于是**无论查什么都是空结果** —— 这是 BM25 的数学，不是索引坏了
    （CLAUDE.md 记录过这个坑）。

    断言只要求「前两名都出自劳动法文档」：两篇都含「劳动合同」「解除」，都是合法
    命中，谁排第一由 BM25 的长度归一化决定（短文档占便宜），不该被写死。
    """
    index = BM25Index()
    index.build([
        Document(page_content="劳动合同法第三十六条规定协商一致可以解除劳动合同", metadata={"source": "law.pdf"}),
        Document(page_content="Python是一种广泛使用的高级编程语言", metadata={"source": "py.pdf"}),
        Document(page_content="FastAPI 是现代 Python Web 框架", metadata={"source": "py.pdf"}),
        Document(page_content="劳动者提前三十日书面通知可以解除劳动合同", metadata={"source": "law.pdf"}),
        Document(page_content="ChromaDB 用于向量持久化", metadata={"source": "infra.pdf"}),
    ])

    hits = index.search("劳动合同怎么解除", top_k=5)

    assert hits, "复合词查询应当有命中"
    assert [doc.metadata["source"] for doc, _ in hits[:2]] == ["law.pdf", "law.pdf"]


# --- 重建 ---

class _FakeVectorStore:
    def __init__(self, payload):
        self._payload = payload

    def get(self, include=None):
        return self._payload


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    """把索引路径指到 tmp_path，并让 get_bm25_index 返回这个隔离实例。"""
    monkeypatch.setattr(mod.settings, "data_dir", tmp_path)
    index = BM25Index()
    monkeypatch.setattr(mod, "get_bm25_index", lambda: index)
    return index


def test_rebuild_from_chroma_restores_every_chunk(isolated_index, monkeypatch):
    monkeypatch.setattr(
        "app.rag.vector_store.get_vectorstore",
        lambda: _FakeVectorStore({
            "documents": ["劳动合同法第三十六条", "FastAPI 支持异步"],
            "metadatas": [{"source": "law.pdf"}, {"source": "py.pdf"}],
        }),
    )

    count = mod.rebuild_from_chroma()

    assert count == 2
    assert isolated_index.document_count == 2
    assert (mod.settings.data_dir / "bm25_index.pkl").exists()


def test_rebuild_survives_a_broken_existing_index(isolated_index, monkeypatch):
    """索引读不出来时，重建应当自愈 —— 这正是 save(force=True) 的用途。"""
    isolated_index._load_failed = True
    monkeypatch.setattr(
        "app.rag.vector_store.get_vectorstore",
        lambda: _FakeVectorStore({
            "documents": ["只有一条"],
            "metadatas": [{"source": "a.pdf"}],
        }),
    )

    mod.rebuild_from_chroma()

    assert isolated_index._load_failed is False
    assert isolated_index.document_count == 1


def test_rebuild_refuses_an_empty_chroma(isolated_index, monkeypatch):
    """空语料不能覆盖磁盘：会把 2000 篇索引清成 0 篇。"""
    monkeypatch.setattr(
        "app.rag.vector_store.get_vectorstore",
        lambda: _FakeVectorStore({"documents": [], "metadatas": []}),
    )

    with pytest.raises(RuntimeError):
        mod.rebuild_from_chroma()

    assert not (mod.settings.data_dir / "bm25_index.pkl").exists()


def test_rebuild_reports_a_missing_vector_store(isolated_index, monkeypatch):
    monkeypatch.setattr("app.rag.vector_store.get_vectorstore", lambda: None)

    with pytest.raises(RuntimeError):
        mod.rebuild_from_chroma()