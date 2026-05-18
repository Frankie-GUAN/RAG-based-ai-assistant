from langchain_core.documents import Document
from app.rag.hybrid_search import _rrf_fuse


def test_rrf_fusion_ranks_both_sources():
    docs = [
        Document(page_content="A"),
        Document(page_content="B"),
        Document(page_content="C"),
    ]
    vector = [(docs[0], 0.9), (docs[1], 0.5)]
    bm25 = [(docs[1], 0.8), (docs[2], 0.3)]

    result = _rrf_fuse(vector, bm25, top_k=3)
    ids = [doc.page_content for doc, _ in result]
    # B appears in both lists -> should rank highest
    assert ids[0] == "B"
    assert len(result) == 3


def test_rrf_empty_inputs():
    result = _rrf_fuse([], [], top_k=5)
    assert result == []


def test_rrf_handles_duplicate_content():
    doc = Document(page_content="same")
    result = _rrf_fuse([(doc, 1.0)], [(doc, 1.0)], top_k=5)
    assert len(result) == 1
