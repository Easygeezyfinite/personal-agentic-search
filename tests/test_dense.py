import pytest
from fakes import HashEmbedder

from agentic_search.retrieval.base import Filters
from agentic_search.retrieval.dense import DenseRetriever
from agentic_search.types import Chunk


def _chunks():
    return [
        Chunk("email-0001#0", "email-0001", "email", "2026-03-01", "Flight", "flight YVR departs early morning"),
        Chunk("note-0001#0", "note-0001", "note", "2026-01-10", "Recipe", "banana bread bake oven"),
        Chunk("note-0002#0", "note-0002", "note", "2026-05-02", "OS", "round robin time quantum"),
    ]


def test_nearest_neighbour_is_the_matching_chunk():
    r = DenseRetriever(_chunks(), HashEmbedder())
    assert r.search("banana bread", k=1)[0].chunk.doc_id == "note-0001"


def test_k_larger_than_index_is_ok():
    r = DenseRetriever(_chunks(), HashEmbedder())
    assert len(r.search("anything", k=50)) == 3


def test_filter_searches_whole_index():
    r = DenseRetriever(_chunks(), HashEmbedder())
    hits = r.search("banana bread", k=1, filters=Filters(source="email"))
    assert [h.chunk.doc_id for h in hits] == ["email-0001"]


def test_save_and_load_roundtrip(tmp_path):
    chunks = _chunks()
    r = DenseRetriever(chunks, HashEmbedder())
    r.save(tmp_path)
    r2 = DenseRetriever.load(tmp_path, chunks, HashEmbedder())
    assert [h.chunk.chunk_id for h in r2.search("round robin")] == [h.chunk.chunk_id for h in r.search("round robin")]


def test_load_rejects_stale_index(tmp_path):
    chunks = _chunks()
    DenseRetriever(chunks, HashEmbedder()).save(tmp_path)
    with pytest.raises(ValueError, match="stale"):
        DenseRetriever.load(tmp_path, chunks[:2], HashEmbedder())
