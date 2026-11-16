from agentic_search.retrieval.base import Filters, dedupe_by_doc
from agentic_search.retrieval.bm25 import BM25Retriever, tokenize
from agentic_search.types import Chunk, Hit


def _chunks():
    return [
        Chunk("email-0001#0", "email-0001", "email", "2026-03-01", "Flight", "Flight YYC to YVR departs 07:45"),
        Chunk("note-0001#0", "note-0001", "note", "2026-01-10", "Recipe", "banana bread bake 350F for an hour"),
        Chunk("note-0002#0", "note-0002", "note", "2026-05-02", "OS", "round robin time quantum context switch"),
        Chunk("note-0002#1", "note-0002", "note", "2026-05-02", "OS", "round robin preempts the process"),
    ]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What time is the flight to YVR?") == ["time", "flight", "yvr"]


def test_exact_term_ranks_first():
    r = BM25Retriever(_chunks())
    hits = r.search("YVR flight")
    assert hits[0].chunk.doc_id == "email-0001"


def test_no_overlap_returns_nothing():
    r = BM25Retriever(_chunks())
    assert r.search("kubernetes") == []
    assert r.search("the of and") == []


def test_filters_apply():
    r = BM25Retriever(_chunks())
    hits = r.search("round robin flight", filters=Filters(source="email"))
    assert {h.chunk.source for h in hits} == {"email"}
    hits = r.search("round robin", filters=Filters(date_to="2026-04-01"))
    assert hits == []


def test_dedupe_keeps_best_chunk_per_doc():
    cs = _chunks()
    hits = [Hit(cs[2], 3.0), Hit(cs[3], 2.0), Hit(cs[0], 1.0)]
    assert [h.chunk.chunk_id for h in dedupe_by_doc(hits)] == ["note-0002#0", "email-0001#0"]
