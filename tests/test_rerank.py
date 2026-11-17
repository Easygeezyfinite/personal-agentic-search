from fakes import HashEmbedder, OverlapCrossEncoder

from agentic_search.retrieval.bm25 import BM25Retriever
from agentic_search.retrieval.dense import DenseRetriever
from agentic_search.retrieval.rerank import HybridRetriever, RerankRetriever, reciprocal_rank_fusion
from agentic_search.types import Chunk, Hit


def _c(i):
    return Chunk(f"d{i}#0", f"d{i}", "note", "2026-01-01", "t", f"text {i}")


def test_rrf_rewards_agreement():
    a, b, c = _c(1), _c(2), _c(3)
    # b is 2nd in both lists, a is 1st in only one -> b should win
    fused = reciprocal_rank_fusion([[Hit(a, 9), Hit(b, 8)], [Hit(c, 9), Hit(b, 8)]])
    assert fused[0].chunk.chunk_id == "d2#0"
    assert {h.chunk.chunk_id for h in fused} == {"d1#0", "d2#0", "d3#0"}


def test_reranker_reorders_candidates():
    chunks = [
        Chunk("a#0", "a", "note", "2026-01-01", "x", "deadlock circular wait hold"),
        Chunk("b#0", "b", "note", "2026-01-01", "x", "deadlock"),
        Chunk("c#0", "c", "note", "2026-01-01", "x", "banana bread"),
    ]
    hybrid = HybridRetriever(BM25Retriever(chunks), DenseRetriever(chunks, HashEmbedder()))
    rr = RerankRetriever(hybrid, OverlapCrossEncoder(), candidates=3)
    hits = rr.search("deadlock circular wait", k=2)
    assert [h.chunk.doc_id for h in hits] == ["a", "b"]
    assert all(h.retriever == "rerank" for h in hits)


def test_reranker_empty_pool():
    chunks = [Chunk("a#0", "a", "note", "2026-01-01", "x", "hello")]

    class Empty:
        def search(self, *a, **kw):
            return []

    assert RerankRetriever(Empty(), OverlapCrossEncoder()).search("q") == []
