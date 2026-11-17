"""Second stage: hybrid candidates (BM25 + dense, fused with RRF), then a cross-encoder.

Why two stages: the bi-encoder embeds query and passage separately, which is
fast but loses word-level interaction. A cross-encoder reads (query, passage)
together and is much better at ranking, but it's too slow to run on every
chunk. So: cheap retrievers pull a candidate pool, cross-encoder re-orders it.

Candidates come from *both* retrievers because they fail differently: BM25 is
great at exact IDs ("order 482913") and dense is great at paraphrases.
"""

from __future__ import annotations

from agentic_search.retrieval.base import Filters
from agentic_search.types import Hit

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def reciprocal_rank_fusion(result_lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """RRF: score(chunk) = sum over lists of 1 / (k + rank). k=60 is from the
    original Cormack et al. paper and I didn't tune it."""
    scores: dict[str, float] = {}
    first_hit: dict[str, Hit] = {}
    for hits in result_lists:
        for rank, h in enumerate(hits, start=1):
            cid = h.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            first_hit.setdefault(cid, h)
    order = sorted(scores, key=lambda cid: -scores[cid])
    return [Hit(first_hit[cid].chunk, scores[cid], "hybrid") for cid in order]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, bm25, dense, per_retriever_k: int = 30):
        self.bm25 = bm25
        self.dense = dense
        self.per_retriever_k = per_retriever_k

    def search(self, query: str, k: int = 10, filters: Filters | None = None) -> list[Hit]:
        lists = [self.bm25.search(query, self.per_retriever_k, filters),
                 self.dense.search(query, self.per_retriever_k, filters)]
        return reciprocal_rank_fusion(lists)[:k]


class CrossEncoderScorer:
    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER, device: str | None = None):
        from sentence_transformers import CrossEncoder

        from agentic_search.retrieval.dense import realign_parameters

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device or "cpu")
        realign_parameters(self.model.model)

    def predict(self, pairs, batch_size: int = 32):
        return self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)


class RerankRetriever:
    name = "rerank"

    def __init__(self, first_stage, scorer, candidates: int = 30):
        self.first_stage = first_stage
        self.scorer = scorer
        self.candidates = candidates

    def search(self, query: str, k: int = 10, filters: Filters | None = None) -> list[Hit]:
        pool = self.first_stage.search(query, self.candidates, filters)
        if not pool:
            return []
        scores = self.scorer.predict([(query, h.chunk.text) for h in pool])
        ranked = sorted(zip(pool, scores), key=lambda pair: -float(pair[1]))
        return [Hit(h.chunk, float(s), self.name) for h, s in ranked[:k]]
