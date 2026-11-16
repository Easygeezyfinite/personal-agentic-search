"""Keyword baseline: BM25 (Okapi) over lowercased word tokens."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from agentic_search.retrieval.base import Filters
from agentic_search.types import Chunk, Hit

# small hand-picked list; a big stopword list hurt queries like "who is ..." less
# than I expected but it made "what time" style queries noisy, so keep it short
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "to", "of", "in", "on", "for", "and", "or",
    "my", "i", "me", "do", "does", "what", "when", "where", "how", "which", "who",
    "it", "at", "with", "be", "that", "this", "did", "we", "should", "can",
}
_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks], k1=k1, b=b)

    def search(self, query: str, k: int = 10, filters: Filters | None = None) -> list[Hit]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(-scores)
        hits = []
        for i in order:
            if scores[i] <= 0:
                break  # no query term in this chunk at all, not a real match
            c = self.chunks[i]
            if filters and not filters(c):
                continue
            hits.append(Hit(c, float(scores[i]), self.name))
            if len(hits) == k:
                break
        return hits
