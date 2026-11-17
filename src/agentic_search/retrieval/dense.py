"""Dense retrieval: sentence-transformers embeddings in a FAISS index.

Embeddings are L2-normalized, so an inner-product index (IndexFlatIP) gives
cosine similarity directly. Flat (exact) search is fine at personal scale:
a few thousand chunks is nothing. If this ever indexes years of email I'd
switch to IVF or HNSW, but that's a problem for later.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from agentic_search.retrieval.base import Filters
from agentic_search.types import Chunk, Hit

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def realign_parameters(module) -> None:
    """Copy every weight into freshly allocated memory.

    Bug I hit on macOS (arm64): safetensors memory-maps the weights file and
    some tensors end up at addresses that aren't even 4-byte aligned (the
    cross-encoder's pooler weight had data_ptr % 64 == 2). Apple's Accelerate
    matmul then either returned NaN for every score or crashed with SIGBUS.
    The reranker was silently returning NaNs, so "re-ranked" results were
    just the first-stage order. Cloning the params gives aligned buffers.
    """
    import torch

    with torch.no_grad():
        for p in module.parameters():
            p.data = p.data.clone().contiguous()


class Embedder:
    """Thin wrapper so tests can swap in a fake without loading torch models."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # slow import, keep it lazy

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device or "cpu")
        realign_parameters(self.model)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        return vecs.astype(np.float32)


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder, vectors: np.ndarray | None = None):
        self.chunks = chunks
        self.embedder = embedder
        if vectors is None:
            vectors = embedder.encode([c.text for c in chunks])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, k: int = 10, filters: Filters | None = None) -> list[Hit]:
        q = self.embedder.encode([query])
        # with a filter, over-fetch and drop non-matching chunks afterwards.
        # FAISS has IDSelector for this, but over-fetching is simpler and at
        # this size we can just look at everything.
        fetch = self.index.ntotal if (filters and not filters.is_empty()) else min(k, self.index.ntotal)
        scores, ids = self.index.search(q, fetch)
        hits = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            c = self.chunks[i]
            if filters and not filters(c):
                continue
            hits.append(Hit(c, float(score), self.name))
            if len(hits) == k:
                break
        return hits

    # --- persistence: so `pas ask` doesn't re-embed everything every time

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "chunks.faiss"))
        with open(d / "meta.json", "w") as f:
            json.dump({"model": getattr(self.embedder, "model_name", "unknown"),
                       "chunk_ids": [c.chunk_id for c in self.chunks]}, f)

    @classmethod
    def load(cls, directory: str | Path, chunks: list[Chunk], embedder) -> "DenseRetriever":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        if meta["chunk_ids"] != [c.chunk_id for c in chunks]:
            raise ValueError("index is stale (chunks changed since it was built) - run `pas ingest` again")
        index = faiss.read_index(str(d / "chunks.faiss"))
        vectors = index.reconstruct_n(0, index.ntotal)
        return cls(chunks, embedder, vectors=vectors)
