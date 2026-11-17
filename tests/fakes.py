"""Cheap stand-ins for the neural models so most tests run in milliseconds."""

import hashlib

import numpy as np

from agentic_search.retrieval.bm25 import tokenize


class HashEmbedder:
    """Bag-of-words hashed into a fixed-size unit vector. Not semantic at all,
    but deterministic and good enough to test the FAISS plumbing."""

    model_name = "fake-hash"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class OverlapCrossEncoder:
    """Scores (query, passage) by shared-token count."""

    def predict(self, pairs, batch_size=32):
        return np.array([len(set(tokenize(q)) & set(tokenize(p))) for q, p in pairs], dtype=np.float32)
