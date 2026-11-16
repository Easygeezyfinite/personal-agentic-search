"""Shared bits for retrievers: chunk filters and doc-level de-duplication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from agentic_search.types import Chunk, Hit

ChunkFilter = Callable[[Chunk], bool]


@dataclass
class Filters:
    """Metadata constraints. Dates are ISO strings so comparing them as strings works."""

    source: str | None = None
    date_from: str | None = None
    date_to: str | None = None

    def is_empty(self) -> bool:
        return not (self.source or self.date_from or self.date_to)

    def __call__(self, chunk: Chunk) -> bool:
        if self.source and chunk.source != self.source:
            return False
        if self.date_from and chunk.date < self.date_from:
            return False
        if self.date_to and chunk.date > self.date_to:
            return False
        return True


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10, filters: Filters | None = None) -> list[Hit]: ...


def dedupe_by_doc(hits: list[Hit]) -> list[Hit]:
    """Keep only the best-ranked chunk per document (hits must already be sorted)."""
    seen: set[str] = set()
    out = []
    for h in hits:
        if h.chunk.doc_id not in seen:
            seen.add(h.chunk.doc_id)
            out.append(h)
    return out
