"""Load documents and split them into chunks.

Chunking is word-window based with overlap. I tried splitting on blank lines
first but emails have tiny paragraphs ("Thanks, Priya") that make useless
chunks, so windows ended up simpler and more predictable.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentic_search.types import Chunk, Document


def load_documents(path: str | Path) -> list[Document]:
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(Document(**json.loads(line)))
    return docs


def chunk_document(doc: Document, max_words: int = 120, overlap: int = 30) -> list[Chunk]:
    if overlap >= max_words:
        raise ValueError("overlap has to be smaller than max_words or we'd loop forever")
    words = doc.body.split()
    # prepend the title to every chunk: for emails the subject is often the
    # most informative part and a chunk from the middle would otherwise lose it
    header = f"{doc.title}. "
    chunks = []
    step = max_words - overlap
    start = 0
    n = 0
    while True:
        window = words[start:start + max_words]
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}#{n}",
            doc_id=doc.doc_id,
            source=doc.source,
            date=doc.date,
            title=doc.title,
            text=header + " ".join(window),
        ))
        n += 1
        if start + max_words >= len(words):
            break
        start += step
    return chunks


def chunk_documents(docs: list[Document], max_words: int = 120, overlap: int = 30) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, max_words, overlap))
    return out
