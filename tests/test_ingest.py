from pathlib import Path

import pytest

from agentic_search.ingest import chunk_document, chunk_documents, load_documents
from agentic_search.synth.generate import build_corpus
from agentic_search.types import Document

DATA = Path(__file__).resolve().parents[1] / "data"


def _doc(n_words: int) -> Document:
    return Document("note-0001", "note", "Title", "2026-01-01", " ".join(f"w{i}" for i in range(n_words)))


def test_short_doc_is_one_chunk():
    chunks = chunk_document(_doc(10), max_words=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "note-0001#0"
    assert chunks[0].text.startswith("Title. w0")


def test_long_doc_windows_overlap_and_cover_every_word():
    chunks = chunk_document(_doc(250), max_words=100, overlap=20)
    assert [c.chunk_id for c in chunks] == ["note-0001#0", "note-0001#1", "note-0001#2"]
    # second window starts 80 words in, so w80..w99 appear in both
    assert "w99" in chunks[0].text and "w80" in chunks[1].text
    assert "w249" in chunks[-1].text


def test_overlap_must_be_smaller_than_window():
    with pytest.raises(ValueError):
        chunk_document(_doc(10), max_words=10, overlap=10)


def test_committed_corpus_matches_generator():
    # if someone edits the generator they need to re-run scripts/generate_corpus.py
    docs, queries = build_corpus()
    committed = load_documents(DATA / "corpus.jsonl")
    assert [d.doc_id for d in docs] == [d.doc_id for d in committed]
    assert docs[0].body == committed[0].body


def test_every_query_label_points_at_a_real_doc():
    docs, queries = build_corpus()
    ids = {d.doc_id for d in docs}
    assert queries
    for q in queries:
        assert q["relevant"] and all(r in ids for r in q["relevant"]), q


def test_chunks_keep_metadata():
    docs = load_documents(DATA / "corpus.jsonl")
    chunks = chunk_documents(docs)
    assert len(chunks) >= len(docs)
    email = next(c for c in chunks if c.source == "email")
    assert email.doc_id.startswith("email-") and email.date
