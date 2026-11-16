"""Core data types shared by ingest, retrieval and the agent."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    source: str  # "email" | "note" | "doc"
    title: str
    date: str  # ISO yyyy-mm-dd, keeps date filtering a plain string compare
    body: str
    meta: dict = field(default_factory=dict)  # e.g. {"from": ..., "to": ...} for emails


@dataclass
class Chunk:
    chunk_id: str  # "<doc_id>#<n>" so a citation always points back at a document
    doc_id: str
    source: str
    date: str
    title: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float
    retriever: str = ""
