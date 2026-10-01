"""Retrieval hits and traces for the offline text substrate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ingestion.models import TextChunk


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    document_id: str
    document_title: str
    score: float
    rank: int
    text: str
    section: str
    kind: str
    event_id: str | None
    retrieval_method: str
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "score": self.score,
            "rank": self.rank,
            "section": self.section,
            "kind": self.kind,
            "event_id": self.event_id,
            "retrieval_method": self.retrieval_method,
        }


@dataclass
class RetrievalResult:
    query: str
    method: str
    hits: list[RetrievalHit]
    elapsed_ms: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)

    def document_ranks(self) -> dict[str, int]:
        ranks: dict[str, int] = {}
        for hit in self.hits:
            ranks.setdefault(hit.document_id, hit.rank)
        return ranks
