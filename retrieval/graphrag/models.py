"""GraphRAG retrieval contracts. No LLM, no agent loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from retrieval.graph.results import GraphQueryResult
from retrieval.structured.models import QuerySpec

Cardinality = Literal["empty", "one", "multiple", "ambiguous", "complete_set", "unresolved"]
EvidenceType = Literal["entity", "fact", "chunk", "edge"]

RETRIEVAL_QUERIES = ("chunks_for_events", "event_neighborhood")

LOOKUP_OPERATIONS = frozenset({"lookup_nations", "lookup_event"})
AGGREGATION_OPERATIONS = frozenset({"count_over_threshold"})
TEMPORAL_OPERATIONS = frozenset({"previous_event_gold", "next_event_gold"})
VENUE_OPERATIONS = frozenset({"events_at_venue_date"})
NEIGHBORHOOD_OPERATION = "event_neighborhood"
CHUNK_OPERATION = "chunks_for_events"


@dataclass(frozen=True)
class GraphRef:
    vertex_type: str | None = None
    vertex_id: str | None = None
    edge_type: str | None = None
    attribute: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEvidence:
    evidence_id: str
    evidence_type: str
    retrieval_method: str
    tool_call_id: str
    graph_refs: list[GraphRef] = field(default_factory=list)
    event_id: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    source_chunk_id: str | None = None
    field_name: str | None = None
    value: str | None = None
    text: str | None = None
    retrieval_score: float | None = None
    source_url: str | None = None
    source_date: str | None = None
    source_version: str | None = None
    observation_truncated: bool = False
    why_retrieved: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["graph_refs"] = [ref.to_dict() for ref in self.graph_refs]
        return payload


@dataclass
class GraphRetrievalResult:
    operation: str
    query_name: str
    status: str
    cardinality: str
    tool_call_id: str
    retrieval_method: str
    params: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    event_ids: list[str] = field(default_factory=list)
    facts: list[GraphEvidence] = field(default_factory=list)
    entities: list[GraphEvidence] = field(default_factory=list)
    chunks: list[GraphEvidence] = field(default_factory=list)
    edges: list[GraphEvidence] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    spec: QuerySpec | None = None
    graph_result: GraphQueryResult | None = None

    @property
    def has_supporting_chunks(self) -> bool:
        return bool(self.chunks)

    @property
    def is_complete_set(self) -> bool:
        return self.cardinality == "complete_set"

    def all_evidence(self) -> list[GraphEvidence]:
        return [*self.entities, *self.facts, *self.edges, *self.chunks]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "query_name": self.query_name,
            "status": self.status,
            "cardinality": self.cardinality,
            "reason": self.reason,
            "tool_call_id": self.tool_call_id,
            "retrieval_method": self.retrieval_method,
            "params": dict(self.params),
            "event_ids": list(self.event_ids),
            "facts": [item.to_dict() for item in self.facts],
            "entities": [item.to_dict() for item in self.entities],
            "chunks": [item.to_dict() for item in self.chunks],
            "edges": [item.to_dict() for item in self.edges],
            "has_supporting_chunks": self.has_supporting_chunks,
            "is_complete_set": self.is_complete_set,
            "notes": dict(self.notes),
            "elapsed_ms": self.elapsed_ms,
            "spec": self.spec.to_dict() if self.spec is not None else None,
        }
