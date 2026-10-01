"""Intermediate representations for the offline structured solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ingestion.models import ParsedEvent, Provenance

SolverStatus = Literal["supported", "ambiguous", "unresolved", "not_found"]


@dataclass(frozen=True)
class QuerySpec:
    """Deterministic structured question representation.

    Multiple public questions share the same operation; the spec never stores a
    question id or a gold answer.
    """

    qtype: str
    operation: str
    raw_question: str
    matched_template: bool
    sport: str | None = None
    year: int | None = None
    season: str | None = None
    event_title: str | None = None
    event_name: str | None = None
    event_desc: str | None = None
    venue: str | None = None
    date_text: str | None = None
    threshold: int | None = None
    comparison: str | None = None
    aggregation: str | None = None
    temporal_relation: str | None = None
    requested_field: str | None = None
    named_year: int | None = None
    inferred_qtype: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "qtype": self.qtype,
            "operation": self.operation,
            "matched_template": self.matched_template,
            "sport": self.sport,
            "year": self.year,
            "season": self.season,
            "event_title": self.event_title,
            "event_name": self.event_name,
            "event_desc": self.event_desc,
            "venue": self.venue,
            "date_text": self.date_text,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "aggregation": self.aggregation,
            "temporal_relation": self.temporal_relation,
            "requested_field": self.requested_field,
            "named_year": self.named_year,
        }


@dataclass(frozen=True)
class EvidenceItem:
    """Structured evidence that can later map to a TigerGraph fact."""

    evidence_id: str
    evidence_type: str
    document_id: str | None
    event_id: str | None
    field_name: str | None
    value: str | None
    raw_text: str | None
    retrieval_method: str
    provenance: Provenance | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "document_id": self.document_id,
            "event_id": self.event_id,
            "field_name": self.field_name,
            "value": self.value,
            "raw_text": self.raw_text,
            "retrieval_method": self.retrieval_method,
        }


@dataclass
class SolverResult:
    status: SolverStatus
    spec: QuerySpec
    method: str
    answer: list[str] | None = None
    reason: str | None = None
    events: list[ParsedEvent] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    elapsed_ms: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return self.status == "supported" and self.answer is not None
