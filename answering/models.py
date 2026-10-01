"""Shared context, citation, generation, and answer contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from retrieval.graphrag.models import GraphRef, GraphRetrievalResult

if TYPE_CHECKING:
    from retrieval.graphrag.parser import ParsedRetrievalRequest

AnswerStatus = Literal[
    "answered",
    "ambiguous",
    "not_found",
    "unresolved",
    "unsupported",
    "abstained",
    "invalid_citations",
    "generation_error",
]


@dataclass(frozen=True)
class PackedContextItem:
    evidence_id: str
    evidence_type: str
    retrieval_method: str
    tool_call_id: str | None
    graph_refs: tuple[GraphRef, ...] = ()
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


@dataclass(frozen=True)
class PackedContext:
    items: tuple[PackedContextItem, ...]
    retrieval_status: str
    cardinality: str
    answer_suppressed: bool
    complete_set: bool
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.items)

    @property
    def textual_items(self) -> tuple[PackedContextItem, ...]:
        return tuple(item for item in self.items if item.evidence_type == "chunk" and bool((item.text or "").strip()))

    @property
    def structured_items(self) -> tuple[PackedContextItem, ...]:
        return tuple(
            item
            for item in self.items
            if item.evidence_type in {"fact", "entity", "edge"}
            and (item.value is not None or item.field_name is not None)
        )

    @property
    def is_ambiguous(self) -> bool:
        return self.cardinality == "ambiguous" or self.retrieval_status == "ambiguous"

    def item_by_id(self) -> dict[str, PackedContextItem]:
        return {item.evidence_id: item for item in self.items}

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "evidence_ids": list(self.evidence_ids),
            "retrieval_status": self.retrieval_status,
            "cardinality": self.cardinality,
            "answer_suppressed": self.answer_suppressed,
            "complete_set": self.complete_set,
            "notes": dict(self.notes),
        }


@dataclass(frozen=True)
class Citation:
    evidence_id: str
    source_url: str | None = None
    source_date: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    event_id: str | None = None

    @classmethod
    def from_context_item(cls, item: PackedContextItem) -> Citation:
        return cls(
            evidence_id=item.evidence_id,
            source_url=item.source_url,
            source_date=item.source_date,
            document_id=item.document_id,
            chunk_id=item.chunk_id,
            event_id=item.event_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratorRequest:
    question: str
    parsed_request: ParsedRetrievalRequest | None
    context: PackedContext
    qtype: str | None = None

    def __post_init__(self) -> None:
        if self.qtype:
            return
        if self.parsed_request is None:
            return
        spec = getattr(self.parsed_request, "spec", None)
        derived = getattr(spec, "qtype", None)
        if derived:
            object.__setattr__(self, "qtype", derived)


@dataclass(frozen=True)
class GeneratorResult:
    answer_text: str
    citation_ids: tuple[str, ...] = ()
    status: str = "answered"
    warnings: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CitationValidation:
    valid: bool
    citations: tuple[Citation, ...] = ()
    evidence_used: tuple[PackedContextItem, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineTimings:
    parsing_ms: float = 0.0
    retrieval_ms: float = 0.0
    packing_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class CitedAnswer:
    question: str
    answer_text: str
    citations: list[Citation]
    retrieval_result: GraphRetrievalResult
    evidence_used: list[PackedContextItem]
    status: AnswerStatus | str
    parsed_request: ParsedRetrievalRequest | None = None
    warnings: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    timings: PipelineTimings = field(default_factory=PipelineTimings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer_text": self.answer_text,
            "citations": [citation.to_dict() for citation in self.citations],
            "retrieval_result": self.retrieval_result.to_dict(),
            "evidence_used": [item.to_dict() for item in self.evidence_used],
            "status": self.status,
            "parsed_request": self.parsed_request.to_dict() if self.parsed_request else None,
            "warnings": list(self.warnings),
            "notes": dict(self.notes),
            "timings": self.timings.to_dict(),
        }
