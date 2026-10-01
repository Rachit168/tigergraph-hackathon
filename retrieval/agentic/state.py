"""Investigation state / slot ledger. Tracks needed information, not presumed answers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from retrieval.graphrag.models import GraphEvidence, GraphRetrievalResult
from retrieval.graphrag.parser import ParsedRetrievalRequest
from retrieval.structured.models import QuerySpec

SlotStatus = Literal["unknown", "searching", "resolved", "ambiguous", "unsupported"]
StopReason = Literal[
    "answered",
    "ambiguous",
    "unresolved",
    "not_found",
    "unsupported",
    "budget_exhausted",
    "no_progress",
    "tool_failure",
]


@dataclass
class Slot:
    slot_id: str
    role: str
    spec: str
    status: SlotStatus = "unknown"
    evidence_ids: list[str] = field(default_factory=list)
    independent_sources: list[str] = field(default_factory=list)
    value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "role": self.role,
            "spec": self.spec,
            "status": self.status,
            "evidence_ids": list(self.evidence_ids),
            "independent_sources": list(self.independent_sources),
            "value": self.value,
        }


def slots_for_request(parsed: ParsedRetrievalRequest) -> list[Slot]:
    spec = parsed.spec
    qtype = spec.qtype
    requested = spec.requested_field or "value"
    slots = [
        Slot(
            slot_id="target_event",
            role="entity",
            spec="Identify the Event vertex or Event set required by the question.",
        )
    ]
    if qtype == "aggregation":
        slots.append(
            Slot(
                slot_id="complete_count",
                role="comparison",
                spec="Count of Events in the complete matching set, not a top-k sample.",
            )
        )
    elif qtype == "superlative":
        slots.append(
            Slot(
                slot_id="max_attribute",
                role="attribute",
                spec="Attribute values needed to identify a unique maximum, if one exists.",
            )
        )
    elif qtype == "multi_hop":
        slots.append(
            Slot(
                slot_id="graph_relation",
                role="relation",
                spec="Typed graph hop confirming venue/date relationship to candidate Events.",
            )
        )
        slots.append(
            Slot(
                slot_id="answer_field",
                role="attribute",
                spec=f"Requested field {requested} only if a unique Event is identified.",
            )
        )
    else:
        slots.append(
            Slot(
                slot_id="answer_field",
                role="attribute",
                spec=f"Requested field {requested} from retrieved Event evidence.",
            )
        )
    return slots


@dataclass
class InvestigationState:
    question: str
    parsed: ParsedRetrievalRequest
    slots: dict[str, Slot]
    event_ids: list[str] = field(default_factory=list)
    evidence: dict[str, GraphEvidence] = field(default_factory=dict)
    retrievals: list[GraphRetrievalResult] = field(default_factory=list)
    fingerprints: set[str] = field(default_factory=set)
    unused_tools: set[str] = field(default_factory=set)
    iteration: int = 0
    follow_ups_used: int = 0
    tool_calls_used: int = 0
    stop_reason: StopReason | None = None
    primary_status: str | None = None
    primary_reason: str | None = None
    primary_operation: str | None = None
    has_chunks: bool = False
    has_neighborhood: bool = False
    last_error: str | None = None
    require_multihop_neighborhood: bool = True

    @classmethod
    def from_parsed(cls, parsed: ParsedRetrievalRequest) -> InvestigationState:
        return cls(
            question=parsed.question,
            parsed=parsed,
            slots={slot.slot_id: slot for slot in slots_for_request(parsed)},
            unused_tools={"retrieve_spec", "supporting_chunks", "event_neighborhood"},
            primary_operation=parsed.spec.operation,
        )

    @property
    def spec(self) -> QuerySpec:
        return self.parsed.spec

    def slot(self, slot_id: str) -> Slot | None:
        return self.slots.get(slot_id)

    def mark_searching(self, *slot_ids: str) -> None:
        for slot_id in slot_ids:
            slot = self.slots.get(slot_id)
            if slot is not None and slot.status == "unknown":
                slot.status = "searching"

    def record_fingerprint(self, fingerprint: str) -> bool:
        if fingerprint in self.fingerprints:
            return False
        self.fingerprints.add(fingerprint)
        return True

    def apply_retrieval(self, result: GraphRetrievalResult, tool_name: str) -> None:
        self.retrievals.append(result)
        self.unused_tools.discard(tool_name)
        for item in result.all_evidence():
            self.evidence[item.evidence_id] = item
        for event_id in result.event_ids:
            if event_id not in self.event_ids:
                self.event_ids.append(event_id)
        if tool_name == "retrieve_spec":
            self.primary_status = result.status
            self.primary_reason = result.reason
            self.primary_operation = result.operation
            self._apply_primary(result)
            if result.chunks:
                self.has_chunks = True
        elif tool_name == "supporting_chunks":
            self.has_chunks = bool(result.chunks)
            self._attach_evidence("target_event", result)
        elif tool_name == "event_neighborhood":
            self.has_neighborhood = bool(result.edges or result.event_ids)
            relation = self.slot("graph_relation")
            if relation is not None:
                if result.status == "supported" and (result.edges or result.event_ids):
                    relation.status = "resolved"
                    relation.evidence_ids = [item.evidence_id for item in result.edges or result.entities]
                    relation.independent_sources = sorted(
                        {item.retrieval_method for item in result.all_evidence() if item.retrieval_method}
                    )
                    relation.value = ",".join(item.field_name or "" for item in result.edges if item.field_name) or None
                elif result.status in {"not_found", "unresolved"}:
                    relation.status = "unsupported"
                self._attach_evidence("graph_relation", result)

    def _apply_primary(self, result: GraphRetrievalResult) -> None:
        target = self.slot("target_event")
        answer = self.slot("answer_field") or self.slot("complete_count") or self.slot("max_attribute")
        if target is not None:
            self._attach_evidence("target_event", result)
            if result.status == "supported":
                target.status = "resolved"
                target.value = ",".join(result.event_ids) or None
            elif result.status == "ambiguous":
                target.status = "ambiguous"
                target.value = None
            elif result.status == "not_found":
                target.status = "unsupported"
            else:
                target.status = "unsupported"
        if answer is not None:
            self._attach_evidence(answer.slot_id, result)
            facts = [item for item in result.facts if item.value]
            titles = [
                item
                for item in result.entities
                if item.value and item.field_name == "title"
            ]
            if result.status == "supported" and facts:
                answer.status = "resolved"
                answer.value = " | ".join(item.value for item in facts if item.value)
            elif (
                result.status == "supported"
                and answer.slot_id == "max_attribute"
                and titles
            ):
                # argmax_competitors already selected the winner Event; the answer
                # slot is that Event title, not a later competitor-fact repair.
                answer.status = "resolved"
                answer.value = " | ".join(item.value for item in titles if item.value)
            elif result.status == "ambiguous":
                answer.status = "ambiguous"
                answer.value = None
            elif result.status == "supported" and result.is_complete_set and result.facts:
                answer.status = "resolved"
            elif result.status in {"not_found", "unresolved"}:
                answer.status = "unsupported"
        relation = self.slot("graph_relation")
        if relation is not None and relation.status == "unknown":
            relation.status = "searching"

    def _attach_evidence(self, slot_id: str, result: GraphRetrievalResult) -> None:
        slot = self.slot(slot_id)
        if slot is None:
            return
        for item in result.all_evidence():
            if item.evidence_id not in slot.evidence_ids:
                slot.evidence_ids.append(item.evidence_id)
            if item.retrieval_method and item.retrieval_method not in slot.independent_sources:
                slot.independent_sources.append(item.retrieval_method)

    def answer_ready(self) -> bool:
        if self.primary_status != "supported":
            return False
        if (
            self.require_multihop_neighborhood
            and self.spec.qtype == "multi_hop"
            and not self.has_neighborhood
        ):
            return False
        answer = self.slot("answer_field") or self.slot("complete_count") or self.slot("max_attribute")
        target = self.slot("target_event")
        if target is not None and target.status != "resolved":
            return False
        if answer is not None and answer.status != "resolved":
            return False
        return True

    def terminal_from_primary(self) -> StopReason | None:
        if self.primary_status == "ambiguous":
            return "ambiguous"
        if self.primary_status == "not_found":
            return "not_found"
        if self.primary_status == "unresolved":
            return "unresolved"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "qtype": self.spec.qtype,
            "operation": self.spec.operation,
            "slots": [slot.to_dict() for slot in self.slots.values()],
            "event_ids": list(self.event_ids),
            "primary_status": self.primary_status,
            "primary_reason": self.primary_reason,
            "iteration": self.iteration,
            "follow_ups_used": self.follow_ups_used,
            "tool_calls_used": self.tool_calls_used,
            "stop_reason": self.stop_reason,
            "has_chunks": self.has_chunks,
            "has_neighborhood": self.has_neighborhood,
        }
