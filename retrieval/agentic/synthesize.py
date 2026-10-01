"""Merge bounded retrieval observations into one provenance-preserving result."""

from __future__ import annotations

from retrieval.graphrag.models import GraphEvidence, GraphRetrievalResult
from retrieval.graphrag.normalize import empty_result


def merge_retrievals(
    retrievals: list[GraphRetrievalResult],
    *,
    operation: str,
    status: str,
    reason: str | None,
) -> GraphRetrievalResult:
    if not retrievals:
        return empty_result(
            operation=operation,
            query_name=operation,
            status=status,
            reason=reason or "no_retrievals",
            tool_call_id="agent:none",
        )
    event_ids: list[str] = []
    entities: list[GraphEvidence] = []
    facts: list[GraphEvidence] = []
    edges: list[GraphEvidence] = []
    chunks: list[GraphEvidence] = []
    notes: dict = {"agentic": True, "truncated_set": False}
    seen_events: set[str] = set()
    seen_evidence: set[str] = set()
    elapsed = 0.0
    primary = retrievals[0]
    for result in retrievals:
        elapsed += result.elapsed_ms
        notes[f"tool:{result.query_name}:status"] = result.status
        if result.is_complete_set:
            notes["complete_set"] = True
            notes["truncated_set"] = False
        for event_id in result.event_ids:
            if event_id not in seen_events:
                seen_events.add(event_id)
                event_ids.append(event_id)
        for collection, bucket in (
            (result.entities, entities),
            (result.facts, facts),
            (result.edges, edges),
            (result.chunks, chunks),
        ):
            for item in collection:
                if item.evidence_id in seen_evidence:
                    continue
                seen_evidence.add(item.evidence_id)
                bucket.append(item)
    if status != "supported":
        notes["answer_suppressed"] = True
    cardinality = "complete_set" if notes.get("complete_set") and status == "supported" else _cardinality(status, event_ids)
    merged = GraphRetrievalResult(
        operation=primary.operation or operation,
        query_name=primary.query_name or operation,
        status=status,
        cardinality=cardinality,
        tool_call_id="agent:merged",
        retrieval_method="agentic+gsql",
        params=dict(primary.params),
        reason=reason,
        event_ids=event_ids,
        entities=entities,
        facts=facts,
        edges=edges,
        chunks=chunks,
        notes=notes,
        elapsed_ms=elapsed,
        spec=primary.spec,
    )
    return merged


def _cardinality(status: str, event_ids: list[str]) -> str:
    if status == "unresolved":
        return "unresolved"
    if status == "ambiguous":
        return "ambiguous"
    if status == "not_found" or not event_ids:
        return "empty"
    if len(event_ids) == 1:
        return "one"
    return "multiple"
