"""Provenance-preserving context packing for shared generators."""

from __future__ import annotations

import re

from answering.models import (
    Citation,
    CitationValidation,
    GeneratorResult,
    PackedContext,
    PackedContextItem,
)
from retrieval.graphrag.models import GraphEvidence, GraphRetrievalResult
from retrieval.rag.models import RetrievalHit, RetrievalResult


class ContextPacker:
    """Convert retrieval records to compact items without losing provenance."""

    def pack_graph(self, result: GraphRetrievalResult) -> PackedContext:
        ordered = [
            *result.entities,
            *result.facts,
            *result.edges,
            *result.chunks,
        ]
        items = self._deduplicate(self._from_graph(item) for item in ordered)
        return PackedContext(
            items=items,
            retrieval_status=result.status,
            cardinality=result.cardinality,
            answer_suppressed=bool(result.notes.get("answer_suppressed")),
            complete_set=result.is_complete_set,
            notes={
                "operation": result.operation,
                "query_name": result.query_name,
                "event_ids": list(result.event_ids),
                "retrieval_notes": dict(result.notes),
            },
        )

    def pack_text(self, result: RetrievalResult) -> PackedContext:
        items = self._deduplicate(self._from_hit(hit, result.method) for hit in result.hits)
        cardinality = "empty"
        if len(items) == 1:
            cardinality = "one"
        elif len(items) > 1:
            cardinality = "multiple"
        return PackedContext(
            items=items,
            retrieval_status="supported" if items else "not_found",
            cardinality=cardinality,
            answer_suppressed=not items,
            complete_set=False,
            notes={
                "query": result.query,
                "method": result.method,
                "params": dict(result.params),
                "retrieval_elapsed_ms": result.elapsed_ms,
                "hit_count": len(items),
                "multi_hit": len(items) > 1,
                "text_conflict_not_detected": len(items) > 1,
            },
        )

    @staticmethod
    def _from_graph(item: GraphEvidence) -> PackedContextItem:
        return PackedContextItem(
            evidence_id=item.evidence_id,
            evidence_type=item.evidence_type,
            retrieval_method=item.retrieval_method,
            tool_call_id=item.tool_call_id,
            graph_refs=tuple(item.graph_refs),
            event_id=item.event_id,
            document_id=item.document_id,
            chunk_id=item.chunk_id,
            source_chunk_id=item.source_chunk_id,
            field_name=item.field_name,
            value=item.value,
            text=item.text,
            retrieval_score=item.retrieval_score,
            source_url=item.source_url,
            source_date=item.source_date,
            source_version=item.source_version,
            observation_truncated=item.observation_truncated,
            why_retrieved=item.why_retrieved,
        )

    @staticmethod
    def _from_hit(hit: RetrievalHit, method: str) -> PackedContextItem:
        return PackedContextItem(
            evidence_id=f"rag:{method}:{hit.chunk_id}",
            evidence_type="chunk",
            retrieval_method=hit.retrieval_method,
            tool_call_id=f"rag:{method}",
            event_id=hit.event_id,
            document_id=hit.document_id,
            chunk_id=hit.chunk_id,
            source_chunk_id=hit.chunk_id,
            text=hit.text,
            retrieval_score=hit.score,
            source_url=hit.source_url,
            why_retrieved=f"text retrieval rank {hit.rank}",
        )

    @staticmethod
    def _deduplicate(items) -> tuple[PackedContextItem, ...]:
        unique: list[PackedContextItem] = []
        seen: set[str] = set()
        for item in items:
            if item.evidence_id in seen:
                continue
            seen.add(item.evidence_id)
            unique.append(item)
        return tuple(unique)


def validate_generator_citations(
    context: PackedContext,
    generated: GeneratorResult,
) -> CitationValidation:
    """Resolve citations against context and reject unsupported claims."""

    by_id = context.item_by_id()
    errors: list[str] = []
    citation_ids: list[str] = []
    seen: set[str] = set()
    for evidence_id in generated.citation_ids:
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        citation_ids.append(evidence_id)
        if evidence_id not in by_id:
            errors.append(f"orphan_citation:{evidence_id}")
    if generated.answer_text.strip() and not citation_ids:
        errors.append("missing_citations")
    used = tuple(by_id[evidence_id] for evidence_id in citation_ids if evidence_id in by_id)
    if generated.answer_text.strip() and used and not answer_grounded_in_items(generated.answer_text, used):
        errors.append("ungrounded_answer")
    if errors:
        return CitationValidation(valid=False, errors=tuple(errors))
    chunk_sources = {
        item.chunk_id: item
        for item in context.items
        if item.evidence_type == "chunk" and item.chunk_id is not None
    }
    citations = tuple(
        _citation_with_support(item, chunk_sources.get(item.source_chunk_id))
        for item in used
    )
    return CitationValidation(
        valid=True,
        citations=citations,
        evidence_used=used,
    )


def answer_grounded_in_items(answer: str, items: tuple[PackedContextItem, ...]) -> bool:
    """True when the answer string is extractable from cited item values/text."""

    if not answer.strip():
        return True
    if not items:
        return False
    if any(span_in_item(answer, item) for item in items):
        return True
    parts = [part.strip() for part in answer.split(" | ") if part.strip()]
    if len(parts) > 1 and all(any(span_in_item(part, item) for item in items) for part in parts):
        return True
    return False


def span_in_item(needle: str, item: PackedContextItem) -> bool:
    return span_grounded_in_text(needle, _item_haystack(item))


_NUMERIC_ANSWER = re.compile(r"^[0-9]+(?:[.,][0-9]+)?$")


def span_grounded_in_text(needle: str, haystack: str) -> bool:
    """Boundary-aware extractive match. Shared by generator and citation validator."""

    needle = (needle or "").strip()
    haystack = haystack or ""
    if not needle or not haystack.strip():
        return False
    if needle.casefold() == haystack.strip().casefold():
        return True
    if _NUMERIC_ANSWER.fullmatch(needle):
        pattern = rf"(?<![0-9]){re.escape(needle)}(?![0-9])"
        return re.search(pattern, haystack) is not None
    pattern = rf"(?<!\w){re.escape(needle)}(?!\w)"
    for match in re.finditer(pattern, haystack, flags=re.IGNORECASE):
        if re.match(r"\s*:", haystack[match.end() :]):
            continue
        return True
    return False


def _item_haystack(item: PackedContextItem) -> str:
    parts: list[str] = []
    if item.value is not None:
        parts.append(str(item.value))
    if item.text:
        parts.append(item.text)
    return "\n".join(parts)


def _citation_with_support(
    item: PackedContextItem,
    supporting_chunk: PackedContextItem | None,
) -> Citation:
    return Citation(
        evidence_id=item.evidence_id,
        source_url=item.source_url
        or (supporting_chunk.source_url if supporting_chunk is not None else None),
        source_date=item.source_date
        or (supporting_chunk.source_date if supporting_chunk is not None else None),
        document_id=item.document_id
        or (supporting_chunk.document_id if supporting_chunk is not None else None),
        chunk_id=item.chunk_id or item.source_chunk_id,
        event_id=item.event_id
        or (supporting_chunk.event_id if supporting_chunk is not None else None),
    )
