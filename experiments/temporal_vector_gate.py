"""Narrow temporal V0 vs V2 experiment helpers. Does not change production retrieval."""

from __future__ import annotations

import re
from typing import Any

from ingestion.models import ParsedDocument
from retrieval.rag.models import RetrievalHit, RetrievalResult

GOLD_QTYPES_TEMPORAL = "temporal"
NAMED_YEAR_PATTERNS = (
    re.compile(r"immediately before\s+(\d{4})", re.I),
    re.compile(r"previous Olympiad before\s+(\d{4})", re.I),
    re.compile(r"prior edition of(?: the)?\s+(\d{4})", re.I),
    re.compile(r"prior to(?: the)?\s+(\d{4})", re.I),
    re.compile(r"preceding(?:\s+the)?\s+(\d{4})", re.I),
    re.compile(r"Games before\s+(\d{4})", re.I),
)


class RecordingRetriever:
    """Experiment-only wrapper. Forwards retrieve() and stores the last result."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.last_result: RetrievalResult | None = None

    def retrieve(self, *args: Any, **kwargs: Any) -> RetrievalResult:
        result = self.inner.retrieve(*args, **kwargs)
        self.last_result = result
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def parse_named_year(question: str) -> int | None:
    text = question or ""
    for pattern in NAMED_YEAR_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def document_year(document: ParsedDocument | None) -> int | None:
    if document is None:
        return None
    if document.event is not None and document.event.year is not None:
        return int(document.event.year)
    if document.title_parse is not None and document.title_parse.year is not None:
        return int(document.title_parse.year)
    return None


def previous_olympiad_gold(
    gold_doc_ids: list[str],
    named_year: int | None,
    documents_by_id: dict[str, ParsedDocument],
) -> dict[str, Any]:
    """Identify the gold document for the previous Games vs the named year."""
    gold = [str(doc_id) for doc_id in gold_doc_ids]
    years = {doc_id: document_year(documents_by_id.get(doc_id)) for doc_id in gold}
    if named_year is None:
        return {
            "named_year": None,
            "previous_olympiad_doc_id": None,
            "named_year_doc_id": None,
            "gold_years": years,
            "identifiable": False,
            "reason": "named_year_not_parsed",
        }
    earlier = [doc_id for doc_id, year in years.items() if year is not None and year < named_year]
    same = [doc_id for doc_id, year in years.items() if year == named_year]
    previous_id = None
    if earlier:
        previous_id = max(earlier, key=lambda doc_id: years[doc_id] or 0)
    return {
        "named_year": named_year,
        "previous_olympiad_doc_id": previous_id,
        "named_year_doc_id": same[0] if len(same) == 1 else (same[0] if same else None),
        "gold_years": years,
        "identifiable": previous_id is not None,
        "reason": None if previous_id else "no_gold_doc_older_than_named_year",
    }


def hits_payload(result: RetrievalResult | None, *, include_scores: bool = False) -> list[dict[str, Any]]:
    if result is None:
        return []
    rows: list[dict[str, Any]] = []
    for hit in result.hits:
        if not isinstance(hit, RetrievalHit):
            continue
        item: dict[str, Any] = {
            "chunk_id": hit.chunk_id,
            "document_id": hit.document_id,
            "rank": hit.rank,
            "retrieval_method": hit.retrieval_method,
        }
        if include_scores:
            item["score"] = hit.score
        rows.append(item)
    return rows


def packed_chunk_ids(evidence_metadata: list[dict[str, Any]] | None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in evidence_metadata or []:
        chunk_id = item.get("chunk_id") or item.get("source_chunk_id")
        if not chunk_id:
            continue
        text = str(chunk_id)
        if text in seen:
            continue
        seen.add(text)
        ids.append(text)
    return ids


def packed_document_ids(evidence_metadata: list[dict[str, Any]] | None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in evidence_metadata or []:
        doc_id = item.get("document_id") or item.get("event_id")
        if not doc_id:
            continue
        text = str(doc_id)
        if text in seen:
            continue
        seen.add(text)
        ids.append(text)
    return ids


def previous_doc_entered_packed(previous_doc_id: str | None, packed_docs: list[str]) -> bool | None:
    if not previous_doc_id:
        return None
    return previous_doc_id in set(packed_docs)
