"""Typed agent tools that wrap GraphRetriever. No second TigerGraph client."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from retrieval.graphrag.models import GraphEvidence, GraphRetrievalResult
from retrieval.graph.vector import SEARCH_QUERY
from retrieval.rag.models import RetrievalResult
from retrieval.graphrag.retriever import GraphRetriever
from retrieval.structured.models import QuerySpec

RETRIEVE_SPEC = "retrieve_spec"
SUPPORTING_CHUNKS = "supporting_chunks"
EVENT_NEIGHBORHOOD = "event_neighborhood"
VECTOR_SEARCH = "vector_search"
VECTOR_TOP_K = 5


@dataclass(frozen=True)
class ToolAction:
    tool: str
    arguments: dict[str, Any]
    reason: str
    parallel_group: str | None = None

    def fingerprint(self) -> str:
        return json.dumps(
            {"tool": self.tool, "arguments": self.arguments},
            sort_keys=True,
            default=str,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "reason": self.reason,
            "parallel_group": self.parallel_group,
            "fingerprint": self.fingerprint(),
        }


@dataclass
class ToolObservation:
    tool_call_id: str
    tool: str
    arguments: dict[str, Any]
    reason: str
    success: bool
    started_ms: float
    elapsed_ms: float
    fingerprint: str
    result: GraphRetrievalResult | None = None
    error: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    retrieval_method: str | None = None
    observation_truncated: bool = False
    parallel_group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "reason": self.reason,
            "success": self.success,
            "started_ms": self.started_ms,
            "elapsed_ms": self.elapsed_ms,
            "fingerprint": self.fingerprint,
            "error": self.error,
            "evidence_ids": list(self.evidence_ids),
            "retrieval_method": self.retrieval_method,
            "observation_truncated": self.observation_truncated,
            "parallel_group": self.parallel_group,
            "result_status": None if self.result is None else self.result.status,
            "event_ids": [] if self.result is None else list(self.result.event_ids),
        }


class GraphRetrieverTools:
    """Adapter from planner actions onto graph and optional vector operations."""

    def __init__(self, retriever: GraphRetriever, vector_retriever: Any | None = None) -> None:
        self.retriever = retriever
        self.vector_retriever = vector_retriever

    def run(self, action: ToolAction, spec: QuerySpec) -> GraphRetrievalResult:
        if action.tool == RETRIEVE_SPEC:
            include_chunks = bool(action.arguments.get("include_chunks", False))
            return self.retriever.retrieve(spec, include_chunks=include_chunks, max_extra_chunks=0)
        if action.tool == SUPPORTING_CHUNKS:
            event_ids = [str(value) for value in action.arguments.get("event_ids") or []]
            max_extra = int(action.arguments.get("max_extra") or 0)
            return self.retriever.supporting_chunks(event_ids, max_extra=max_extra)
        if action.tool == EVENT_NEIGHBORHOOD:
            event_id = str(action.arguments.get("event_id") or "")
            return self.retriever.event_neighborhood(event_id)
        if action.tool == VECTOR_SEARCH:
            if self.vector_retriever is None:
                raise RuntimeError("vector_search_unavailable")
            query = str(action.arguments.get("query") or spec.raw_question or "").strip()
            top_k = _bounded_vector_top_k(action.arguments.get("top_k"))
            result = self.vector_retriever.retrieve(
                query,
                top_k=top_k,
                method="tigergraph_vector",
            )
            return _vector_result(result)
        raise ValueError(f"unknown tool: {action.tool}")


def _bounded_vector_top_k(value: Any) -> int:
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = VECTOR_TOP_K
    return max(1, min(VECTOR_TOP_K, requested))


def _vector_result(result: RetrievalResult) -> GraphRetrievalResult:
    """Adapt existing TigerGraph vector hits into graph evidence."""

    tool_call_id = f"{VECTOR_SEARCH}:{uuid.uuid4().hex[:12]}"
    chunks = [
        GraphEvidence(
            evidence_id=f"{tool_call_id}:chunk:{hit.chunk_id}",
            evidence_type="chunk",
            retrieval_method="tigergraph_vector",
            tool_call_id=tool_call_id,
            event_id=hit.event_id,
            document_id=hit.document_id,
            chunk_id=hit.chunk_id,
            source_chunk_id=hit.chunk_id,
            text=hit.text,
            retrieval_score=hit.score,
            source_url=hit.source_url,
            why_retrieved="bounded TigerGraph vector fallback",
        )
        for hit in result.hits
    ]
    event_ids: list[str] = []
    for hit in result.hits:
        if hit.event_id and hit.event_id not in event_ids:
            event_ids.append(hit.event_id)
    if not chunks:
        status = "not_found"
        cardinality = "empty"
        reason = "vector_no_hits"
    else:
        status = "supported"
        cardinality = "one" if len(chunks) == 1 else "multiple"
        reason = None
    params = dict(result.params)
    params["top_k"] = _bounded_vector_top_k(params.get("top_k"))
    return GraphRetrievalResult(
        operation=VECTOR_SEARCH,
        query_name=SEARCH_QUERY,
        status=status,
        cardinality=cardinality,
        tool_call_id=tool_call_id,
        retrieval_method="tigergraph_vector",
        params=params,
        reason=reason,
        event_ids=event_ids,
        chunks=chunks,
        notes={
            "vector_fallback": True,
            "hit_count": len(chunks),
            "truncated_set": False,
        },
        elapsed_ms=result.elapsed_ms,
    )
