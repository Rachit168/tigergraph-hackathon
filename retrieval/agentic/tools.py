"""Typed agent tools that wrap GraphRetriever. No second TigerGraph client."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from retrieval.graphrag.models import GraphRetrievalResult
from retrieval.graphrag.retriever import GraphRetriever
from retrieval.structured.models import QuerySpec

RETRIEVE_SPEC = "retrieve_spec"
SUPPORTING_CHUNKS = "supporting_chunks"
EVENT_NEIGHBORHOOD = "event_neighborhood"


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
    """Adapter from planner actions onto existing GraphRetriever operations."""

    def __init__(self, retriever: GraphRetriever) -> None:
        self.retriever = retriever

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
        raise ValueError(f"unknown tool: {action.tool}")
