"""Deterministic GraphRAG retrieval over installed OlympicGraph queries."""

from __future__ import annotations

import time
import uuid
from typing import Any, Protocol

from retrieval.graph.client import TigerGraphClient
from retrieval.graph.contract import GraphStore
from retrieval.graph.params import params_for_spec
from retrieval.graph.results import GraphQueryResult
from retrieval.graphrag.models import GraphRetrievalResult
from retrieval.graphrag.normalize import (
    attach_chunks,
    cardinality_for,
    chunk_rows_from_payload,
    empty_result,
    evidence_from_chunks,
    evidence_from_neighborhood_payload,
    flatten_payload,
    normalize_graph_result,
    retrieval_method_for,
)
from retrieval.graphrag.validate import (
    chunk_query_params,
    neighborhood_params,
    validate_event_id,
    validate_spec,
)
from retrieval.structured.models import QuerySpec


class GraphQueryBackend(Protocol):
    def execute_spec(self, spec: QuerySpec, events_by_id: dict | None = None) -> GraphQueryResult: ...

    def run_query(self, name: str, params: dict | None = None) -> Any: ...


class GraphStoreBackend:
    """Offline GraphStore adapter. Spec operations only; no chunk GSQL."""

    source = "local_contract"

    def __init__(self, store: GraphStore) -> None:
        self.store = store

    def execute_spec(self, spec: QuerySpec, events_by_id: dict | None = None) -> GraphQueryResult:
        return self.store.execute(spec)

    def run_query(self, name: str, params: dict | None = None) -> Any:
        raise RuntimeError(f"GraphStoreBackend cannot run installed query {name}")


def _tool_call_id(operation: str) -> str:
    return f"{operation}:{uuid.uuid4().hex[:12]}"


class GraphRetriever:
    """Explicit graph retrieval operations for a future orchestrator tool slot."""

    def __init__(self, backend: GraphQueryBackend) -> None:
        self.backend = backend

    @classmethod
    def from_client(cls, client: TigerGraphClient) -> GraphRetriever:
        return cls(client)

    @classmethod
    def from_store(cls, store: GraphStore) -> GraphRetriever:
        return cls(GraphStoreBackend(store))

    def retrieve(
        self,
        spec: QuerySpec,
        *,
        include_chunks: bool = False,
        max_extra_chunks: int = 0,
    ) -> GraphRetrievalResult:
        tool_call_id = _tool_call_id(spec.operation)
        reason = validate_spec(spec)
        if reason:
            status = "not_found" if reason == "missing_event_title" else "unresolved"
            query_name = spec.operation
            try:
                query_name, _params = params_for_spec(spec)
            except ValueError:
                pass
            return empty_result(
                operation=spec.operation,
                query_name=query_name,
                status=status,
                reason=reason,
                tool_call_id=tool_call_id,
                spec=spec,
            )
        query_name, params = params_for_spec(spec)
        started = time.perf_counter()
        graph = self.backend.execute_spec(spec)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = normalize_graph_result(
            graph,
            query_name=query_name,
            params=params,
            tool_call_id=tool_call_id,
            spec=spec,
            elapsed_ms=elapsed_ms,
        )
        if include_chunks and result.event_ids:
            chunk_result = self.supporting_chunks(
                result.event_ids,
                max_extra=max_extra_chunks,
                tool_call_id=tool_call_id,
            )
            attach_chunks(result, chunk_result.chunks)
            result.notes["chunk_status"] = chunk_result.status
            result.notes["chunk_reason"] = chunk_result.reason
        return result

    def lookup_event(self, title: str, *, include_chunks: bool = False) -> GraphRetrievalResult:
        spec = QuerySpec(
            qtype="lookup",
            operation="lookup_nations",
            raw_question="",
            matched_template=True,
            event_title=title,
            requested_field="nations",
        )
        return self.retrieve(spec, include_chunks=include_chunks)

    def count_over_threshold(
        self,
        sport: str,
        year: int,
        season: str,
        threshold: int,
        *,
        include_chunks: bool = False,
    ) -> GraphRetrievalResult:
        spec = QuerySpec(
            qtype="aggregation",
            operation="count_over_threshold",
            raw_question="",
            matched_template=True,
            sport=sport,
            year=year,
            season=season,
            threshold=threshold,
        )
        return self.retrieve(spec, include_chunks=include_chunks)

    def argmax_competitors(
        self,
        sport: str,
        year: int,
        season: str,
        *,
        include_chunks: bool = False,
    ) -> GraphRetrievalResult:
        spec = QuerySpec(
            qtype="superlative",
            operation="argmax_competitors",
            raw_question="",
            matched_template=True,
            sport=sport,
            year=year,
            season=season,
        )
        return self.retrieve(spec, include_chunks=include_chunks)

    def previous_event_gold(
        self,
        sport: str,
        event_name: str,
        named_year: int,
        season: str,
        relation: str = "previous",
        *,
        include_chunks: bool = False,
    ) -> GraphRetrievalResult:
        operation = "next_event_gold" if relation == "next" else "previous_event_gold"
        spec = QuerySpec(
            qtype="temporal",
            operation=operation,
            raw_question="",
            matched_template=True,
            sport=sport,
            event_name=event_name,
            event_desc=event_name,
            named_year=named_year,
            season=season,
            temporal_relation=relation,
            requested_field="gold",
        )
        return self.retrieve(spec, include_chunks=include_chunks)

    def events_at_venue_date(
        self,
        venue: str,
        date_text: str,
        year: int | None = None,
        *,
        include_chunks: bool = False,
    ) -> GraphRetrievalResult:
        spec = QuerySpec(
            qtype="multi_hop",
            operation="events_at_venue_date",
            raw_question="",
            matched_template=True,
            venue=venue,
            date_text=date_text,
            year=year,
            requested_field="gold",
        )
        return self.retrieve(spec, include_chunks=include_chunks)

    def supporting_chunks(
        self,
        event_ids: list[str],
        *,
        max_extra: int = 0,
        tool_call_id: str | None = None,
    ) -> GraphRetrievalResult:
        operation = "chunks_for_events"
        call_id = tool_call_id or _tool_call_id(operation)
        cleaned = [event_id for event_id in event_ids if event_id]
        if not cleaned:
            return empty_result(
                operation=operation,
                query_name=operation,
                status="not_found",
                reason="missing_event_ids",
                tool_call_id=call_id,
            )
        params = chunk_query_params(cleaned, max_extra=max_extra)
        started = time.perf_counter()
        try:
            raw = self.backend.run_query(operation, params)
        except Exception as exc:
            result = empty_result(
                operation=operation,
                query_name=operation,
                status="unresolved",
                reason="chunk_query_unavailable",
                tool_call_id=call_id,
                params=params,
            )
            result.notes["error"] = exc.__class__.__name__
            return result
        payload = flatten_payload(raw)
        rows = chunk_rows_from_payload(payload)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        chunks = evidence_from_chunks(
            rows,
            tool_call_id=call_id,
            retrieval_method=retrieval_method_for(operation),
        )
        status = "supported" if chunks else "not_found"
        graph = GraphQueryResult(
            operation=operation,
            status=status,
            reason=None if chunks else "no_chunks_for_events",
            event_ids=cleaned,
        )
        result = GraphRetrievalResult(
            operation=operation,
            query_name=operation,
            status=status,
            cardinality="complete_set" if chunks else "empty",
            tool_call_id=call_id,
            retrieval_method=retrieval_method_for(operation),
            params=params,
            reason=graph.reason,
            event_ids=list(cleaned),
            chunks=chunks,
            notes={"requested_event_count": len(cleaned), "returned_chunk_count": len(chunks), "truncated_set": False},
            elapsed_ms=elapsed_ms,
            graph_result=graph,
        )
        return result

    def event_neighborhood(self, event_id: str) -> GraphRetrievalResult:
        operation = "event_neighborhood"
        call_id = _tool_call_id(operation)
        reason = validate_event_id(event_id)
        params = neighborhood_params(event_id)
        if reason:
            return empty_result(
                operation=operation,
                query_name=operation,
                status="unresolved",
                reason=reason,
                tool_call_id=call_id,
                params=params,
            )
        started = time.perf_counter()
        try:
            raw = self.backend.run_query(operation, params)
        except Exception as exc:
            result = empty_result(
                operation=operation,
                query_name=operation,
                status="unresolved",
                reason="neighborhood_query_unavailable",
                tool_call_id=call_id,
                params=params,
            )
            result.notes["error"] = exc.__class__.__name__
            return result
        payload = flatten_payload(raw)
        entities, facts, edges, chunks = evidence_from_neighborhood_payload(payload, tool_call_id=call_id)
        event_ids = [item.event_id for item in entities if item.event_id]
        status = "supported" if event_ids else "not_found"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        graph = GraphQueryResult(operation=operation, status=status, event_ids=event_ids)
        result = GraphRetrievalResult(
            operation=operation,
            query_name=operation,
            status=status,
            cardinality=cardinality_for(graph),
            tool_call_id=call_id,
            retrieval_method=retrieval_method_for(operation),
            params=params,
            reason=None if event_ids else "event_id_not_found",
            event_ids=event_ids,
            entities=entities,
            facts=facts,
            edges=edges,
            chunks=chunks,
            notes={"hop_edge_count": len(edges), "supporting_chunk_count": len(chunks)},
            elapsed_ms=elapsed_ms,
            graph_result=graph,
        )
        if not event_ids:
            result.notes["answer_suppressed"] = True
        return result


def ensure_retrieval_queries(client: TigerGraphClient) -> dict[str, Any]:
    """Install Phase 5 helpers if missing. Never recreates the graph."""
    report = client.verify_retrieval_queries()
    if report.get("ok"):
        return report
    created = client.install_retrieval_queries()
    report = client.verify_retrieval_queries()
    report["install_output"] = created[:2000]
    return report
