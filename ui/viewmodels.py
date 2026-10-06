"""Presentation view-models. Frontend must not import retrieval dataclasses."""

from __future__ import annotations

from typing import Any

from evaluation.trace_export import EXCLUDED_KEYS, SCHEMA_VERSION
from ui.catalog import (
    PIPELINE_LABELS,
    RUNTIME_MODES,
    STATUS_LABELS,
    STOP_EXPLANATIONS,
    SYSTEM_STATUS,
    VECTOR_EXPERIMENT,
    WHY_AGENTIC,
    canonical_benchmark,
)

EDGE_TYPES = {"HELD_AT": "Venue", "IN_GAMES": "Games", "OF_SPORT": "Sport"}
MAX_GRAPH_NODES = 64
MAX_GRAPH_EDGES = 96
FACT_KINDS = {
    "gold_raw": "athlete",
    "nations": "fact",
    "competitors": "fact",
    "count": "fact",
    "year": "fact",
    "date_raw": "fact",
    "prev_year": "fact",
}


def investigation_from_export(record: dict[str, Any] | None) -> dict[str, Any]:
    """Map a schema_version=1 export record into an InvestigationViewModel dict."""

    source = dict(record or {})
    pipeline = str(source.get("pipeline") or "unknown")
    agent_trace = source.get("agent_trace") if isinstance(source.get("agent_trace"), dict) else None
    citations = [item for item in (source.get("citations") or []) if isinstance(item, dict)]
    evidence = evidence_items_from_export(source)
    timings = source.get("timings") if isinstance(source.get("timings"), dict) else {}
    status = str(source.get("status") or "")
    failure_class = source.get("failure_class")
    stop_reason = None
    if agent_trace:
        raw_stop = agent_trace.get("stop_reason")
        stop_reason = str(raw_stop) if raw_stop else None
    steps = _steps(source, agent_trace, timings, evidence)
    evidence_views = [_evidence_view(item, citations) for item in evidence]
    graph = graph_from_evidence(evidence, citations)
    agent = _agent_panel(agent_trace) if pipeline == "agentic_graphrag" else None
    attempts = source.get("attempts") if isinstance(source.get("attempts"), list) else []
    mode = runtime_mode(status, recorded=False, failure_class=failure_class)
    payload = {
        "question": str(source.get("question") or ""),
        "question_id": source.get("question_id"),
        "pipeline": pipeline,
        "pipeline_label": PIPELINE_LABELS.get(pipeline, pipeline),
        "status": status,
        "status_label": status_label(status, failure_class),
        "runtime_mode": mode,
        "runtime_mode_label": RUNTIME_MODES.get(mode, mode.upper() if mode else "—"),
        "failure_class": failure_class,
        "answer": str(source.get("answer") or ""),
        "answer_display": "",
        "citations": [_citation_view(item) for item in citations],
        "tokens": _int(source.get("total_tokens")),
        "tokens_unknown": bool(source.get("tokens_unknown")),
        "model_calls": _int(source.get("model_calls")),
        "latency_ms": _float(source.get("latency_ms")),
        "timings": dict(timings),
        "steps": steps,
        "evidence": evidence_views,
        "graph": graph,
        "stop_reason": stop_reason,
        "stop_explanation": stop_explanation(stop_reason, pipeline, status, failure_class),
        "continue_explanation": continue_explanation(agent_trace, pipeline),
        "why_continued": why_continued(agent_trace),
        "evidence_diff": evidence_diff(agent_trace, evidence_views),
        "lineage": evidence_lineage(citations, evidence_views),
        "evidence_status": evidence_status(evidence_views, citations, graph, source),
        "efficiency": efficiency_panel(source, agent, steps, timings),
        "reliability": reliability_panel(source, status, failure_class, attempts),
        "agent": agent,
        "retrieval_methods": list(source.get("retrieval_methods") or []),
        "warnings": list(source.get("warnings") or []),
        "errors": list(source.get("errors") or []),
        "schema_version": int(source.get("schema_version") or SCHEMA_VERSION),
        "recorded": False,
        "recorded_label": "",
        "export_record": source if source.get("schema_version") else None,
        "evidence_ids": list(source.get("evidence_ids") or []),
        "chunk_ids": list(source.get("chunk_ids") or []),
        "document_ids": list(source.get("document_ids") or []),
        "event_ids": list(source.get("event_ids") or []),
        "attempts": list(attempts),
    }
    return refresh_status_fields(payload)


def evidence_items_from_export(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer packed evidence objects; otherwise rebuild rows from exported IDs."""

    packed = [item for item in (source.get("evidence") or []) if isinstance(item, dict)]
    if packed:
        return packed
    citations = [item for item in (source.get("citations") or []) if isinstance(item, dict)]
    event_ids = [str(item) for item in (source.get("event_ids") or []) if item]
    document_ids = [str(item) for item in (source.get("document_ids") or []) if item]
    chunk_ids = [str(item) for item in (source.get("chunk_ids") or []) if item]
    methods = list(source.get("retrieval_methods") or [])
    method = methods[0] if methods else None
    prefix_events: dict[str, str] = {}
    items: list[dict[str, Any]] = []
    for evidence_id in source.get("evidence_ids") or []:
        parsed = _parse_evidence_id(str(evidence_id), event_ids=event_ids, document_ids=document_ids)
        parsed["retrieval_method"] = parsed.get("retrieval_method") or method
        cite = next((row for row in citations if row.get("evidence_id") == evidence_id), None)
        if cite:
            parsed["document_id"] = parsed.get("document_id") or cite.get("document_id")
            parsed["chunk_id"] = parsed.get("chunk_id") or cite.get("chunk_id")
            parsed["event_id"] = parsed.get("event_id") or cite.get("event_id")
            parsed["source_url"] = cite.get("source_url")
            if parsed.get("evidence_type") == "fact" and source.get("answer"):
                parsed["value"] = source.get("answer")
        if parsed.get("evidence_type") == "entity" and parsed.get("event_id") and parsed.get("call"):
            prefix_events[str(parsed["call"])] = str(parsed["event_id"])
        items.append(parsed)
    for item in items:
        if item.get("evidence_type") in {"edge", "fact"} and not item.get("event_id"):
            item["event_id"] = prefix_events.get(str(item.get("call") or "")) or (event_ids[0] if event_ids else None)
        if item.get("evidence_type") == "chunk" and not item.get("event_id"):
            chunk_id = str(item.get("chunk_id") or "")
            for event_id in event_ids:
                if chunk_id.startswith(event_id):
                    item["event_id"] = event_id
                    break
        if item.get("evidence_type") == "chunk" and not item.get("document_id"):
            if item.get("event_id") and item["event_id"] in document_ids:
                item["document_id"] = item["event_id"]
            elif document_ids:
                item["document_id"] = document_ids[0]
        if item.get("evidence_type") == "chunk" and not item.get("chunk_id") and chunk_ids:
            item["chunk_id"] = chunk_ids[0]
    return items


def _parse_evidence_id(
    evidence_id: str,
    *,
    event_ids: list[str],
    document_ids: list[str],
) -> dict[str, Any]:
    parts = evidence_id.split(":")
    item: dict[str, Any] = {
        "evidence_id": evidence_id,
        "evidence_type": "unknown",
        "graph_refs": [],
        "why_retrieved": None,
        "text": None,
        "value": None,
        "field_name": None,
        "event_id": None,
        "document_id": None,
        "chunk_id": None,
        "call": parts[0] if parts else None,
    }
    if len(parts) >= 3 and parts[1] == "entity":
        event_id = ":".join(parts[2:])
        item.update({"evidence_type": "entity", "event_id": event_id, "value": event_id, "field_name": "title"})
        item["graph_refs"] = [{"vertex_type": "Event", "vertex_id": event_id}]
        return item
    if len(parts) >= 4 and parts[1] == "fact":
        field = parts[2]
        event_id = ":".join(parts[3:])
        item.update({"evidence_type": "fact", "field_name": field, "event_id": event_id, "value": field})
        item["graph_refs"] = [{"vertex_type": "Event", "vertex_id": event_id, "attribute": field}]
        return item
    if len(parts) >= 4 and parts[1] == "edge":
        edge_type = parts[2]
        neighbor = ":".join(parts[3:])
        vertex_type = EDGE_TYPES.get(edge_type, "entity")
        item.update({"evidence_type": "edge", "field_name": edge_type, "value": neighbor})
        item["graph_refs"] = [{"vertex_type": vertex_type, "vertex_id": neighbor, "edge_type": edge_type}]
        return item
    if len(parts) >= 3 and parts[1] == "chunk":
        chunk_id = ":".join(parts[2:])
        document_id = chunk_id.split("::")[0] if "::" in chunk_id else (document_ids[0] if document_ids else None)
        item.update({"evidence_type": "chunk", "chunk_id": chunk_id, "document_id": document_id})
        return item
    if evidence_id in event_ids:
        item.update({"evidence_type": "entity", "event_id": evidence_id, "value": evidence_id})
        return item
    if evidence_id in document_ids:
        item.update({"evidence_type": "document", "document_id": evidence_id})
        return item
    return item


def graph_from_evidence(
    evidence: list[dict[str, Any]],
    citations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build nodes/edges only from returned evidence. No invented graph hops."""

    cited = {str(item.get("evidence_id") or "") for item in citations or [] if item.get("evidence_id")}
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(node_id: str, kind: str, label: str, **extra: Any) -> None:
        if not node_id:
            return
        existing = nodes.get(node_id)
        payload = {"id": node_id, "kind": kind, "label": label or node_id, **extra}
        if existing is None:
            nodes[node_id] = payload
            return
        if extra.get("cited"):
            existing["cited"] = True
        if label and existing["label"] == existing["id"]:
            existing["label"] = label

    def add_edge(source: str, target: str, label: str, evidence_id: str | None = None) -> None:
        if not source or not target or source == target:
            return
        key = (source, target, label)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append(
            {
                "source": source,
                "target": target,
                "label": label,
                "evidence_id": evidence_id,
            }
        )

    for item in evidence:
        evidence_id = str(item.get("evidence_id") or "")
        evidence_type = str(item.get("evidence_type") or "")
        event_id = _qid(item.get("event_id"))
        document_id = str(item.get("document_id") or "").strip()
        chunk_id = str(item.get("chunk_id") or item.get("source_chunk_id") or "").strip()
        cited_here = evidence_id in cited
        if evidence_type == "entity" and event_id:
            add_node(
                f"event:{event_id}",
                "event",
                str(item.get("value") or event_id),
                event_id=event_id,
                evidence_id=evidence_id,
                cited=cited_here,
            )
        elif evidence_type == "fact":
            field = str(item.get("field_name") or "fact")
            value = str(item.get("value") or "")
            kind = FACT_KINDS.get(field, "fact")
            fact_id = f"fact:{evidence_id or field}:{value}"
            add_node(fact_id, kind, value or field, field=field, evidence_id=evidence_id, cited=cited_here)
            if event_id:
                add_node(f"event:{event_id}", "event", event_id, event_id=event_id)
                add_edge(f"event:{event_id}", fact_id, field, evidence_id)
        elif evidence_type == "edge":
            edge_type = str(item.get("field_name") or item.get("value") or "related")
            neighbor = str(item.get("value") or "")
            neighbor_kind = EDGE_TYPES.get(edge_type, "entity")
            neighbor_id = f"{neighbor_kind.lower()}:{neighbor}"
            add_node(neighbor_id, neighbor_kind.lower(), neighbor.replace("_", " "), evidence_id=evidence_id, cited=cited_here)
            if event_id:
                add_node(f"event:{event_id}", "event", event_id, event_id=event_id)
                add_edge(f"event:{event_id}", neighbor_id, edge_type, evidence_id)
            for ref in item.get("graph_refs") or []:
                if not isinstance(ref, dict):
                    continue
                vertex_type = str(ref.get("vertex_type") or "")
                vertex_id = str(ref.get("vertex_id") or "")
                if vertex_type and vertex_id and vertex_type != "Event":
                    add_node(f"{vertex_type.lower()}:{vertex_id}", vertex_type.lower(), vertex_id.replace("_", " "))
        elif evidence_type == "document" and document_id:
            add_node(f"document:{document_id}", "document", document_id, evidence_id=evidence_id, cited=cited_here)
        elif evidence_type == "chunk":
            if chunk_id:
                add_node(
                    f"chunk:{chunk_id}",
                    "chunk",
                    chunk_id,
                    evidence_id=evidence_id,
                    cited=cited_here,
                    text=item.get("text"),
                )
            if document_id:
                add_node(f"document:{document_id}", "document", document_id, evidence_id=evidence_id, cited=cited_here)
                if chunk_id:
                    add_edge(f"document:{document_id}", f"chunk:{chunk_id}", "CONTAINS_CHUNK", evidence_id)
            if event_id and document_id:
                add_node(f"event:{event_id}", "event", event_id, event_id=event_id)
                add_edge(f"document:{document_id}", f"event:{event_id}", "DESCRIBES", evidence_id)
            elif event_id and chunk_id:
                add_node(f"event:{event_id}", "event", event_id, event_id=event_id)
                add_edge(f"chunk:{chunk_id}", f"event:{event_id}", "event_id", evidence_id)

    node_list = list(nodes.values())
    truncated = len(node_list) > MAX_GRAPH_NODES or len(edges) > MAX_GRAPH_EDGES
    node_list = node_list[:MAX_GRAPH_NODES]
    kept = {node["id"] for node in node_list}
    edge_list = [edge for edge in edges if edge["source"] in kept and edge["target"] in kept][:MAX_GRAPH_EDGES]
    return {
        "nodes": node_list,
        "edges": edge_list,
        "cited_evidence_ids": sorted(eid for eid in cited if eid),
        "truncated": truncated,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def benchmark_view() -> dict[str, Any]:
    payload = canonical_benchmark()
    payload["headline"] = {
        key: value["correctness"] for key, value in payload["pipelines"].items()
    }
    return payload


def system_view() -> dict[str, Any]:
    return {
        "graph_status": dict(SYSTEM_STATUS["tigergraph"]),
        "vector_status": dict(SYSTEM_STATUS["vector"]),
        "production_routing": dict(SYSTEM_STATUS["routing"]),
        "architecture": _architecture(),
        "why_agentic": dict(WHY_AGENTIC),
        "vector_experiment": dict(VECTOR_EXPERIMENT),
        "label": SYSTEM_STATUS["label"],
        "production_note": SYSTEM_STATUS["production_note"],
    }


def empty_investigation(*, question: str = "", pipeline: str = "rag") -> dict[str, Any]:
    view = investigation_from_export(
        {
            "schema_version": SCHEMA_VERSION,
            "question": question,
            "pipeline": pipeline,
            "answer": "",
            "status": "",
            "citations": [],
            "evidence": [],
            "agent_trace": None,
            "stop_reason": None,
            "total_tokens": 0,
            "model_calls": 0,
            "latency_ms": 0,
            "timings": {},
            "retrieval_methods": [],
        }
    )
    view["export_record"] = None
    return view


def status_label(status: str, failure_class: Any = None) -> str:
    failure = str(failure_class or "").strip().casefold()
    if failure == "timeout":
        return STATUS_LABELS["timeout"]
    if failure == "malformed_output":
        return STATUS_LABELS["generation_error"]
    key = str(status or "").strip().casefold()
    if key in STATUS_LABELS:
        return STATUS_LABELS[key]
    return str(status or "").replace("_", " ").upper() or "—"


def stop_explanation(
    stop_reason: str | None,
    pipeline: str,
    status: str | None,
    failure_class: Any = None,
) -> str:
    key = str(status or "").strip().casefold()
    fail = str(stop_reason or "").strip().casefold()
    if fail == "malformed_output" or str(failure_class or "").strip().casefold() == "malformed_output":
        return "Investigation stopped because generation returned malformed output."
    if key in {"unavailable", "error", "timeout", "generation_error"}:
        return STOP_EXPLANATIONS.get(key, "Investigation stopped because the backend failed.")
    if fail == "timeout" or key == "timeout":
        return STOP_EXPLANATIONS["timeout"]
    if pipeline == "agentic_graphrag" and stop_reason:
        return STOP_EXPLANATIONS.get(str(stop_reason), f"Investigation stopped because `{stop_reason}`.")
    if not key:
        return ""
    if pipeline == "rag":
        return "RAG finished after BM25 retrieval, packing, and shared generation."
    if pipeline == "graphrag":
        return "Fixed GraphRAG finished after one typed retrieval and shared generation."
    return f"Pipeline finished with status `{status}`."


def display_answer(view: dict[str, Any]) -> str:
    failure = str(view.get("failure_class") or "").strip().casefold()
    if failure == "malformed_output":
        return "No answer was produced because generation returned malformed output."
    answer = str(view.get("answer") or "").strip()
    if answer:
        return answer
    status = str(view.get("status") or "").strip().casefold()
    if status == "unavailable":
        return "No answer was produced because this pipeline is unavailable in the current runtime."
    if status == "error":
        return "No answer was produced because the backend failed."
    if status == "timeout":
        return "No answer was produced because the request timed out."
    if status == "generation_error":
        return "No answer was produced because generation failed."
    if status:
        return "The pipeline finished without answer text."
    return ""


def refresh_status_fields(view: dict[str, Any]) -> dict[str, Any]:
    status = str(view.get("status") or "")
    failure = view.get("failure_class")
    pipeline = str(view.get("pipeline") or "")
    mode = runtime_mode(status, recorded=bool(view.get("recorded")), failure_class=failure)
    view["status_label"] = status_label(status, failure)
    view["runtime_mode"] = mode
    view["runtime_mode_label"] = RUNTIME_MODES.get(mode, mode.upper() if mode else "—")
    view["stop_explanation"] = stop_explanation(
        view.get("stop_reason"),
        pipeline,
        status,
        failure,
    )
    view["answer_display"] = display_answer(view)
    view["reliability"] = reliability_panel(view, status, failure, list(view.get("attempts") or []))
    return view


def runtime_mode(status: str, *, recorded: bool = False, failure_class: Any = None) -> str:
    if recorded:
        return "preview"
    key = str(status or "").strip().casefold()
    if key == "unavailable":
        return "unavailable"
    failure = str(failure_class or "").strip().casefold()
    if failure in {"timeout", "malformed_output"} or key in {"error", "generation_error", "timeout"}:
        return "error"
    if key:
        return "live"
    return ""


def continue_explanation(trace: dict[str, Any] | None, pipeline: str) -> str:
    if pipeline != "agentic_graphrag" or not isinstance(trace, dict):
        return ""
    follow_ups = [str(item) for item in (trace.get("follow_up_decisions") or []) if item]
    continue_reasons = [item for item in follow_ups if not item.startswith("stop:")]
    if not continue_reasons:
        plan_follow = [
            str(step.get("reason") or "")
            for step in (trace.get("plan_steps") or [])
            if isinstance(step, dict) and str(step.get("kind") or "") == "follow_up"
        ]
        continue_reasons = [item for item in plan_follow if item]
    if not continue_reasons:
        return ""
    return _judge_continue_text(continue_reasons[0])


def why_continued(trace: dict[str, Any] | None) -> list[str]:
    if not isinstance(trace, dict):
        return []
    lines: list[str] = []
    for step in trace.get("plan_steps") or []:
        if not isinstance(step, dict) or str(step.get("kind") or "") != "follow_up":
            continue
        reason = str(step.get("reason") or "").strip()
        if reason:
            lines.append(_judge_continue_text(reason))
    for decision in trace.get("follow_up_decisions") or []:
        text = str(decision or "").strip()
        if text and not text.startswith("stop:"):
            mapped = _judge_continue_text(text)
            if mapped not in lines:
                lines.append(mapped)
    return lines


def evidence_diff(trace: dict[str, Any] | None, evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(trace, dict):
        return None
    observations = [item for item in (trace.get("tool_observations") or []) if isinstance(item, dict)]
    if not observations:
        observations = [item for item in (trace.get("tools") or []) if isinstance(item, dict)]
    if not observations:
        return None
    by_id = {str(item.get("evidence_id") or ""): item for item in evidence if item.get("evidence_id")}
    known: list[str] = []
    steps: list[dict[str, Any]] = []
    slot_updates = [item for item in (trace.get("slot_updates") or []) if isinstance(item, dict)]
    for index, observation in enumerate(observations, start=1):
        added = [str(eid) for eid in (observation.get("evidence_ids") or []) if eid and str(eid) not in known]
        before = list(known)
        known.extend(eid for eid in added if eid not in known)
        iteration = observation.get("iteration")
        if iteration is None and index > 1:
            iteration = index - 1
        resolved, remaining = _slots_at(slot_updates, iteration if iteration is not None else index - 1)
        steps.append(
            {
                "index": index,
                "tool": str(observation.get("tool") or ""),
                "retrieval_method": str(observation.get("retrieval_method") or ""),
                "reason": str(observation.get("reason") or ""),
                "before": before,
                "added": added,
                "added_items": [_diff_item(by_id.get(eid), eid) for eid in added],
                "resolved": resolved,
                "remaining": remaining,
                "strategy": str(observation.get("tool") or ""),
            }
        )
    return {"steps": steps, "final_ids": list(known)}


def evidence_lineage(
    citations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(item.get("evidence_id") or ""): item for item in evidence if item.get("evidence_id")}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cite in citations:
        evidence_id = str(cite.get("evidence_id") or "")
        item = by_id.get(evidence_id, {})
        rows.append(
            {
                "claim_ref": evidence_id or "citation",
                "evidence_id": evidence_id,
                "evidence_type": item.get("evidence_type") or "unknown",
                "chunk_id": cite.get("chunk_id") or item.get("chunk_id"),
                "document_id": cite.get("document_id") or item.get("document_id"),
                "event_id": cite.get("event_id") or item.get("event_id"),
                "source_url": cite.get("source_url"),
                "graph_refs": list(item.get("graph_refs") or []),
                "text": item.get("text"),
                "value": item.get("value"),
                "retrieval_method": item.get("retrieval_method"),
            }
        )
        if evidence_id:
            seen.add(evidence_id)
    for item in evidence:
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id and evidence_id not in seen:
            rows.append(
                {
                    "claim_ref": None,
                    "evidence_id": evidence_id,
                    "evidence_type": item.get("evidence_type") or "unknown",
                    "chunk_id": item.get("chunk_id"),
                    "document_id": item.get("document_id"),
                    "event_id": item.get("event_id"),
                    "source_url": None,
                    "graph_refs": list(item.get("graph_refs") or []),
                    "text": item.get("text"),
                    "value": item.get("value"),
                    "retrieval_method": item.get("retrieval_method"),
                }
            )
    return rows


def evidence_status(
    evidence: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    graph: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    graph_present = any(
        str(item.get("evidence_type") or "") in {"entity", "edge", "fact"} for item in evidence
    ) or bool((graph or {}).get("edges"))
    return {
        "retrieved": len(evidence),
        "used": len(citations),
        "citation_count": len(citations),
        # RAG source/chunk links are metadata, not evidence of graph retrieval.
        "graph_present": bool(graph_present) and source.get("pipeline") != "rag",
        "source_metadata_present": bool((graph or {}).get("nodes")) if source.get("pipeline") == "rag" else False,
        "retrieval_methods": list(source.get("retrieval_methods") or []),
    }


def efficiency_panel(
    source: dict[str, Any],
    agent: dict[str, Any] | None,
    steps: list[dict[str, Any]],
    timings: dict[str, Any],
) -> dict[str, Any]:
    attempts = source.get("attempts") if isinstance(source.get("attempts"), list) else []
    retries = max(0, len(attempts) - 1) if attempts else 0
    return {
        "tokens": _int(source.get("total_tokens")),
        "tokens_unknown": bool(source.get("tokens_unknown")),
        "model_calls": _int(source.get("model_calls")),
        "tool_calls": _int((agent or {}).get("total_tool_calls"), default=0),
        "steps": len(steps),
        "latency_ms": _float(source.get("latency_ms")),
        "retries": retries,
        "strategy_changes": list((agent or {}).get("strategy_changes") or []),
        "retrieval_methods": list(source.get("retrieval_methods") or []),
        "timings": dict(timings or {}),
    }


def reliability_panel(
    source: dict[str, Any],
    status: str,
    failure_class: Any,
    attempts: list[Any],
) -> dict[str, Any]:
    fail = str(failure_class or "").strip()
    key = str(status or "").strip().casefold()
    return {
        "attempts": len(attempts) if attempts else _int(source.get("model_calls"), default=0),
        "model_calls": _int(source.get("model_calls")),
        "retries": max(0, len(attempts) - 1) if attempts else 0,
        "failure_class": fail or None,
        "tokens_unknown": bool(source.get("tokens_unknown")),
        "timeout": fail.casefold() == "timeout" or key == "timeout",
        "unavailable": key == "unavailable",
        "malformed": key == "generation_error" or fail.casefold() == "malformed_output",
        "status": status or "",
        "status_label": status_label(status, failure_class),
    }


def compare_analysis(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [compare_row(item) for item in results]
    return {
        "comparison": rows,
        "what_changed": what_changed(results),
        "fixed_vs_agentic": fixed_vs_agentic(results),
    }


def compare_row(view: dict[str, Any]) -> dict[str, Any]:
    agent = view.get("agent") if isinstance(view.get("agent"), dict) else {}
    return {
        "pipeline": view.get("pipeline"),
        "pipeline_label": view.get("pipeline_label"),
        "answer": view.get("answer") or "",
        "answer_display": view.get("answer_display") or display_answer(view),
        "status": view.get("status"),
        "status_label": view.get("status_label"),
        "runtime_mode": view.get("runtime_mode"),
        "latency_ms": view.get("latency_ms"),
        "tokens": view.get("tokens"),
        "tokens_unknown": view.get("tokens_unknown"),
        "model_calls": view.get("model_calls"),
        "citation_count": len(view.get("citations") or []),
        "evidence_count": len(view.get("evidence") or []),
        "tool_calls": _int(agent.get("total_tool_calls"), default=0),
        "steps": len(view.get("steps") or []),
        "retrieval_methods": list(view.get("retrieval_methods") or []),
        "stop_reason": view.get("stop_reason") or "",
        "continue_explanation": view.get("continue_explanation") or "",
        "errors": list(view.get("errors") or []),
    }


def what_changed(results: list[dict[str, Any]]) -> list[str]:
    by_name = {str(item.get("pipeline") or ""): item for item in results}
    lines: list[str] = []
    rag = by_name.get("rag") or {}
    graph = by_name.get("graphrag") or {}
    agentic = by_name.get("agentic_graphrag") or {}
    if rag:
        lines.append(_retrieval_line(rag, "no retrieval method"))
    if graph:
        lines.append(_retrieval_line(graph, "typed graph retrieval"))
    if agentic:
        lines.append(_retrieval_line(agentic, "bounded graph investigation"))
    answers = {str((item.get("answer") or "")).strip() for item in results}
    if len(results) > 1 and len(answers) == 1 and next(iter(answers)):
        lines.append("All compared pipelines returned the same answer text.")
    elif len(answers) > 1:
        lines.append("Answer text differs across pipelines.")
    if agentic.get("agent") and graph:
        tool_calls = _int((agentic.get("agent") or {}).get("total_tool_calls"), default=len(agentic.get("steps") or []))
        lines.append(
            f"Agentic GraphRAG recorded {tool_calls} tool call(s); Fixed GraphRAG uses one typed retrieval."
        )
        if (agentic.get("agent") or {}).get("follow_up_occurred"):
            reason = agentic.get("continue_explanation") or "A follow-up retrieval was observed in the agent trace."
            lines.append(reason)
    statuses = sorted({str(item.get("status_label") or item.get("status") or "") for item in results if item.get("status")})
    if len(statuses) > 1:
        lines.append("Pipeline statuses: " + ", ".join(statuses) + ".")
    graph_flags = []
    for item in results:
        present = bool((item.get("evidence_status") or {}).get("graph_present"))
        if item.get("pipeline") == "rag":
            graph_flags.append(f"{item.get('pipeline_label')}: text-only retrieval")
        else:
            graph_flags.append(f"{item.get('pipeline_label')}: {'graph context present' if present else 'no graph context'}")
    if graph_flags:
        lines.append("; ".join(graph_flags) + ".")
    return lines


def _retrieval_line(view: dict[str, Any], fallback: str) -> str:
    label = str(view.get("pipeline_label") or view.get("pipeline") or "Pipeline")
    status = str(view.get("status") or "").strip().casefold()
    if status in {"unavailable", "error", "timeout"}:
        return f"{label} did not retrieve because the backend was {status}."
    methods = ", ".join(str(item) for item in (view.get("retrieval_methods") or []) if item)
    return f"{label} used {methods or fallback}."


def fixed_vs_agentic(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    by_name = {str(item.get("pipeline") or ""): item for item in results}
    graph = by_name.get("graphrag")
    agentic = by_name.get("agentic_graphrag")
    if not graph or not agentic:
        return None
    return {
        "graphrag": _side_card(graph, one_retrieval=True),
        "agentic_graphrag": _side_card(agentic, one_retrieval=False),
        "notes": [
            note
            for note in (
                agentic.get("continue_explanation"),
                agentic.get("stop_explanation"),
                graph.get("stop_explanation"),
            )
            if note
        ],
    }


def health_checks(payload: dict[str, Any]) -> list[dict[str, str]]:
    caps = payload.get("capabilities") if isinstance(payload.get("capabilities"), dict) else {}
    conn = payload.get("connection") if isinstance(payload.get("connection"), dict) else {}
    vector = payload.get("vector") if isinstance(payload.get("vector"), dict) else {}
    blockers = [str(item) for item in (payload.get("blockers") or []) if item]
    vector_status = str(vector.get("status") or "").upper()
    if vector_status == "READY" and vector.get("ok"):
        vector_check_status = "PASS"
        vector_detail = "TigerGraph vector schema, search query, and index are ready (read-only check)."
    elif vector_status == "NOT_READY":
        vector_check_status = "WARN"
        vector_detail = str(vector.get("error") or "TigerGraph vector capability is configured but not ready.")
    elif vector_status == "UNAVAILABLE":
        vector_check_status = "UNAVAILABLE"
        vector_detail = str(vector.get("error") or "TigerGraph vector capability is unavailable or not configured.")
    else:
        vector_check_status = "ERROR"
        vector_detail = str(vector.get("error") or "TigerGraph vector capability status is unknown.")
    checks = [
        _check("runtime", "Application runtime", "PASS", "Console process is serving this health endpoint."),
        _check(
            "llm",
            "LLM configuration",
            "PASS" if caps.get("llm_configured") else "UNAVAILABLE",
            "Provider settings are present." if caps.get("llm_configured") else "LLM_* settings are not configured. Secrets are not displayed.",
        ),
        _check(
            "corpus",
            "Corpus availability",
            "PASS" if caps.get("corpus") else "UNAVAILABLE",
            "Local corpus file is present." if caps.get("corpus") else "corpus.jsonl is not in this public snapshot.",
        ),
        _check(
            "tigergraph",
            "TigerGraph connectivity",
            "PASS" if conn.get("ok") else ("ERROR" if caps.get("tigergraph_configured") else "UNAVAILABLE"),
            "Read-only connection succeeded." if conn.get("ok") else (conn.get("error") or "TigerGraph is not configured."),
        ),
        _check(
            "queries",
            "Installed query readiness",
            "PASS" if payload.get("queries_ok") else "UNAVAILABLE",
            "Installed queries verified." if payload.get("queries_ok") else "Query verification did not pass.",
        ),
        _check(
            "retrieval",
            "Retrieval query readiness",
            "PASS" if payload.get("retrieval_queries_ok") else "UNAVAILABLE",
            "Retrieval queries verified." if payload.get("retrieval_queries_ok") else "Retrieval queries were not verified.",
        ),
        _check("vector", "TigerGraph vector capability", vector_check_status, vector_detail),
        _check(
            "benchmark",
            "Published benchmark data",
            "PASS",
            "Published public 100-question scores are loaded from the catalog.",
        ),
        _check(
            "export",
            "Trace export",
            "PASS",
            "Investigations export schema_version=1 without evaluator-only scoring fields.",
        ),
    ]
    if blockers and not conn.get("ok"):
        checks.append(_check("blockers", "Health blockers", "WARN", "; ".join(blockers)))
    return checks


def _check(check_id: str, label: str, status: str, detail: str) -> dict[str, str]:
    return {"id": check_id, "label": label, "status": status, "detail": detail}


def _side_card(view: dict[str, Any], *, one_retrieval: bool) -> dict[str, Any]:
    agent = view.get("agent") if isinstance(view.get("agent"), dict) else {}
    return {
        "pipeline": view.get("pipeline"),
        "pipeline_label": view.get("pipeline_label"),
        "retrieval_path": ", ".join(view.get("retrieval_methods") or []) or ("one typed retrieval" if one_retrieval else "bounded investigation"),
        "evidence_count": len(view.get("evidence") or []),
        "citation_count": len(view.get("citations") or []),
        "steps": len(view.get("steps") or []),
        "tool_calls": _int(agent.get("total_tool_calls"), default=0),
        "tokens": view.get("tokens"),
        "tokens_unknown": view.get("tokens_unknown"),
        "model_calls": view.get("model_calls"),
        "latency_ms": view.get("latency_ms"),
        "status": view.get("status"),
        "status_label": view.get("status_label"),
        "stop_explanation": view.get("stop_explanation") or "",
        "continue_explanation": view.get("continue_explanation") or "",
        "follow_up_occurred": bool(agent.get("follow_up_occurred")),
    }


def _judge_continue_text(reason: str) -> str:
    text = reason.strip()
    lowered = text.casefold()
    if "relational gap" in lowered or "held_at" in lowered or "in_games" in lowered:
        return "Initial evidence did not resolve the requested relationship, so a follow-up graph retrieval ran."
    if "vector" in lowered:
        return "Graph retrieval left a provenance gap, so one bounded TigerGraph vector fallback ran."
    if "provenance" in lowered or "source chunks" in lowered or "chunks were not retrieved" in lowered:
        return "A second retrieval was performed because an evidence slot for source documents remained unresolved."
    if "multiple candidates" in lowered or "collision" in lowered:
        return "Follow-up document retrieval ran because candidates remained ambiguous; the collision is preserved."
    if text.startswith("stop:"):
        return STOP_EXPLANATIONS.get(text.split(":", 1)[-1], text)
    if text:
        return text[0].upper() + text[1:]
    return "A follow-up retrieval was observed in the agent trace."


def _slots_at(updates: list[dict[str, Any]], iteration: Any) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    remaining: list[str] = []
    latest: dict[str, str] = {}
    for item in updates:
        slot_id = str(item.get("slot_id") or "")
        if not slot_id:
            continue
        if iteration is not None and item.get("iteration") is not None and item.get("iteration") > iteration:
            continue
        latest[slot_id] = str(item.get("status") or "")
    for slot_id, status in latest.items():
        if status == "resolved":
            resolved.append(slot_id)
        elif status in {"unknown", "searching"}:
            remaining.append(slot_id)
    return resolved, remaining


def _diff_item(item: dict[str, Any] | None, evidence_id: str) -> dict[str, Any]:
    item = item or {}
    return {
        "evidence_id": evidence_id,
        "evidence_type": item.get("evidence_type") or "unknown",
        "event_id": item.get("event_id"),
        "chunk_id": item.get("chunk_id"),
        "document_id": item.get("document_id"),
        "field_name": item.get("field_name"),
        "value": item.get("value"),
    }


def record_is_submission_safe(record: dict[str, Any]) -> bool:
    return not _contains_excluded(record)


def _contains_excluded(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in EXCLUDED_KEYS:
                return True
            if _contains_excluded(item):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_excluded(item) for item in value)
    return False


def _citation_view(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item.get("evidence_id"),
        "document_id": item.get("document_id"),
        "chunk_id": item.get("chunk_id"),
        "event_id": item.get("event_id"),
        "source_url": item.get("source_url"),
        "source_date": item.get("source_date"),
    }


def _evidence_view(item: dict[str, Any], citations: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_id = item.get("evidence_id")
    cited = any(row.get("evidence_id") == evidence_id for row in citations)
    return {
        "evidence_id": evidence_id,
        "evidence_type": item.get("evidence_type") or "unknown",
        "retrieval_method": item.get("retrieval_method"),
        "event_id": item.get("event_id"),
        "document_id": item.get("document_id"),
        "chunk_id": item.get("chunk_id") or item.get("source_chunk_id"),
        "field_name": item.get("field_name"),
        "value": item.get("value"),
        "text": item.get("text"),
        "why_retrieved": item.get("why_retrieved"),
        "graph_refs": list(item.get("graph_refs") or []),
        "cited": cited,
    }


def _agent_panel(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(trace, dict) or not trace:
        return None
    tool_names = [str(item.get("tool") or "") for item in (trace.get("tools") or []) if isinstance(item, dict)]
    observations = [item for item in (trace.get("tool_observations") or []) if isinstance(item, dict)]
    plan_steps = [item for item in (trace.get("plan_steps") or []) if isinstance(item, dict)]
    interpreted = dict(trace.get("interpreted") or {})
    stop = trace.get("stop_reason")
    if not (tool_names or observations or plan_steps or stop or any(interpreted.values())):
        return None
    follow_up_occurred = any(
        str(step.get("kind") or "") == "follow_up" for step in plan_steps
    ) or any(name in {"event_neighborhood", "supporting_chunks", "vector_search"} for name in tool_names)
    return {
        "interpreted": interpreted,
        "strategy_changes": list(trace.get("strategy_changes") or []),
        "stop_reason": str(stop) if stop else "",
        "stop_explanation": stop_explanation(str(stop) if stop else None, "agentic_graphrag", None),
        "total_tool_calls": _int(trace.get("total_tool_calls"), default=len(tool_names)),
        "total_steps": _int(trace.get("total_steps"), default=len(plan_steps)),
        "follow_up_occurred": bool(follow_up_occurred),
        "follow_up_decisions": list(trace.get("follow_up_decisions") or []),
        "tools": list(tool_names),
    }


def _steps(
    record: dict[str, Any],
    agent_trace: dict[str, Any] | None,
    timings: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if agent_trace and (agent_trace.get("tool_observations") or agent_trace.get("plan_steps")):
        return _agent_steps(agent_trace)
    if not _observed_run(record, timings, evidence):
        return []
    pipeline = str(record.get("pipeline") or "")
    methods = [str(item) for item in (record.get("retrieval_methods") or []) if item]
    steps: list[dict[str, Any]] = []
    if pipeline != "rag" and _float(timings.get("parsing_ms")):
        steps.append(_step(1, "parse", "QuestionParser", "typed QuerySpec", "ok", timings.get("parsing_ms"), 0, None))
    if methods or evidence or _float(timings.get("retrieval_ms")):
        action, reason, method = _retrieval_step_labels(pipeline, methods)
        steps.append(
            _step(
                len(steps) + 1,
                "retrieval",
                action,
                reason,
                "ok",
                timings.get("retrieval_ms") if _float(timings.get("retrieval_ms")) else None,
                len(evidence),
                method,
            )
        )
    if _float(timings.get("packing_ms")):
        steps.append(
            _step(len(steps) + 1, "pack", "ContextPacker", "shared evidence pack", "ok", timings.get("packing_ms"), len(evidence), None)
        )
    if _float(timings.get("generation_ms")) or record.get("answer") or record.get("status"):
        steps.append(
            _step(
                len(steps) + 1,
                "generate",
                "shared generator",
                "shared generator",
                str(record.get("status") or ""),
                timings.get("generation_ms") if _float(timings.get("generation_ms")) else None,
                0,
                None,
            )
        )
    return steps


def _observed_run(record: dict[str, Any], timings: dict[str, Any], evidence: list[dict[str, Any]]) -> bool:
    if evidence or record.get("retrieval_methods") or record.get("answer"):
        return True
    if any(_float(timings.get(key)) for key in ("parsing_ms", "retrieval_ms", "packing_ms", "generation_ms", "total_ms")):
        return True
    status = str(record.get("status") or "").strip().casefold()
    return bool(status) and status not in {"error", "unavailable"}


def _retrieval_step_labels(pipeline: str, methods: list[str]) -> tuple[str, str, str]:
    method = methods[0] if methods else ""
    if pipeline == "rag":
        return "BM25", "lexical chunk ranking; no graph", method or "bm25"
    if pipeline == "graphrag":
        return "typed GSQL", "one Fixed GraphRAG retrieval", method or "typed GSQL"
    return method or "retrieval", "observed retrieval", method


def _agent_steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    observations = [item for item in (trace.get("tool_observations") or []) if isinstance(item, dict)]
    if not observations:
        steps = []
        plan_steps = [item for item in (trace.get("plan_steps") or []) if isinstance(item, dict)]
        for index, plan in enumerate(plan_steps, start=1):
            actions = plan.get("actions") or []
            tool = actions[0].get("tool") if actions and isinstance(actions[0], dict) else plan.get("kind")
            nxt = plan_steps[index] if index < len(plan_steps) else None
            continued = bool(nxt and str(nxt.get("kind") or "") == "follow_up")
            steps.append(
                _step(
                    index,
                    str(plan.get("kind") or "step"),
                    str(tool or "plan"),
                    str(plan.get("reason") or ""),
                    "ok",
                    None,
                    0,
                    None,
                    continued=continued,
                    why_next=_judge_continue_text(str(nxt.get("reason") or "")) if continued and nxt else "",
                    result_summary=str(plan.get("kind") or "plan"),
                    evidence_ids=[],
                )
            )
        return steps
    steps = []
    for index, item in enumerate(observations, start=1):
        status = "ok" if item.get("success") else "error"
        if item.get("result_status"):
            status = str(item.get("result_status"))
        kind = "follow_up" if item.get("tool") in {"event_neighborhood", "supporting_chunks", "vector_search"} or item.get("parallel_group") == "repair" else "primary"
        if index == 1:
            kind = "primary"
        nxt = observations[index] if index < len(observations) else None
        continued = bool(nxt)
        ids = [str(eid) for eid in (item.get("evidence_ids") or []) if eid]
        count = len(ids)
        summary = f"{count} evidence id(s)" if count else ("tool error" if status == "error" else "no new evidence ids")
        why_next = ""
        if continued and nxt:
            why_next = _judge_continue_text(str(nxt.get("reason") or "") or "A follow-up retrieval was observed in the agent trace.")
        steps.append(
            _step(
                index,
                kind,
                str(item.get("tool") or "tool"),
                str(item.get("reason") or ""),
                status,
                item.get("elapsed_ms"),
                count,
                item.get("retrieval_method"),
                continued=continued,
                why_next=why_next,
                result_summary=summary,
                evidence_ids=ids,
            )
        )
    return steps


def _step(
    index: int,
    kind: str,
    action: str,
    reason: str,
    status: str,
    elapsed_ms: Any,
    evidence_count: int,
    retrieval_method: str | None,
    *,
    continued: bool = False,
    why_next: str = "",
    result_summary: str = "",
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "index": index,
        "kind": kind,
        "action": action,
        "reason": reason or "",
        "status": status or "",
        "elapsed_ms": _float(elapsed_ms) if elapsed_ms is not None else None,
        "evidence_count": evidence_count,
        "retrieval_method": retrieval_method or "",
        "continued": bool(continued),
        "why_next": why_next or "",
        "result_summary": result_summary or "",
        "evidence_ids": list(evidence_ids or []),
    }


def _architecture() -> dict[str, Any]:
    return {
        "ascii": (
            "                QUESTION\n"
            "                   |\n"
            "        +----------+----------+\n"
            "        |                     |\n"
            "       RAG              Query Parser\n"
            "      BM25                    |\n"
            "                           +--+--+\n"
            "                           |     |\n"
            "                       GraphRAG Agentic\n"
            "                          |       |\n"
            "                          +---+---+\n"
            "                              |\n"
            "                        TigerGraph\n"
            "                    graph / vector backend\n"
            "                              |\n"
            "                       grounded evidence\n"
            "                              |\n"
            "                     shared SemanticGenerator\n"
        ),
        "nodes": [
            {"id": "question", "label": "QUESTION"},
            {"id": "rag", "label": "RAG · BM25"},
            {"id": "parser", "label": "Query Parser"},
            {"id": "graphrag", "label": "GraphRAG"},
            {"id": "agentic", "label": "Agentic"},
            {"id": "tigergraph", "label": "TigerGraph"},
            {"id": "evidence", "label": "grounded evidence"},
            {"id": "generator", "label": "shared SemanticGenerator"},
            {"id": "vector", "label": "Vector · Agentic fallback"},
        ],
        "edges": [
            {"from": "question", "to": "rag"},
            {"from": "question", "to": "parser"},
            {"from": "parser", "to": "graphrag"},
            {"from": "parser", "to": "agentic"},
            {"from": "graphrag", "to": "tigergraph"},
            {"from": "agentic", "to": "tigergraph"},
            {"from": "agentic", "to": "vector", "dashed": True, "note": "optional bounded fallback"},
            {"from": "tigergraph", "to": "evidence"},
            {"from": "rag", "to": "evidence", "note": "chunks only; no graph"},
            {"from": "evidence", "to": "generator"},
            {"from": "vector", "to": "tigergraph", "dashed": True, "note": "optional bounded fallback"},
        ],
        "notes": [
            "RAG does NOT use the graph.",
            "GraphRAG uses fixed graph retrieval.",
            "Agentic adds evidence-state-driven follow-up investigation.",
            "TigerGraph vector search is an optional bounded Agentic fallback, not the default RAG retriever.",
        ],
    }


def _qid(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text == "aggregate":
        return ""
    return text


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
