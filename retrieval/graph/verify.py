"""Export, load, and verify the Phase 4 TigerGraph substrate."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from config.embeddings import load_embedding_settings
from config.settings import load_settings
from ingestion.chunker import chunk_corpus
from ingestion.graph_export import export_graph
from ingestion.graph_records import INSTALLED_QUERIES
from ingestion.loader import load_questions, parse_corpus
from ingestion.paths import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_GRAPH_EXPORT_DIR,
    DEFAULT_GRAPH_VERIFY_PATH,
    DEFAULT_PUBLIC_QUESTIONS_PATH,
    REPO_ROOT,
)
from retrieval.graph.client import (
    GraphOperationError,
    TigerGraphClient,
    _live_count_mismatches,
    exception_details,
    format_operation_error,
    repo_gsql_queries,
)
from retrieval.graph.contract import GraphStore, compare_solver_to_graph
from retrieval.graph.vector import SEARCH_QUERY
from retrieval.graph.vector_store import TigerGraphVectorStore
from retrieval.structured.index import StructuredIndex
from retrieval.structured.solver import StructuredSolver

SAMPLE_QTYPES = ("lookup", "aggregation", "superlative", "temporal", "multi_hop")


def run_export(corpus_path: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    corpus = parse_corpus(corpus_path or DEFAULT_CORPUS_PATH)
    chunks = chunk_corpus(corpus.documents)
    payload = export_graph(corpus.documents, output_dir or DEFAULT_GRAPH_EXPORT_DIR, chunks=chunks)
    graph = payload["graph"]
    return {
        "export": payload["manifest"],
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "graph": graph,
        "documents": corpus.documents,
        "chunks": chunks,
    }


def run_bind() -> dict[str, Any]:
    settings = load_settings()
    client = TigerGraphClient(settings)
    report: dict[str, Any] = {
        "settings": settings.redacted(),
        "connection": {"ok": False, "error": "", "environment": "unconfigured"},
        "bind": {"attempted": False, "ok": False},
        "schema": {"attempted": False, "ok": False},
        "load": {"attempted": False, "ok": False},
        "blockers": [],
    }
    connected = client.connect()
    report["connection"] = {
        "ok": connected,
        "error": client.error,
        "environment": client.environment,
    }
    if not connected:
        report["blockers"].append(client.error or "TigerGraph is not reachable")
        return report
    try:
        report["bind"]["attempted"] = True
        report["schema"]["attempted"] = True
        result = client.bind_global_types()
        report["bind"].update(result)
        report["bind"]["ok"] = bool(result.get("ok"))
        report["schema"]["ok"] = bool(result.get("ok"))
        report["schema"]["vertices"] = result.get("vertices")
        report["schema"]["edges"] = result.get("edges")
        report["schema"]["rest"] = {
            "missing_vertices": result.get("missing_vertices") or [],
            "missing_edges": result.get("missing_edges") or [],
        }
        report["schema"]["snapshot"] = result.get("schema")
    except GraphOperationError as exc:
        report["bind"]["error"] = exc.details
        report["blockers"].append(str(exc))
    except Exception as exc:
        details = exception_details(exc, operation="bind_global_types", method="run_bind")
        report["bind"]["error"] = details
        report["blockers"].append(format_operation_error(details))
    return report


def run_verify() -> dict[str, Any]:
    """Read-only live check: connect, schema, installed queries. No upsert or install."""

    settings = load_settings()
    client = TigerGraphClient(settings)
    report: dict[str, Any] = {
        "settings": settings.redacted(),
        "connection": {"ok": False, "error": "", "environment": "unconfigured"},
        "schema": {"attempted": False, "ok": False},
        "queries": {"attempted": False, "ok": False, "names": list(INSTALLED_QUERIES)},
        "retrieval_queries": {"attempted": False, "ok": False},
        "vector": _vector_unavailable("not_checked"),
        "load": {"attempted": False, "ok": False},
        "read_only": True,
        "blockers": [],
    }
    try:
        connected = client.connect()
    except Exception as exc:
        error = _safe_verify_error(exc)
        report["connection"]["error"] = error
        report["blockers"].append(error)
        report["vector"] = _vector_unavailable(error)
        return report
    connection_error = _redact_diagnostic(client.error or "")
    report["connection"] = {
        "ok": connected,
        "error": connection_error,
        "environment": client.environment,
    }
    if not connected:
        error = connection_error or "TigerGraph is not reachable"
        report["blockers"].append(error)
        report["vector"] = _vector_unavailable(error)
        return report
    try:
        report["schema"]["attempted"] = True
        schema = client.verify_schema()
        report["schema"].update(schema)
        if not schema.get("ok"):
            missing_v = ",".join(schema.get("missing_vertices") or [])
            missing_e = ",".join(schema.get("missing_edges") or [])
            report["blockers"].append(
                f"live schema verification failed vertices={missing_v} edges={missing_e}"
            )
        report["queries"]["attempted"] = True
        queries = client.verify_queries()
        report["queries"].update(queries)
        if not queries.get("ok"):
            missing = ",".join(queries.get("missing") or [])
            report["blockers"].append(f"required GSQL queries missing REST endpoints: {missing}")
        report["retrieval_queries"]["attempted"] = True
        retrieval = client.verify_retrieval_queries()
        report["retrieval_queries"].update(retrieval)
        if not retrieval.get("ok"):
            missing = ",".join(retrieval.get("missing") or [])
            report["blockers"].append(f"required retrieval queries missing REST endpoints: {missing}")
        report["vector"] = _verify_vector_read_only(client)
    except Exception as exc:
        error = _safe_verify_error(exc)
        report["blockers"].append(error)
        report["vector"] = _vector_unavailable(error)
    return report


def _vector_unavailable(error: str) -> dict[str, Any]:
    dimension = load_embedding_settings().dimension
    return {
        "attempted": False,
        "ok": False,
        "status": "UNAVAILABLE",
        "availability": "UNAVAILABLE",
        "schema_ok": False,
        "search_query_ok": False,
        "index_ok": False,
        "search_query": SEARCH_QUERY,
        "dimension": dimension,
        "error": _redact_diagnostic(error),
    }


def _verify_vector_read_only(client: TigerGraphClient) -> dict[str, Any]:
    """Probe vector readiness without schema, query, index, or data mutation."""

    dimension = load_embedding_settings().dimension
    result: dict[str, Any] = {
        "attempted": True,
        "ok": False,
        "status": "NOT_READY",
        "availability": "UNAVAILABLE",
        "schema_ok": False,
        "search_query_ok": False,
        "index_ok": False,
        "search_query": SEARCH_QUERY,
        "dimension": dimension,
        "error": "",
    }
    store = TigerGraphVectorStore(client, dimension=dimension)
    errors: list[str] = []
    try:
        schema = store.require_schema()
        result["schema_ok"] = True
        spec = schema.get("spec") if isinstance(schema, dict) else None
        if isinstance(spec, dict):
            result["dimension"] = spec.get("dimension") or dimension
            result["metric"] = spec.get("metric")
            result["index"] = spec.get("index_type")
    except Exception as exc:
        errors.append(_safe_vector_error("schema", exc))
    try:
        store.require_search_query()
        result["search_query_ok"] = True
    except Exception as exc:
        errors.append(_safe_vector_error("query", exc))
    try:
        index = store.require_index()
        result["index_ok"] = True
        if isinstance(index, dict):
            result["index_status"] = index.get("status")
    except Exception as exc:
        errors.append(_safe_vector_error("index", exc))
    if result["schema_ok"] and result["search_query_ok"] and result["index_ok"]:
        result["ok"] = True
        result["status"] = "READY"
        result["availability"] = "AVAILABLE"
    result["error"] = "; ".join(errors)
    return result


def _safe_vector_error(stage: str, exc: Exception) -> str:
    safe_message = _redact_diagnostic(str(exc))
    details = exception_details(
        RuntimeError(safe_message),
        operation=f"vector_{stage}",
        method=f"TigerGraphVectorStore.require_{stage}",
    )
    formatted = format_operation_error(details)
    return _redact_diagnostic(formatted)[:1000]


def _safe_verify_error(exc: Exception) -> str:
    if isinstance(exc, GraphOperationError):
        # Preserve structured operation/class context without forwarding raw details.
        return _redact_diagnostic(format_operation_error(exc.details))[:1000]
    details = exception_details(
        RuntimeError(_redact_diagnostic(str(exc))), operation="verify", method="run_verify"
    )
    details["exception_class"] = exc.__class__.__name__
    return _redact_diagnostic(format_operation_error(details))[:1000]


def _redact_diagnostic(text: str) -> str:
    """Keep diagnostic context while removing endpoint, credential and path values."""
    redacted = re.sub(r"""https?://[^\s\'"\\]+""", "[REDACTED_URL]", text or "")
    redacted = re.sub(
        r"""(?i)(\b(?:[\w-]*[_-])?(?:api[_-]?key|token|password|passwd|secret)\b["']?\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,;]+)""",
        r"\1[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"""(?i)(\bauthorization\b["']?\s*[:=]\s*)(?:"[^"]*"|'[^']*'|(?:(?:Bearer|Basic|\[REDACTED\])\s+(?![\w-]+\s*[:=]))?[^\s,;]+)""",
        r"\1[REDACTED]",
        redacted,
    )
    redacted = re.sub(r"(?i)(\bbearer\s+)[^\s,;]+", r"\1[REDACTED]", redacted)
    return re.sub(
        r"""(?i)(?:"(?:[a-z]:[\\/]|\\\\|/)[^"]*"|'(?:[a-z]:[\\/]|\\\\|/)[^']*'|\b[a-z]:[\\/][^\s,;'"]+|\\\\[^\s,;'"]+|(?<![\w:])/(?:[^\s,;'"]+))""",
        "[REDACTED_PATH]",
        redacted,
    )


def run_probe_write(cleanup: bool = True) -> dict[str, Any]:
    settings = load_settings()
    client = TigerGraphClient(settings)
    report: dict[str, Any] = {
        "settings": settings.redacted(),
        "connection": {"ok": False, "error": "", "environment": "unconfigured"},
        "probe": {"attempted": False, "ok": False},
        "load": {"attempted": False, "ok": False},
        "blockers": [],
    }
    connected = client.connect()
    report["connection"] = {
        "ok": connected,
        "error": client.error,
        "environment": client.environment,
    }
    if not connected:
        report["blockers"].append(client.error or "TigerGraph is not reachable")
        return report
    report["probe"]["attempted"] = True
    result = client.probe_write(cleanup=cleanup)
    report["probe"].update(result)
    report["probe"]["ok"] = bool(result.get("write_ok") and result.get("read_ok") and (not cleanup or result.get("cleaned_up")))
    if not report["probe"]["ok"]:
        report["blockers"].append(result.get("error") or "write probe failed")
    return report


def run_ingest(reset: bool = False, skip_load: bool = False) -> dict[str, Any]:
    settings = load_settings()
    exported = run_export()
    client = TigerGraphClient(settings)
    report: dict[str, Any] = {
        "settings": settings.redacted(),
        "export": exported["export"],
        "connection": {"ok": False, "error": "", "environment": "unconfigured"},
        "schema": {"attempted": False, "ok": False},
        "load": {"attempted": False, "ok": False, "strategy": "upsert_by_stable_id"},
        "queries": {"attempted": False, "ok": False, "names": list(INSTALLED_QUERIES)},
        "counts": {},
        "verification": {},
        "equivalence": {},
        "latency_ms": {},
        "blockers": [],
    }
    connected = client.connect()
    report["connection"] = {
        "ok": connected,
        "error": client.error,
        "environment": client.environment,
    }
    if not connected:
        report["blockers"].append(client.error or "TigerGraph is not reachable")
        report["local_contract"] = _local_verification(exported)
        return report
    if skip_load:
        report["blockers"].append("load skipped by flag")
        return report
    load_started = time.perf_counter()
    try:
        report["schema"]["attempted"] = True
        client.ensure_schema(reset=reset)
        report["schema"]["ok"] = client.verify_schema()["ok"]
        report["schema"].update(client.verify_schema())
        report["load"]["attempted"] = True
        loaded = client.upsert_export(DEFAULT_GRAPH_EXPORT_DIR)
        expected_counts = exported["export"]["counts"]
        shortfalls = _count_shortfalls(expected_counts, loaded)
        report["load"]["upserted"] = loaded
        report["load"]["elapsed_ms"] = (time.perf_counter() - load_started) * 1000.0
        report["load"]["ok"] = not shortfalls
        if shortfalls:
            report["load"]["shortfalls"] = shortfalls
            report["blockers"].append(
                "upsert accepted counts do not match export: " + json.dumps(shortfalls, sort_keys=True)
            )
        counts, mismatches = client.wait_for_export_counts(expected_counts)
        report["counts"] = counts
        report["load"]["ok"] = not shortfalls and not mismatches
        if mismatches:
            report["blockers"].append(
                "live TigerGraph counts do not match export: " + json.dumps(mismatches, sort_keys=True)
            )
        report["queries"]["attempted"] = True
        client.install_queries()
        query_check = client.verify_queries()
        report["queries"].update(query_check)
        if not query_check.get("ok"):
            missing = ",".join(query_check.get("missing") or [])
            report["blockers"].append(f"required GSQL queries missing REST endpoints: {missing}")
        if query_check.get("ok"):
            report["verification"] = _live_verification(client, exported)
            report["equivalence"] = _equivalence(client, exported)
        else:
            report["verification"] = {
                "ok": False,
                "count_mismatches": mismatches,
                "schema": client.verify_schema(),
                "installed": query_check,
            }
            report["equivalence"] = _equivalence(None, exported)
        report["latency_ms"] = (report["verification"] or {}).get("latency_ms", {})
        verify_mismatches = report["verification"].get("count_mismatches") or {}
        if verify_mismatches:
            report["load"]["ok"] = False
            if "live TigerGraph counts do not match export" not in " ".join(report["blockers"]):
                report["blockers"].append(
                    "live TigerGraph counts do not match export: " + json.dumps(verify_mismatches, sort_keys=True)
                )
        if not (report["verification"].get("schema") or {}).get("ok"):
            report["schema"]["ok"] = False
            report["blockers"].append("live schema verification failed")
        local_mis = int((report["equivalence"] or {}).get("local_mismatches") or 0)
        live_n = int((report["equivalence"] or {}).get("tigergraph_n") or 0)
        live_mis = int((report["equivalence"] or {}).get("tigergraph_mismatches") or 0)
        if local_mis:
            report["blockers"].append(f"local/export mismatches={local_mis}")
        if query_check.get("ok") and live_n <= 0:
            report["blockers"].append("live query/equivalence verification did not run")
        elif live_mis:
            report["blockers"].append(f"TigerGraph equivalence mismatches={live_mis}")
    except GraphOperationError as exc:
        report["load"]["error"] = exc.details
        report["blockers"].append(str(exc))
        report["local_contract"] = _local_verification(exported)
        return report
    except Exception as exc:
        detail = str(exc)
        if "schema is missing or incomplete" in detail.casefold():
            report["blockers"].append(detail)
        else:
            details = exception_details(exc, operation="ingest", method="run_ingest")
            report["load"]["error"] = details
            report["blockers"].append(format_operation_error(details))
        report["local_contract"] = _local_verification(exported)
        return report
    report["local_contract"] = _local_verification(exported)
    return report


def _count_shortfalls(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, dict[str, int]]:
    shortfalls: dict[str, dict[str, int]] = {}
    for name, count in expected.items():
        accepted = int(actual.get(name, 0) or 0)
        if accepted != int(count):
            shortfalls[name] = {"expected": int(count), "accepted": accepted}
    return shortfalls


def _local_verification(exported: dict[str, Any]) -> dict[str, Any]:
    graph = exported["graph"]
    documents = exported["documents"]
    store = GraphStore.from_export(graph)
    index = StructuredIndex.from_documents(documents)
    solver = StructuredSolver(index)
    questions = load_questions(DEFAULT_PUBLIC_QUESTIONS_PATH) if DEFAULT_PUBLIC_QUESTIONS_PATH.exists() else []
    equivalence = _compare_questions(solver, store, questions, client=None)
    title_only = [document for document in documents if document.doc_id == "Q3046361"]
    event_ids = {event.id for event in graph.events}
    chunk_docs = {chunk.document_id for chunk in graph.chunks}
    latency = {}
    samples = {
        "lookup_event": "How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?",
        "count_over_threshold": "According to the provided corpus, how many cycling events at the 2000 Summer Olympics had more than 30 competitors?",
        "argmax_competitors": "According to the provided corpus, which cycling event at the 2000 Summer Olympics had the highest number of competitors?",
        "previous_event_gold": "Who won the gold medal in the men's pole vault athletics event at the Summer Olympics held immediately before 2016?",
        "events_at_venue_date": "Who won the gold medal in the event held at Laura Biathlon & Ski Complex on 22 February 2014?",
    }
    qtypes = {
        "lookup_event": "lookup",
        "count_over_threshold": "aggregation",
        "argmax_competitors": "superlative",
        "previous_event_gold": "temporal",
        "events_at_venue_date": "multi_hop",
    }
    for name, question in samples.items():
        spec = solver.parser.parse(question, qtype=qtypes[name])
        started = time.perf_counter()
        store.execute(spec)
        latency[name] = round((time.perf_counter() - started) * 1000.0, 3)
    _ = title_only
    return {
        "gsql_file_queries": repo_gsql_queries(),
        "title_only_q3046361_is_event": "Q3046361" in event_ids,
        "title_only_has_document": any(document.id == "Q3046361" for document in graph.documents),
        "title_only_has_chunks": "Q3046361" in chunk_docs,
        "event_to_document": all(event.id == event.document_id for event in graph.events),
        "chunk_to_document": all(chunk.document_id in {document.id for document in graph.documents} for chunk in graph.chunks),
        "duplicate_event_ids": len(graph.events) - len(event_ids),
        "equivalence": equivalence,
        "venue_collision": _venue_collision(solver, store, None),
        "latency_ms": latency,
    }


def _live_verification(client: TigerGraphClient, exported: dict[str, Any]) -> dict[str, Any]:
    expected = exported["export"]["counts"]
    vertices = client.vertex_counts()
    edges = client.edge_counts()
    mismatches = _live_count_mismatches(expected, vertices, edges)
    latency = {}
    documents = exported["documents"]
    index = StructuredIndex.from_documents(documents)
    solver = StructuredSolver(index)
    store = GraphStore.from_export(exported["graph"])
    events_by_id = store.by_id
    questions = load_questions(DEFAULT_PUBLIC_QUESTIONS_PATH)
    by_type = {}
    for question in questions:
        by_type.setdefault(question["qtype"], question)
    for qtype in SAMPLE_QTYPES:
        question = by_type.get(qtype)
        if not question:
            continue
        spec = solver.parser.parse(question["question"], qtype=qtype)
        started = time.perf_counter()
        result = client.execute_spec(spec, events_by_id)
        latency[result.operation] = round(result.elapsed_ms, 3)
        _ = started
    return {
        "count_mismatches": mismatches,
        "ok": not mismatches,
        "latency_ms": latency,
        "installed": client.verify_queries(),
        "schema": client.verify_schema(),
    }


def _equivalence(client: TigerGraphClient | None, exported: dict[str, Any]) -> dict[str, Any]:
    documents = exported["documents"]
    index = StructuredIndex.from_documents(documents)
    solver = StructuredSolver(index)
    store = GraphStore.from_export(exported["graph"])
    questions = load_questions(DEFAULT_PUBLIC_QUESTIONS_PATH)
    return _compare_questions(solver, store, questions, client=client)


def _compare_questions(solver, store, questions, client: TigerGraphClient | None) -> dict[str, Any]:
    rows = []
    mismatches = 0
    live_mismatches = 0
    live_n = 0
    events_by_id = store.by_id
    started = time.perf_counter()
    for question in questions:
        spec = solver.parser.parse(question["question"], qtype=question.get("qtype"))
        python_result = solver.solve_spec(spec)
        local = store.execute(spec)
        local_cmp = compare_solver_to_graph(python_result, local)
        row = {
            "qid": question.get("qid"),
            "qtype": question.get("qtype"),
            "local_ok": local_cmp["ok"],
            "local_mismatches": local_cmp["mismatches"],
        }
        if not local_cmp["ok"]:
            mismatches += 1
        if client is not None and client.available:
            live_n += 1
            live = client.execute_spec(spec, events_by_id)
            live_cmp = compare_solver_to_graph(python_result, live)
            row["tg_ok"] = live_cmp["ok"]
            row["tg_mismatches"] = live_cmp["mismatches"]
            if not live_cmp["ok"]:
                live_mismatches += 1
        rows.append(row)
    return {
        "n": len(questions),
        "local_mismatches": mismatches,
        "tigergraph_n": live_n,
        "tigergraph_mismatches": live_mismatches,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "rows": [row for row in rows if not row.get("local_ok") or row.get("tg_ok") is False],
    }


def _venue_collision(solver, store, client) -> dict[str, Any]:
    question = "Who won the gold medal in the event held at Laura Biathlon & Ski Complex on 22 February 2014?"
    spec = solver.parser.parse(question, qtype="multi_hop")
    python_result = solver.solve_spec(spec)
    local = store.execute(spec)
    payload = {
        "python_status": python_result.status,
        "python_event_ids": [event.event_id for event in python_result.events],
        "local_status": local.status,
        "local_event_ids": local.event_ids,
        "returns_multiple": len(python_result.events) > 1,
        "does_not_guess": python_result.status == "ambiguous",
    }
    if client is not None and client.available:
        live = client.execute_spec(spec, store.by_id)
        payload["tg_status"] = live.status
        payload["tg_event_ids"] = live.event_ids
    return payload


def write_verify_payload(report: dict[str, Any], path: Path | None = None) -> Path:
    destination = path or DEFAULT_GRAPH_VERIFY_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(report, default=_json_default))
    serializable.pop("graph", None)
    serializable.pop("documents", None)
    serializable.pop("chunks", None)
    destination.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _json_default(value: Any):
    if hasattr(value, "counts"):
        return value.counts()
    return str(value)
