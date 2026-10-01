"""Live investigation runtime over existing harness adapters. No scoring."""

from __future__ import annotations

import threading
from typing import Any

from answering.factory import build_generator
from answering.packer import ContextPacker
from config.llm import load_llm_settings
from config.settings import load_settings
from evaluation.harness import AgenticGraphRAGAdapter, GraphRAGAdapter, HarnessResult, RAGAdapter
from evaluation.trace_export import _strip_excluded, export_submission_record
from ingestion.chunker import ChunkingConfig, chunk_corpus
from ingestion.loader import parse_corpus
from ingestion.paths import DEFAULT_CORPUS_PATH
from retrieval.agentic.pipeline import AgenticGraphRAGPipeline
from retrieval.graph.client import TigerGraphClient
from retrieval.graph.verify import run_verify
from retrieval.graphrag.parser import GraphRAGQuestionParser
from retrieval.graphrag.pipeline import FixedGraphRAGPipeline
from retrieval.graphrag.retriever import GraphRetriever
from retrieval.rag.retriever import TextRetriever
from retrieval.structured.question import QuestionParser
from ui.catalog import PIPELINE_LABELS, PIPELINE_ORDER
from ui.viewmodels import (
    compare_analysis,
    empty_investigation,
    health_checks,
    investigation_from_export,
    record_is_submission_safe,
    refresh_status_fields,
    status_label,
)

MAX_QUESTION_CHARS = 4000
GRAPH_PIPELINES = frozenset({"graphrag", "agentic_graphrag"})

_LOCK = threading.Lock()
_RUNTIME: ConsoleRuntime | None = None


class ConsoleRuntime:
    """Lazy, process-local adapters for the judge console."""

    def __init__(self, *, harness=None) -> None:
        self._harness = harness
        self._init_error = ""
        self._generator_name = ""
        self._corpus_ok = False
        self._graph_ok = False
        self._ready = harness is not None

    def capabilities(self) -> dict[str, Any]:
        llm = load_llm_settings()
        tg = load_settings()
        corpus = DEFAULT_CORPUS_PATH.is_file()
        return {
            "corpus": corpus,
            "tigergraph_configured": bool(tg.configured),
            "llm_configured": bool(llm.configured),
            "generator": "semantic" if llm.configured else "deterministic",
            "rag": corpus,
            "graphrag": bool(tg.configured),
            "agentic_graphrag": bool(tg.configured),
            "read_only": True,
            "mode": "live" if (corpus or tg.configured) else "unavailable",
        }

    def investigate(self, question: str, pipeline: str) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            view = empty_investigation(question="", pipeline=pipeline)
            view["status"] = "error"
            view["status_label"] = status_label("error")
            view["runtime_mode"] = "error"
            view["runtime_mode_label"] = "ERROR"
            view["errors"] = ["empty_question"]
            view["warnings"] = ["Enter a question to investigate."]
            return refresh_status_fields(view)
        if len(question) > MAX_QUESTION_CHARS:
            view = empty_investigation(question=question[:MAX_QUESTION_CHARS], pipeline=pipeline)
            view["status"] = "error"
            view["status_label"] = status_label("error")
            view["runtime_mode"] = "error"
            view["runtime_mode_label"] = "ERROR"
            view["errors"] = ["question_too_long"]
            return refresh_status_fields(view)
        if pipeline not in PIPELINE_LABELS:
            view = empty_investigation(question=question, pipeline=pipeline)
            view["status"] = "error"
            view["status_label"] = status_label("error")
            view["runtime_mode"] = "error"
            view["runtime_mode_label"] = "ERROR"
            view["errors"] = [f"unknown_pipeline:{pipeline}"]
            return refresh_status_fields(view)
        try:
            harness = self._ensure_harness()
        except Exception as exc:
            return _unavailable(question, pipeline, f"runtime_init:{exc.__class__.__name__}: {exc}")
        system = harness.systems.get(pipeline)
        if system is None:
            return _unavailable(question, pipeline, f"pipeline_unavailable:{pipeline}")
        try:
            result = system.run(question)
        except Exception as exc:
            failed = HarnessResult(
                system_name=pipeline,
                question=question,
                answer="",
                citations=[],
                latency_ms=0.0,
                retrieval_metadata={},
                evidence_metadata=[],
                status="error",
                errors=[f"system_error:{exc.__class__.__name__}: {exc}"],
            )
            return investigation_from_result(failed)
        return investigation_from_result(result)

    def compare(self, question: str) -> dict[str, Any]:
        results = [self.investigate(question, pipeline) for pipeline in PIPELINE_ORDER]
        analysis = compare_analysis(results)
        return {
            "question": question,
            "results": results,
            "comparison": analysis["comparison"],
            "what_changed": analysis["what_changed"],
            "fixed_vs_agentic": analysis["fixed_vs_agentic"],
        }

    def _ensure_harness(self):
        if self._harness is not None:
            return self._harness
        with _LOCK:
            if self._harness is not None:
                return self._harness
            self._harness = build_live_harness()
            return self._harness


def get_runtime() -> ConsoleRuntime:
    global _RUNTIME
    with _LOCK:
        if _RUNTIME is None:
            _RUNTIME = ConsoleRuntime()
        return _RUNTIME


def reset_runtime_for_tests(runtime: ConsoleRuntime | None = None) -> None:
    global _RUNTIME
    with _LOCK:
        _RUNTIME = runtime


def build_live_harness():
    settings = load_settings()
    llm = load_llm_settings()
    packer = ContextPacker()
    generator = build_generator("semantic" if llm.configured else "deterministic")
    question_parser = QuestionParser()
    systems: list[Any] = []

    if DEFAULT_CORPUS_PATH.is_file():
        corpus = parse_corpus(DEFAULT_CORPUS_PATH)
        chunks = chunk_corpus(corpus.documents, ChunkingConfig())
        try:
            from evaluation.retrieval_gold import build_parser

            _index, question_parser = build_parser(corpus)
        except Exception:
            question_parser = QuestionParser()
        systems.append(
            RAGAdapter(TextRetriever(chunks), packer, generator, method="sparse", top_k=10)
        )

    graph_parser = GraphRAGQuestionParser(question_parser)
    if settings.configured:
        client = TigerGraphClient(settings)
        if client.connect():
            retriever = GraphRetriever.from_client(client)
            systems.append(GraphRAGAdapter(FixedGraphRAGPipeline(graph_parser, retriever, packer, generator)))
            systems.append(AgenticGraphRAGAdapter(AgenticGraphRAGPipeline(graph_parser, retriever, packer, generator)))

    if not systems:
        raise RuntimeError("no live pipelines available: corpus and/or TigerGraph are missing")
    from evaluation.harness import ThreeWayEvaluationHarness

    return ThreeWayEvaluationHarness(systems)


def investigation_from_result(result: HarnessResult) -> dict[str, Any]:
    exported = export_submission_record(result, pipeline=result.system_name)
    exported["evidence"] = _strip_excluded(list(result.evidence_metadata or []))
    if not record_is_submission_safe(exported):
        raise RuntimeError("investigation export leaked evaluator fields")
    view = investigation_from_export(exported)
    view["recorded"] = False
    view["recorded_label"] = ""
    view["export_record"] = exported
    return view


def health_payload() -> dict[str, Any]:
    caps = get_runtime().capabilities()
    try:
        verify = run_verify()
    except Exception as exc:
        verify = {
            "connection": {"ok": False, "error": str(exc), "environment": "error"},
            "schema": {"ok": False},
            "queries": {"ok": False},
            "retrieval_queries": {"ok": False},
            "blockers": [f"health_error:{exc.__class__.__name__}"],
        }
    connection = verify.get("connection") if isinstance(verify.get("connection"), dict) else {}
    schema = verify.get("schema") if isinstance(verify.get("schema"), dict) else {}
    queries = verify.get("queries") if isinstance(verify.get("queries"), dict) else {}
    retrieval = verify.get("retrieval_queries") if isinstance(verify.get("retrieval_queries"), dict) else {}
    payload = {
        "read_only": True,
        "capabilities": caps,
        "connection": {
            "ok": bool(connection.get("ok")),
            "environment": connection.get("environment"),
            "error": connection.get("error") or "",
        },
        "schema_ok": bool(schema.get("ok")),
        "queries_ok": bool(queries.get("ok")),
        "retrieval_queries_ok": bool(retrieval.get("ok")),
        "blockers": list(verify.get("blockers") or []),
        "tigergraph": {
            "product": "TigerGraph",
            "graph": load_settings().graphname or "OlympicGraph",
            "configured": bool(load_settings().configured),
        },
    }
    payload["checks"] = health_checks(payload)
    payload["overall"] = _health_overall(payload["checks"])
    return payload


def _unavailable(question: str, pipeline: str, reason: str) -> dict[str, Any]:
    view = empty_investigation(question=question, pipeline=pipeline)
    view["status"] = "unavailable"
    view["status_label"] = status_label("unavailable")
    view["runtime_mode"] = "unavailable"
    view["runtime_mode_label"] = "UNAVAILABLE"
    view["errors"] = [reason]
    view["warnings"] = [_human_unavailable(pipeline, reason)]
    return refresh_status_fields(view)


def _health_overall(checks: list[dict[str, str]]) -> str:
    states = {item.get("status") for item in checks}
    if "ERROR" in states:
        return "ERROR"
    if "UNAVAILABLE" in states:
        return "UNAVAILABLE"
    if "WARN" in states:
        return "WARN"
    return "PASS"


def _human_unavailable(pipeline: str, reason: str) -> str:
    if pipeline == "rag":
        return "RAG needs the local corpus at _research/hackathon-resources/corpus/corpus.jsonl."
    if pipeline in GRAPH_PIPELINES:
        return "Graph pipelines need a configured, reachable OlympicGraph (read-only)."
    return reason
