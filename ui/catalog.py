"""Single structured source for published public metrics and system status.

Replace BENCHMARK in this module to publish a later official run. Do not
duplicate these numbers in frontend templates.
"""

from __future__ import annotations

from typing import Any

BENCHMARK_ID = "published_public_100"
CANONICAL_N = 100

PIPELINES: dict[str, dict[str, Any]] = {
    "rag": {
        "id": "rag",
        "name": "RAG",
        "subtitle": "BM25 / sparse · no graph",
        "correctness": 67.0,
        "exact": 67.0,
        "completeness": None,
        "grounding": 95.0,
        "citation_validity": 95.0,
        "errors": None,
        "tokens": None,
        "latency_p50_ms": None,
        "latency_mean_ms": 9136,
        "latency_p95_ms": 21491,
    },
    "graphrag": {
        "id": "graphrag",
        "name": "Fixed GraphRAG",
        "subtitle": "Fixed typed GSQL · one retrieval",
        "correctness": 98.0,
        "exact": 98.0,
        "completeness": 100.0,
        "grounding": 99.0,
        "citation_validity": 99.0,
        "errors": None,
        "tokens": None,
        "latency_p50_ms": None,
        "latency_mean_ms": 6290,
        "latency_p95_ms": 13926,
    },
    "agentic_graphrag": {
        "id": "agentic_graphrag",
        "name": "Agentic GraphRAG",
        "subtitle": "BoundedPlanner · optional follow-up",
        "correctness": 97.0,
        "exact": 97.0,
        "completeness": 100.0,
        "grounding": 99.0,
        "citation_validity": 99.0,
        "errors": None,
        "tokens": None,
        "latency_p50_ms": None,
        "latency_mean_ms": 6396,
        "latency_p95_ms": 17077,
    },
}

FAMILY_LABELS: dict[str, str] = {
    "lookup": "Direct factual",
    "aggregation": "Complete-set count",
    "superlative": "Comparative (maximum)",
    "temporal": "Temporal",
    "multi_hop": "Relationship / multi-hop",
}

FAMILIES: list[dict[str, Any]] = [
    {"id": "lookup", "label": FAMILY_LABELS["lookup"], "n": 19, "rag": 100.0, "graphrag": 100.0, "agentic_graphrag": 100.0},
    {"id": "aggregation", "label": FAMILY_LABELS["aggregation"], "n": 21, "rag": 0.0, "graphrag": 95.2, "agentic_graphrag": 100.0},
    {"id": "superlative", "label": FAMILY_LABELS["superlative"], "n": 10, "rag": 90.0, "graphrag": 100.0, "agentic_graphrag": 100.0},
    {"id": "temporal", "label": FAMILY_LABELS["temporal"], "n": 22, "rag": 77.3, "graphrag": 100.0, "agentic_graphrag": 100.0},
    {"id": "multi_hop", "label": FAMILY_LABELS["multi_hop"], "n": 28, "rag": 78.6, "graphrag": 96.4, "agentic_graphrag": 89.3},
]

AGENT_BEHAVIOR: dict[str, Any] = {
    "avg_tool_calls": 1.55,
    "one_call": 72,
    "follow_ups": 28,
    "neighborhood": 27,
}

UNPUBLISHED_METRICS: tuple[str, ...] = ("errors", "tokens", "latency_p50_ms")

METRIC_NOTES: dict[str, str] = {
    "completeness": "Completeness is complete-set aggregation only (graph). RAG has no complete-set operator.",
    "latency_p50_ms": "Published latency includes mean and p95, not p50.",
    "tokens": "Token totals were not published for this public run.",
    "errors": "A headline error count was not published for this public run.",
}

VECTOR_EXPERIMENT: dict[str, Any] = {
    "label": "Vector retrieval experiment — not the production RAG retriever",
    "production": False,
    "variants": [
        {"id": "V0", "mechanism": "BM25 (production RAG control)", "correctness": 66},
        {"id": "V1", "mechanism": "TigerGraph vector search", "correctness": 53},
        {"id": "V2", "mechanism": "BM25 + TigerGraph vector hybrid", "correctness": 65},
    ],
    "note": "V0 is an experiment-harness BM25 control (66%), not a replacement for the published RAG score (67%).",
}

SYSTEM_STATUS: dict[str, Any] = {
    "label": "Verified infrastructure / experiment capability",
    "production_note": "Not used as the default production RAG retriever",
    "tigergraph": {
        "product": "TigerGraph",
        "graph": "OlympicGraph",
        "version": "4.2.5",
    },
    "vector": {
        "chunk_embeddings_indexed": 28905,
        "chunk_embeddings_total": 28905,
        "dimension": 384,
        "metric": "COSINE",
        "index": "HNSW",
        "status": "READY",
        "search_query": "search_chunk_embedding",
        "search_status": "AVAILABLE",
        "role": "Verified infrastructure / experiment capability",
        "production": False,
        "production_note": "Not used as the default production RAG retriever",
    },
    "routing": {
        "rag": "BM25 TextRetriever. Does not use the graph.",
        "graphrag": "QuestionParser then one typed GSQL retrieval (include_chunks=True).",
        "agentic": "QuestionParser then BoundedPlanner; retrieve_spec then optional follow-up.",
        "vector": "Experiment-only. Not production RAG routing.",
        "grip": "Not production. Not part of this console.",
        "generator": "Shared SemanticGenerator for all three production pipelines.",
    },
}

WHY_AGENTIC: dict[str, Any] = {
    "title": "Why Agentic?",
    "disclaimer": (
        "Agentic did not beat Fixed GraphRAG on the public set (97% vs 98%). "
        "The value is adaptive investigation under a hard budget, not a higher headline score."
    ),
    "rules": [
        {
            "when": "WHEN INITIAL EVIDENCE IS SUFFICIENT",
            "then": "stop",
            "detail": "One retrieve_spec call can be enough; the agent stops when the answer field is filled.",
        },
        {
            "when": "WHEN A RELATIONAL GAP REMAINS",
            "then": "investigate",
            "detail": "A unique multi-hop Event may still be missing typed HELD_AT / IN_GAMES hops.",
        },
        {
            "when": "WHEN NEW EVIDENCE IS REQUIRED",
            "then": "retrieve again",
            "detail": "Follow-up tools: event_neighborhood and supporting_chunks.",
        },
        {
            "when": "WHEN EVIDENCE IS AMBIGUOUS",
            "then": "preserve ambiguity",
            "detail": "Fail closed. Do not guess among colliding candidates.",
        },
        {
            "when": "WHEN EVIDENCE IS SUFFICIENT",
            "then": "answer and stop",
            "detail": "stop_reason comes from the agent budget and slot ledger, not from a second planner.",
        },
    ],
}

PIPELINE_LABELS: dict[str, str] = {
    "rag": "RAG",
    "graphrag": "Fixed GraphRAG",
    "agentic_graphrag": "Agentic GraphRAG",
}

PIPELINE_ORDER: tuple[str, ...] = ("rag", "graphrag", "agentic_graphrag")

STATUS_LABELS: dict[str, str] = {
    "answered": "ANSWERED",
    "not_found": "NOT FOUND",
    "ambiguous": "AMBIGUOUS",
    "unresolved": "UNRESOLVED",
    "unsupported": "UNRESOLVED",
    "generation_error": "GENERATION ERROR",
    "timeout": "TIMEOUT",
    "abstained": "ABSTAINED",
    "invalid_citations": "INVALID CITATIONS",
    "error": "ERROR",
    "unavailable": "UNAVAILABLE",
}

STOP_EXPLANATIONS: dict[str, str] = {
    "answered": "Investigation stopped because the required evidence was sufficient.",
    "ambiguous": "Investigation stopped because multiple candidates remain; the system does not guess.",
    "not_found": "Investigation stopped because no matching evidence was found.",
    "unresolved": "Investigation stopped because the question could not be resolved to a typed query.",
    "unsupported": "Investigation stopped because the question is outside the typed parser contract.",
    "budget_exhausted": "Investigation stopped because the configured investigation limit was reached.",
    "no_progress": "Investigation stopped because follow-up retrieval made no progress.",
    "tool_failure": "Investigation stopped because a retrieval tool failed.",
    "unavailable": "Investigation stopped because the backend was unavailable.",
    "error": "Investigation stopped because the backend failed.",
    "timeout": "Investigation stopped because the request timed out.",
    "generation_error": "Investigation stopped because generation failed.",
}

RUNTIME_MODES: dict[str, str] = {
    "live": "LIVE",
    "preview": "PREVIEW",
    "unavailable": "UNAVAILABLE",
    "error": "ERROR",
}

HEALTH_STATES: tuple[str, ...] = ("PASS", "WARN", "UNAVAILABLE", "ERROR")


def canonical_benchmark() -> dict[str, Any]:
    return {
        "id": BENCHMARK_ID,
        "label": "Published public benchmark",
        "n": CANONICAL_N,
        "population": "100 official public questions",
        "generator": "SemanticGenerator",
        "graph": "OlympicGraph",
        "source": "docs/metrics.md",
        "replaceable": True,
        "pipelines": {key: dict(value) for key, value in PIPELINES.items()},
        "families": [dict(row) for row in FAMILIES],
        "family_labels": dict(FAMILY_LABELS),
        "agent_behavior": dict(AGENT_BEHAVIOR),
        "unpublished_metrics": list(UNPUBLISHED_METRICS),
        "notes": dict(METRIC_NOTES),
        "vector_experiments_excluded": True,
    }
