"""Summaries for the Phase 13 TigerGraph Vector RAG A/B. Does not change scoring."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

VARIANT_ORDER = ("V0", "V1", "V2")
VARIANT_LABELS = {
    "V0": "bm25_rag",
    "V1": "tigergraph_vector_rag",
    "V2": "bm25_tigergraph_vector_hybrid",
}
GOLD_QTYPES = (
    "lookup",
    "temporal",
    "multi_hop",
    "superlative",
    "aggregation",
)
STATUS_BUCKETS = (
    "ambiguity",
    "abstention",
)
PUBLIC_FAMILIES = GOLD_QTYPES + STATUS_BUCKETS
RECALL_AT_K_DEFINITION = (
    "whether at least one gold document appears in packed top-k; "
    "this is not full-set recall of all gold documents"
)
_ABSTAIN = frozenset({"abstained", "not_found", "unresolved", "unsupported"})


def family_for_row(row: dict[str, Any]) -> str:
    """Gold qtype only. Does not alias multi_hop to venue_disambiguation."""
    return str(row.get("qtype") or "").strip() or "unknown"


def variant_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    retrieval_ms = [float(row["retrieval_ms"]) for row in rows if row.get("retrieval_ms") is not None]
    generation_ms = [float(row["generation_ms"]) for row in rows if row.get("generation_ms") is not None]
    complete_values = [row["completeness"] for row in rows if row.get("completeness") is not None]
    grounded = [row for row in rows if row.get("grounding") is True]
    grounding_known = [row for row in rows if row.get("grounding") is not None]
    return {
        "n": n,
        "correctness": _rate(sum(1 for row in rows if row.get("correctness")), n),
        "correctness_exact": _rate(sum(1 for row in rows if row.get("correctness_exact")), n),
        "completeness": _rate(sum(1 for value in complete_values if value), len(complete_values))
        if complete_values
        else None,
        "grounding": _rate(len(grounded), len(grounding_known)) if grounding_known else None,
        "citation_validity": _rate(sum(1 for row in rows if row.get("citation_validity") is True), n),
        "tokens_total": int(sum(int(row.get("tokens") or 0) for row in rows)),
        "latency_avg_ms": round(mean(latencies), 3) if latencies else None,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "retrieval_p50_ms": _percentile(retrieval_ms, 50),
        "retrieval_p95_ms": _percentile(retrieval_ms, 95),
        "generation_p50_ms": _percentile(generation_ms, 50),
        "generation_p95_ms": _percentile(generation_ms, 95),
        "generation_failures": sum(1 for row in rows if row.get("answer_status") == "generation_error"),
        "abstentions": sum(1 for row in rows if row.get("answer_status") in _ABSTAIN),
        "ambiguous": sum(1 for row in rows if row.get("answer_status") == "ambiguous"),
        "not_found": sum(1 for row in rows if row.get("answer_status") == "not_found"),
        "recall_at_5": _rate(sum(1 for row in rows if row.get("recall_at_5")), n),
        "recall_at_10": _rate(sum(1 for row in rows if row.get("recall_at_10")), n),
        "candidate_recall": _mean_optional([row.get("candidate_recall") for row in rows]),
        "packed_recall": _mean_optional([row.get("packed_recall") for row in rows]),
        "status_counts": dict(Counter(str(row.get("answer_status")) for row in rows)),
        "failure_counts": dict(Counter(row["failure_category"] for row in rows if row.get("failure_category"))),
    }


def summarize_vector_ablation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row.get("variant"))].append(row)
    overall = {vid: variant_stats(items) for vid, items in by_variant.items()}
    families: dict[str, dict[str, Any]] = {}
    for family in PUBLIC_FAMILIES:
        family_rows = [row for row in rows if _row_in_family(row, family)]
        if not family_rows:
            continue
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in family_rows:
            grouped[str(row.get("variant"))].append(row)
        families[family] = {vid: variant_stats(items) for vid, items in grouped.items()}
    deltas = {}
    baseline = overall.get("V0")
    if baseline:
        for vid in VARIANT_ORDER:
            if vid == "V0" or vid not in overall:
                continue
            deltas[vid] = _delta(baseline, overall[vid])
    return {"overall": overall, "by_family": families, "deltas_vs_V0": deltas}


def attach_retrieval_metrics(row: dict[str, Any], gold_ids: list[str], result) -> dict[str, Any]:
    packed_ids = _unique(
        [
            str(item.get("document_id") or item.get("event_id") or "")
            for item in (result.evidence_metadata or [])
            if item.get("document_id") or item.get("event_id")
        ]
    )
    params = (result.retrieval_metadata or {}).get("params") or {}
    candidate_ids = [str(value) for value in (params.get("candidate_document_ids") or packed_ids)]
    gold = [str(doc_id) for doc_id in gold_ids]
    packed_ranks = {
        str(item.get("document_id") or item.get("event_id")): index
        for index, item in enumerate(result.evidence_metadata or [], start=1)
        if item.get("document_id") or item.get("event_id")
    }
    row = dict(row)
    row["recall_at_5"] = _hit_at(gold, packed_ranks, 5)
    row["recall_at_10"] = _hit_at(gold, packed_ranks, 10)
    row["recall_at_k_definition"] = RECALL_AT_K_DEFINITION
    row["candidate_recall"] = _set_recall(gold, candidate_ids)
    row["packed_recall"] = _set_recall(gold, packed_ids)
    timings = result.timings
    row["retrieval_ms"] = getattr(timings, "retrieval_ms", None)
    row["generation_ms"] = getattr(timings, "generation_ms", None)
    row["backend"] = params.get("backend") or (result.retrieval_metadata or {}).get("method")
    row["backends"] = list(params.get("backends") or [])
    return row


def fairness_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row.get("variant"))].append(row)
    v0 = by_variant.get("V0") or []
    v1 = by_variant.get("V1") or []
    v2 = by_variant.get("V2") or []
    return {
        "v1_has_graph_ops": any(row.get("graph_operations") for row in v1),
        "v2_has_graph_ops": any(row.get("graph_operations") for row in v2),
        "v0_used_vector": any("vector" in str(row.get("backend") or "") for row in v0),
        "v1_used_bm25": any(row.get("backend") == "bm25" or "bm25" in (row.get("backends") or []) for row in v1),
        "v2_missing_bm25": any("bm25" not in (row.get("backends") or ["bm25"]) for row in v2),
        "v2_missing_vector": any("tigergraph_vector" not in (row.get("backends") or []) for row in v2),
        "private_eval_used": False,
        "same_scorer": "evaluation.benchmark.score_harness_result",
    }


def _row_in_family(row: dict[str, Any], family: str) -> bool:
    status = str(row.get("answer_status") or "")
    if family == "ambiguity":
        return status == "ambiguous"
    if family == "abstention":
        return status in _ABSTAIN
    return family_for_row(row) == family


def _delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in (
        "correctness",
        "correctness_exact",
        "grounding",
        "citation_validity",
        "recall_at_5",
        "recall_at_10",
        "candidate_recall",
        "packed_recall",
    ):
        if left.get(key) is None or right.get(key) is None:
            out[key + "_pp"] = None
        else:
            out[key + "_pp"] = round(float(right[key]) - float(left[key]), 3)
    for key in ("tokens_total", "latency_p50_ms", "latency_p95_ms", "retrieval_p50_ms", "generation_p50_ms"):
        if left.get(key) is None or right.get(key) is None:
            out[key] = None
        else:
            out[key] = round(float(right[key]) - float(left[key]), 3)
    return out


def _hit_at(gold: list[str], ranks: dict[str, int], cutoff: int) -> bool:
    """True if any gold document is among packed ranks 1..cutoff. Not full-set recall."""
    if not gold:
        return False
    return any(ranks.get(doc_id, 10**9) <= cutoff for doc_id in gold)


def _set_recall(gold: list[str], retrieved: list[str]) -> float | None:
    if not gold:
        return None
    hits = sum(1 for doc_id in gold if doc_id in set(retrieved))
    return round(hits / len(gold), 4)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _rate(hits: int, n: int) -> float | None:
    if not n:
        return None
    return round(100.0 * hits / n, 1)


def _mean_optional(values: list[Any]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return round(mean(usable), 4)


def _percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[index], 3)
