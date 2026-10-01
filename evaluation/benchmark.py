"""Score three-way harness outputs. Gold answers are used only here."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from evaluation.harness import HarnessResult
from evaluation.normalize import answers_match_exact, answers_match_normalized, as_answer_list, truncate
from ingestion.paths import REPO_ROOT

SYSTEM_ORDER = ("rag", "graphrag", "agentic_graphrag")
SUPPRESSED_STATUSES = frozenset(
    {
        "ambiguous",
        "not_found",
        "unresolved",
        "unsupported",
        "abstained",
        "invalid_citations",
        "generation_error",
        "error",
        "not_implemented",
    }
)
ERROR_STATUSES = frozenset({"error", "generation_error", "invalid_citations", "not_implemented"})


def public_run_id(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def score_harness_result(record: dict[str, Any], result: HarnessResult) -> dict[str, Any]:
    """Attach evaluation metrics after a system has already produced `result`."""

    gold = record.get("answer")
    status = str(result.status)
    answer = result.answer or ""
    answer_suppressed = (not answer.strip()) or status in SUPPRESSED_STATUSES
    exact = answers_match_exact(answer, gold) if answer.strip() else False
    normalized = answers_match_normalized(answer, gold) if answer.strip() else False
    citation_validity = _citation_validity(result)
    grounding, grounding_status = _grounding(result, answer_suppressed, citation_validity)
    completeness, completeness_reason = _completeness(record, result)
    gold_ids = [str(doc_id) for doc_id in (record.get("gold_doc_ids") or [])]
    retrieved_ids = _retrieved_ids(result)
    gold_hits = sum(1 for doc_id in gold_ids if doc_id in retrieved_ids)
    gold_doc_recall = (gold_hits / len(gold_ids)) if gold_ids else None
    specific = _system_specific(result)
    errors = list(result.errors)
    row = {
        "qid": record.get("qid"),
        "qtype": record.get("qtype"),
        "question": record.get("question"),
        "system_name": result.system_name,
        "answer": answer,
        "expected": gold,
        "answer_status": status,
        "answer_suppressed": answer_suppressed,
        "correctness": normalized,
        "correctness_exact": exact,
        "completeness": completeness,
        "completeness_reason": completeness_reason,
        "grounding": grounding,
        "grounding_status": grounding_status,
        "citation_validity": citation_validity,
        "latency_ms": result.latency_ms,
        "gold_doc_ids": gold_ids,
        "gold_doc_recall": gold_doc_recall,
        "errors": errors,
        "warnings": list(result.warnings),
        "failure_category": _failure_category(
            status=status,
            correctness=normalized,
            answer_suppressed=answer_suppressed,
            errors=errors,
            qtype=str(record.get("qtype") or ""),
        ),
        "model_calls": 0,
        "tokens": 0,
        "model_accounting": "deterministic_generator_no_llm",
        "failure_class": None,
        "tokens_unknown": False,
        "attempts": [],
        **specific,
    }
    usage = _model_usage(result)
    row["model_calls"] = usage["model_calls"]
    row["tokens"] = usage["tokens"]
    row["model_accounting"] = usage["model_accounting"]
    row["failure_class"] = usage["failure_class"]
    row["tokens_unknown"] = usage["tokens_unknown"]
    row["attempts"] = usage["attempts"]
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_system[str(row["system_name"])].append(row)
        by_family[str(row["qtype"])][str(row["system_name"])].append(row)
    return {
        "overall": {name: _system_stats(items) for name, items in _ordered_groups(by_system)},
        "by_qtype": {
            qtype: {name: _system_stats(items) for name, items in _ordered_groups(systems)}
            for qtype, systems in sorted(by_family.items())
        },
        "agentic": _agentic_behavior([row for row in rows if row["system_name"] == "agentic_graphrag"]),
        "failure_counts": {
            name: dict(Counter(row["failure_category"] for row in items if row["failure_category"]))
            for name, items in _ordered_groups(by_system)
        },
        "reliability": {
            name: _reliability_stats(items) for name, items in _ordered_groups(by_system)
        },
    }


def _ordered_groups(groups: dict[str, list[dict[str, Any]]]):
    names = [name for name in SYSTEM_ORDER if name in groups]
    names.extend(sorted(name for name in groups if name not in SYSTEM_ORDER))
    for name in names:
        yield name, groups[name]


def _system_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    latencies = [float(row["latency_ms"]) for row in rows]
    correctness = [row for row in rows if row["correctness"]]
    exact = [row for row in rows if row["correctness_exact"]]
    complete_values = [row["completeness"] for row in rows if row["completeness"] is not None]
    grounded = [row for row in rows if row["grounding"] is True]
    grounding_known = [row for row in rows if row["grounding"] is not None]
    citations_ok = [row for row in rows if row["citation_validity"] is True]
    system_errors = [row for row in rows if row["answer_status"] in ERROR_STATUSES]
    return {
        "n": n,
        "correctness": _rate(len(correctness), n),
        "correctness_exact": _rate(len(exact), n),
        "completeness": _rate(sum(1 for value in complete_values if value), len(complete_values))
        if complete_values
        else None,
        "completeness_n": len(complete_values),
        "grounding": _rate(len(grounded), len(grounding_known)) if grounding_known else None,
        "citation_validity": _rate(len(citations_ok), n),
        "system_errors": len(system_errors),
        "answer_suppressed": sum(1 for row in rows if row["answer_suppressed"]),
        "latency_avg_ms": round(mean(latencies), 3) if latencies else None,
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_max_ms": round(max(latencies), 3) if latencies else None,
        "status_counts": dict(Counter(row["answer_status"] for row in rows)),
        "failure_counts": dict(Counter(row["failure_category"] for row in rows if row["failure_category"])),
        "gold_doc_recall_avg": _mean_optional([row["gold_doc_recall"] for row in rows]),
        "retrieval_count_avg": _mean_optional([row["retrieval_count"] for row in rows]),
        "tool_calls_avg": _mean_optional([row["tool_calls"] for row in rows]),
        "agent_steps_avg": _mean_optional([row["agent_steps"] for row in rows]),
        "followups_avg": _mean_optional([row["followups"] for row in rows]),
    }


def _agentic_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["qtype"])].append(row)
    family = {}
    for qtype, items in sorted(by_type.items()):
        tools = [row["tool_calls"] for row in items if row["tool_calls"] is not None]
        family[qtype] = {
            "n": len(items),
            "stop_reasons": dict(Counter(row["stop_reason"] for row in items if row["stop_reason"])),
            "one_tool_call": sum(1 for row in items if row["tool_calls"] == 1),
            "used_neighborhood": sum(1 for row in items if _used_method(row, "event_neighborhood")),
            "used_temporal": sum(
                1
                for row in items
                if _used_method(row, "previous_event_gold") or _used_method(row, "next_event_gold")
            ),
            "followups": sum(1 for row in items if (row["followups"] or 0) > 0),
            "complete_set": sum(1 for row in items if row.get("cardinality") == "complete_set"),
            "truncated_set": sum(1 for row in items if row.get("truncated_set") is True),
            "tool_calls_avg": round(mean(tools), 3) if tools else None,
        }
    return {
        "stop_reasons": dict(Counter(row["stop_reason"] for row in rows if row["stop_reason"])),
        "one_tool_call": sum(1 for row in rows if row["tool_calls"] == 1),
        "followups": sum(1 for row in rows if (row["followups"] or 0) > 0),
        "neighborhood": sum(1 for row in rows if _used_method(row, "event_neighborhood")),
        "no_progress": sum(1 for row in rows if row["stop_reason"] == "no_progress"),
        "budget_exhausted": sum(1 for row in rows if row["stop_reason"] == "budget_exhausted"),
        "by_qtype": family,
    }


def _used_method(row: dict[str, Any], token: str) -> bool:
    methods = row.get("retrieval_methods") or []
    return any(token in str(method) for method in methods)


def _citation_validity(result: HarnessResult) -> bool:
    if result.status in {"invalid_citations", "error", "generation_error", "not_implemented"}:
        return False
    if any(error.startswith("orphan_citation") or error == "missing_citations" for error in result.errors):
        return False
    evidence_ids = {item.get("evidence_id") for item in result.evidence_metadata if item.get("evidence_id")}
    citation_ids = [citation.evidence_id for citation in result.citations]
    if (result.answer or "").strip() and not citation_ids:
        return False
    return all(evidence_id in evidence_ids for evidence_id in citation_ids)


def _grounding(
    result: HarnessResult,
    answer_suppressed: bool,
    citation_validity: bool,
) -> tuple[bool | None, str]:
    if result.status in {"error", "generation_error", "not_implemented"}:
        return False, "system_failure"
    if answer_suppressed:
        return True, "faithful_abstention"
    if citation_validity and result.citations:
        return True, "cited_retrieved_evidence"
    if not citation_validity:
        return False, "citation_invalid"
    return None, "semantic_grounding_not_evaluated"


def _completeness(record: dict[str, Any], result: HarnessResult) -> tuple[bool | None, str]:
    qtype = str(record.get("qtype") or "")
    if result.system_name == "rag":
        return None, "rag_top_k_has_no_complete_set_semantics"
    meta = result.retrieval_metadata or {}
    notes = _notes(meta)
    cardinality = meta.get("cardinality")
    truncated = bool(notes.get("truncated_set"))
    if qtype == "aggregation":
        if truncated:
            return False, "complete_set_truncated"
        if cardinality == "complete_set" or notes.get("complete_set"):
            return True, "complete_set_preserved"
        return False, "complete_set_not_marked"
    if truncated:
        return False, "set_truncated"
    return None, "complete_set_not_required"


def _model_usage(result: HarnessResult) -> dict[str, Any]:
    notes = _generator_notes(result)
    try:
        calls = int(notes.get("model_calls") or 0)
    except (TypeError, ValueError):
        calls = 0
    try:
        tokens = int(notes.get("tokens") or 0)
    except (TypeError, ValueError):
        tokens = 0
    accounting = notes.get("model_accounting")
    if not accounting:
        accounting = "deterministic_generator_no_llm" if calls == 0 else "provider_completion"
    failure_class = notes.get("failure_class")
    if failure_class is not None:
        failure_class = str(failure_class).strip() or None
    attempts = _compact_attempts(notes.get("attempts"))
    return {
        "model_calls": calls,
        "tokens": tokens,
        "model_accounting": str(accounting),
        "failure_class": failure_class,
        "tokens_unknown": bool(notes.get("tokens_unknown")),
        "attempts": attempts,
    }


def _generator_notes(result: HarnessResult) -> dict[str, Any]:
    meta = result.retrieval_metadata or {}
    notes: dict[str, Any] = {}
    pipeline_notes = meta.get("pipeline_notes")
    if isinstance(pipeline_notes, dict):
        notes.update(pipeline_notes)
    generator_notes = meta.get("generator_notes")
    if isinstance(generator_notes, dict):
        notes.update(generator_notes)
    return notes


def _compact_attempts(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        tokens = item.get("tokens")
        try:
            token_value = None if tokens is None else int(tokens)
        except (TypeError, ValueError):
            token_value = None
        compact.append(
            {
                "ok": bool(item.get("ok")),
                "provider_error": item.get("provider_error"),
                "tokens": token_value,
                "usage_known": bool(item.get("usage_known")),
            }
        )
    return compact


def _reliability_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generation_errors": sum(1 for row in rows if row.get("answer_status") == "generation_error"),
        "failure_class_counts": dict(
            Counter(str(row.get("failure_class")) for row in rows if row.get("failure_class"))
        ),
        "tokens_unknown": sum(1 for row in rows if row.get("tokens_unknown")),
        "retried": sum(1 for row in rows if int(row.get("model_calls") or 0) > 1),
        "tokens_total": int(sum(int(row.get("tokens") or 0) for row in rows)),
        "model_calls_total": int(sum(int(row.get("model_calls") or 0) for row in rows)),
    }


def _notes(meta: dict[str, Any]) -> dict[str, Any]:
    notes = meta.get("notes") if isinstance(meta.get("notes"), dict) else {}
    return notes


def _retrieved_ids(result: HarnessResult) -> set[str]:
    ids: set[str] = set()
    meta = result.retrieval_metadata or {}
    for event_id in meta.get("event_ids") or []:
        ids.add(str(event_id))
    for item in result.evidence_metadata:
        for key in ("event_id", "document_id"):
            value = item.get(key)
            if value:
                ids.add(str(value))
    return ids


def _system_specific(result: HarnessResult) -> dict[str, Any]:
    meta = result.retrieval_metadata or {}
    notes = _notes(meta)
    pipeline_notes = meta.get("pipeline_notes") if isinstance(meta.get("pipeline_notes"), dict) else {}
    trace = meta.get("trace") if isinstance(meta.get("trace"), dict) else None
    evidence_count = len(result.evidence_metadata)
    citation_count = len(result.citations)
    base = {
        "retrieval_count": None,
        "evidence_count": evidence_count,
        "citation_count": citation_count,
        "tool_calls": None,
        "agent_steps": None,
        "followups": None,
        "stop_reason": None,
        "retrieval_methods": None,
        "cardinality": meta.get("cardinality"),
        "truncated_set": notes.get("truncated_set") if "truncated_set" in notes else None,
        "trace": None,
    }
    if result.system_name == "rag":
        methods = [meta["method"]] if meta.get("method") else None
        hit_count = meta.get("hit_count")
        base.update(
            {
                "retrieval_count": 1 if result.status != "error" else None,
                "evidence_count": hit_count if hit_count is not None else evidence_count,
                "retrieval_methods": methods,
            }
        )
        return base
    if result.system_name == "graphrag":
        method = meta.get("retrieval_method")
        base.update(
            {
                "retrieval_count": pipeline_notes.get("graph_retriever_calls", 1 if result.status != "error" else None),
                "followups": pipeline_notes.get("followup_retrievals"),
                "retrieval_methods": [method] if method else None,
            }
        )
        return base
    if result.system_name == "agentic_graphrag":
        methods = None
        if trace and trace.get("retrieval_methods") is not None:
            methods = list(trace.get("retrieval_methods") or [])
        elif meta.get("retrieval_method"):
            methods = [meta["retrieval_method"]]
        tool_calls = meta.get("total_tool_calls")
        if tool_calls is None and trace is not None:
            tool_calls = trace.get("total_tool_calls")
        steps = trace.get("total_steps") if trace else None
        followups = pipeline_notes.get("followup_retrievals")
        if followups is None and trace is not None:
            followups = sum(1 for item in trace.get("plan_steps") or [] if item.get("kind") == "follow_up")
        base.update(
            {
                "retrieval_count": tool_calls,
                "tool_calls": tool_calls,
                "agent_steps": steps,
                "followups": followups,
                "stop_reason": meta.get("stop_reason") or (trace or {}).get("stop_reason"),
                "retrieval_methods": methods,
                "trace": _compact_trace(trace) if trace else None,
            }
        )
        return base
    return base


def _compact_trace(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "stop_reason": trace.get("stop_reason"),
        "total_steps": trace.get("total_steps"),
        "total_tool_calls": trace.get("total_tool_calls"),
        "plan_reasons": [step.get("reason") for step in trace.get("plan_steps") or []],
        "tools": [
            {
                "tool": item.get("tool"),
                "success": item.get("success"),
                "result_status": item.get("result_status"),
            }
            for item in trace.get("tool_calls") or []
        ],
        "follow_up_decisions": list(trace.get("follow_up_decisions") or []),
        "strategy_changes": list(trace.get("strategy_changes") or []),
        "retrieval_methods": list(trace.get("retrieval_methods") or []),
    }


def _failure_category(
    *,
    status: str,
    correctness: bool,
    answer_suppressed: bool,
    errors: list[str],
    qtype: str,
) -> str | None:
    if correctness:
        return None
    if status == "error" or any(item.startswith("system_error") or item.startswith("unknown_system") for item in errors):
        return "system_failure"
    if status == "invalid_citations":
        return "citation_failure"
    if status == "generation_error":
        return "generation_failure"
    if status == "ambiguous":
        return "ambiguous_suppressed"
    if status in {"not_found", "unresolved", "unsupported"}:
        return status
    if status == "abstained" or answer_suppressed:
        return "abstained"
    if qtype == "aggregation":
        return "incorrect_or_incomplete"
    return "incorrect"


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(100.0 * numerator / denominator, 1)


def _mean_optional(values: list[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return round(mean(present), 3)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((percentile / 100.0) * (len(ordered) - 1))))
    return round(ordered[index], 3)


def git_revision() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return output.strip() or None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def failure_table(rows: list[dict[str, Any]], *, limit: int = 80) -> list[dict[str, Any]]:
    table = []
    for row in rows:
        if row["correctness"] and not row["failure_category"]:
            continue
        table.append(
            {
                "qid": row["qid"],
                "qtype": row["qtype"],
                "system_name": row["system_name"],
                "expected": truncate(row.get("expected")),
                "predicted": truncate(row.get("answer")),
                "status": row["answer_status"],
                "failure_category": row["failure_category"],
                "failure_class": row.get("failure_class"),
                "stop_reason": row.get("stop_reason"),
            }
        )
        if len(table) >= limit:
            break
    return table
