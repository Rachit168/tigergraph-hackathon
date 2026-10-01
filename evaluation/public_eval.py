"""Evaluate the offline structured solver on the public question set."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any

from evaluation.normalize import answers_match_exact, answers_match_normalized, truncate
from ingestion.loader import load_questions, parse_corpus
from ingestion.paths import DEFAULT_CORPUS_PATH, DEFAULT_PUBLIC_QUESTIONS_PATH
from retrieval.structured.index import StructuredIndex
from retrieval.structured.solver import StructuredSolver


def evaluate_public(
    corpus_path=DEFAULT_CORPUS_PATH,
    questions_path=DEFAULT_PUBLIC_QUESTIONS_PATH,
) -> dict[str, Any]:
    load_started = time.perf_counter()
    corpus = parse_corpus(corpus_path)
    index = StructuredIndex(corpus)
    solver = StructuredSolver(index)
    questions = load_questions(questions_path)
    load_ms = (time.perf_counter() - load_started) * 1000.0

    rows: list[dict[str, Any]] = []
    eval_started = time.perf_counter()
    for question in questions:
        result = solver.solve(question["question"], qtype=question.get("qtype"))
        gold = question.get("answer")
        exact = answers_match_exact(result.answer, gold)
        normalized = answers_match_normalized(result.answer, gold)
        gold_ids = list(question.get("gold_doc_ids") or [])
        predicted_ids = [event.event_id for event in result.events]
        row = {
            "qid": question.get("qid"),
            "qtype": question.get("qtype"),
            "status": result.status,
            "reason": result.reason,
            "method": result.method,
            "exact": exact,
            "normalized": normalized,
            "expected": gold,
            "predicted": result.answer,
            "elapsed_ms": result.elapsed_ms,
            "event_ids": predicted_ids,
            "gold_doc_ids": gold_ids,
            "notes": result.notes,
            "limitation": _classify_limitation(result, gold_ids, exact),
            "prev_event_reasoning": result.spec.qtype == "temporal",
            "missing_field": bool(result.reason and result.reason.startswith("missing_field")),
            "multiple_events": result.status == "ambiguous",
            "hit_count": len(predicted_ids),
        }
        rows.append(row)
    eval_ms = (time.perf_counter() - eval_started) * 1000.0
    return {
        "n": len(rows),
        "load_ms": load_ms,
        "eval_ms": eval_ms,
        "index_events": len(index.events),
        "index_documents": len(index.documents),
        "summary": _summarize(rows, eval_ms),
        "rows": rows,
    }


def _classify_limitation(result, gold_ids: list[str], exact: bool) -> str | None:
    if exact:
        return None
    notes = result.notes or {}
    if notes.get("limitation") == "data" or (result.reason or "").startswith("missing_field"):
        return "data"
    if result.status == "ambiguous" and gold_ids:
        if set(gold_ids) <= set(event.event_id for event in result.events) or any(
            doc_id in {event.event_id for event in result.events} for doc_id in gold_ids
        ):
            return "data"
        return "solver"
    if result.status in {"unresolved", "not_found"}:
        if result.reason in {
            "previous_event_name_missing",
            "next_event_name_missing",
            "related_year_unresolved",
            "no_sport_games_events",
            "venue_not_found",
            "date_does_not_overlap",
        }:
            return "data"
        if result.reason == "question_template_unparsed":
            return "solver"
        return "solver"
    if result.status == "supported" and not exact:
        return "solver"
    return "solver"


def _summarize(rows: list[dict[str, Any]], eval_ms: float) -> dict[str, Any]:
    n = len(rows) or 1
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["qtype"])].append(row)
    latencies = [row["elapsed_ms"] for row in rows]
    return {
        "exact_accuracy": round(100.0 * sum(1 for row in rows if row["exact"]) / n, 1),
        "normalized_accuracy": round(100.0 * sum(1 for row in rows if row["normalized"]) / n, 1),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "limitation_counts": dict(Counter(row["limitation"] for row in rows if row["limitation"])),
        "reason_counts": dict(Counter(row["reason"] for row in rows if row["reason"])),
        "prev_event_questions": sum(1 for row in rows if row["prev_event_reasoning"]),
        "missing_field_questions": sum(1 for row in rows if row["missing_field"]),
        "multiple_event_questions": sum(1 for row in rows if row["multiple_events"]),
        "avg_query_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "max_query_ms": round(max(latencies), 3) if latencies else 0.0,
        "total_query_ms": round(eval_ms, 3),
        "by_qtype": {qtype: _type_stats(items) for qtype, items in sorted(by_type.items())},
    }


def _type_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    counts = Counter(row["status"] for row in rows)
    return {
        "n": len(rows),
        "solved": counts.get("supported", 0),
        "unresolved": counts.get("unresolved", 0),
        "ambiguous": counts.get("ambiguous", 0),
        "not_found": counts.get("not_found", 0),
        "exact_accuracy": round(100.0 * sum(1 for row in rows if row["exact"]) / n, 1),
        "normalized_accuracy": round(100.0 * sum(1 for row in rows if row["normalized"]) / n, 1),
        "limitation_counts": dict(Counter(row["limitation"] for row in rows if row["limitation"])),
    }


def failure_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        if row["exact"] and row["normalized"]:
            continue
        failures.append(
            {
                "qid": row["qid"],
                "qtype": row["qtype"],
                "expected": truncate(row["expected"]),
                "predicted": truncate(row["predicted"]),
                "status": row["status"],
                "reason": row["reason"] or "",
                "limitation": row["limitation"],
            }
        )
    return failures
