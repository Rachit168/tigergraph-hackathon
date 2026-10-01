"""Public three-way benchmark over RAG, Fixed GraphRAG, and Agentic GraphRAG."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evaluation.benchmark import (
    SYSTEM_ORDER,
    failure_table,
    public_run_id,
    score_harness_result,
    summarize_rows,
)
from evaluation.harness import (
    AgenticGraphRAGAdapter,
    AgenticGraphRAGPlaceholder,
    GraphRAGAdapter,
    RAGAdapter,
    ThreeWayEvaluationHarness,
)
from ingestion.loader import load_questions
from ingestion.paths import DEFAULT_PUBLIC_QUESTIONS_PATH, REPO_ROOT

DEFAULT_OUTPUT_DIR = REPO_ROOT / "_research" / "phase8_public_three_way"
CSV_COLUMNS = (
    "qid",
    "qtype",
    "system_name",
    "answer_status",
    "answer_suppressed",
    "correctness",
    "correctness_exact",
    "completeness",
    "completeness_reason",
    "grounding",
    "grounding_status",
    "citation_validity",
    "latency_ms",
    "retrieval_count",
    "evidence_count",
    "citation_count",
    "tool_calls",
    "agent_steps",
    "followups",
    "stop_reason",
    "retrieval_methods",
    "gold_doc_recall",
    "cardinality",
    "truncated_set",
    "failure_category",
    "model_calls",
    "tokens",
    "tokens_unknown",
    "failure_class",
    "attempts",
    "errors",
    "answer",
)


def load_public_questions(path: str | Path | None = None) -> list[dict[str, Any]]:
    questions_path = Path(path) if path is not None else DEFAULT_PUBLIC_QUESTIONS_PATH
    records = load_questions(questions_path)
    records.sort(key=lambda row: str(row.get("qid") or ""))
    return records


def make_three_way_harness(
    *,
    rag: RAGAdapter,
    graphrag: GraphRAGAdapter,
    agentic: AgenticGraphRAGAdapter,
) -> ThreeWayEvaluationHarness:
    if isinstance(agentic, AgenticGraphRAGPlaceholder):
        raise TypeError("public benchmark requires AgenticGraphRAGAdapter, not the Phase 6 placeholder")
    if not isinstance(rag, RAGAdapter):
        raise TypeError("rag must be RAGAdapter")
    if not isinstance(graphrag, GraphRAGAdapter):
        raise TypeError("graphrag must be GraphRAGAdapter")
    if not isinstance(agentic, AgenticGraphRAGAdapter):
        raise TypeError("agentic must be AgenticGraphRAGAdapter")
    return ThreeWayEvaluationHarness([rag, graphrag, agentic])


def evaluate_three_way(
    records: list[dict[str, Any]],
    harness: ThreeWayEvaluationHarness,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    names = [name for name in SYSTEM_ORDER if name in harness.systems]
    run_config = {
        "systems": names,
        "n_questions": len(records),
        "qids": [record.get("qid") for record in records],
        **dict(config or {}),
    }
    run_config.pop("generated_at", None)
    rows: list[dict[str, Any]] = []
    for record in records:
        results = harness.run_question(
            str(record.get("question") or ""),
            qtype=record.get("qtype"),
            system_names=names,
        )
        for result in results:
            rows.append(score_harness_result(record, result))
    payload = {
        "run_id": public_run_id(
            {
                "systems": names,
                "n_questions": len(records),
                "qids": [record.get("qid") for record in records],
                "questions_name": run_config.get("questions_name"),
                "rag_method": run_config.get("rag_method"),
                "generator": run_config.get("generator"),
            }
        ),
        "config": run_config,
        "n_questions": len(records),
        "n_rows": len(rows),
        "rows": rows,
        "summary": summarize_rows(rows),
        "failures": failure_table(rows),
    }
    return payload


def write_benchmark_artifacts(payload: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    report_path = output_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow(_csv_row(row))
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "report": report_path}


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    overall = summary["overall"]
    lines: list[str] = []
    add = lines.append
    add("# Phase 8: Public three-way benchmark")
    add("")
    add("Diagnostic comparison of RAG, Fixed GraphRAG, and Agentic GraphRAG on the 100 public questions.")
    add("This report does not declare a single winner.")
    add("")
    add(f"- Run id: `{payload['run_id']}`")
    add(f"- Questions: {payload['n_questions']}")
    add(f"- Rows: {payload['n_rows']}")
    add(f"- Systems: {', '.join(payload['config'].get('systems') or SYSTEM_ORDER)}")
    add("- Gold answers are used only in this evaluation layer.")
    generator_name = payload["config"].get("generator") or "DeterministicGroundedGenerator"
    add(f"- Generator: `{generator_name}`.")
    add("")
    add("## 1. Overall comparison")
    add("")
    add(_overall_table(overall))
    add("")
    add("## 2. Results by question family")
    add("")
    add(_family_table(summary.get("by_qtype") or {}))
    add("")
    add("## 3. Correctness")
    add("")
    add("Correctness is normalized gold-answer match after the system has produced an answer.")
    add("Empty or suppressed answers are incorrect, even when abstention is the right safety behavior.")
    add("")
    add(_metric_by_family(summary.get("by_qtype") or {}, "correctness"))
    add("")
    add("## 4. Completeness")
    add("")
    add("Complete-set completeness is scored only for GraphRAG/Agentic aggregation.")
    add("RAG top-k retrieval has no complete-set semantics; those cells are null.")
    add("")
    add(_metric_by_family(summary.get("by_qtype") or {}, "completeness"))
    add("")
    add("## 5. Grounding")
    add("")
    add("Grounding is `cited_retrieved_evidence` when an answer cites retrieved evidence,")
    add("or `faithful_abstention` when the system returns no answer.")
    add("Chunk-prose semantic entailment is not scored.")
    add("")
    add(_metric_by_family(summary.get("by_qtype") or {}, "grounding"))
    add("")
    add("## 6. Citation validity")
    add("")
    add("Fail-closed: orphan citations and unanswered claims with no citations are invalid.")
    add("Empty abstentions with no citations are valid.")
    add("")
    add(_metric_by_family(summary.get("by_qtype") or {}, "citation_validity"))
    add("")
    add("## 7. Latency")
    add("")
    add(_latency_table(overall))
    add("")
    add("## 8. Tool / step behavior")
    add("")
    add(_tool_table(overall, summary.get("agentic") or {}))
    add("")
    add("## 9. Failure categories")
    add("")
    add(_failure_section(summary.get("failure_counts") or {}, payload.get("failures") or []))
    add("")
    add("## 10. Agentic stop reasons")
    add("")
    add(_agentic_section(summary.get("agentic") or {}))
    add("")
    add("## 11. Generation reliability")
    add("")
    add("Shared SemanticGenerator metadata. Does not change correctness or gold scoring.")
    add("Tokens are the sum of known usage across every provider attempt; `tokens_unknown` means at least one attempt lacked usage.")
    add("")
    add(_reliability_section(summary.get("reliability") or {}))
    add("")
    add("## Notes")
    add("")
    add("- Fixed GraphRAG remains one-pass. Agentic GraphRAG may add bounded follow-ups.")
    add("- RAG, GraphRAG, and Agentic share the same generator interface and citation validator.")
    add("- RAG remains text-only and does not receive graph retrieval.")
    add("")
    return "\n".join(lines)


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for key in CSV_COLUMNS:
        value = row.get(key)
        if value is None:
            payload[key] = ""
        elif isinstance(value, (list, dict)):
            payload[key] = json.dumps(value, sort_keys=True, default=str)
        elif isinstance(value, bool):
            payload[key] = "true" if value else "false"
        else:
            payload[key] = value
    return payload


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return str(value)


def _overall_table(overall: dict[str, Any]) -> str:
    lines = [
        "| System | n | correctness % | exact % | completeness % | grounding % | citation % | errors | avg ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SYSTEM_ORDER:
        stats = overall.get(name)
        if not stats:
            continue
        lines.append(
            f"| {name} | {stats['n']} | {_fmt(stats['correctness'])} | {_fmt(stats['correctness_exact'])} | "
            f"{_fmt(stats['completeness'])} | {_fmt(stats['grounding'])} | {_fmt(stats['citation_validity'])} | "
            f"{stats['system_errors']} | {_fmt(stats['latency_avg_ms'])} | {_fmt(stats['latency_p95_ms'])} |"
        )
    return "\n".join(lines)


def _family_table(by_qtype: dict[str, Any]) -> str:
    lines = [
        "| qtype | system | n | correctness % | completeness % | citation % | avg tools | stop / status notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for qtype, systems in sorted(by_qtype.items()):
        for name in SYSTEM_ORDER:
            stats = systems.get(name)
            if not stats:
                continue
            status = json.dumps(stats.get("status_counts") or {}, sort_keys=True)
            lines.append(
                f"| {qtype} | {name} | {stats['n']} | {_fmt(stats['correctness'])} | "
                f"{_fmt(stats['completeness'])} | {_fmt(stats['citation_validity'])} | "
                f"{_fmt(stats['tool_calls_avg'])} | `{status}` |"
            )
    return "\n".join(lines)


def _metric_by_family(by_qtype: dict[str, Any], metric: str) -> str:
    lines = ["| qtype | rag | graphrag | agentic_graphrag |", "|---|---:|---:|---:|"]
    for qtype, systems in sorted(by_qtype.items()):
        cells = [_fmt((systems.get(name) or {}).get(metric)) for name in SYSTEM_ORDER]
        lines.append(f"| {qtype} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(lines)


def _latency_table(overall: dict[str, Any]) -> str:
    lines = [
        "| System | avg ms | p95 ms | max ms |",
        "|---|---:|---:|---:|",
    ]
    for name in SYSTEM_ORDER:
        stats = overall.get(name)
        if not stats:
            continue
        lines.append(
            f"| {name} | {_fmt(stats['latency_avg_ms'])} | {_fmt(stats['latency_p95_ms'])} | {_fmt(stats['latency_max_ms'])} |"
        )
    return "\n".join(lines)


def _tool_table(overall: dict[str, Any], agentic: dict[str, Any]) -> str:
    lines = [
        "| System | retrieval_count avg | tool_calls avg | agent_steps avg | followups avg |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in SYSTEM_ORDER:
        stats = overall.get(name) or {}
        lines.append(
            f"| {name} | {_fmt(stats.get('retrieval_count_avg'))} | "
            f"{_fmt(stats.get('tool_calls_avg'))} | {_fmt(stats.get('agent_steps_avg'))} | {_fmt(stats.get('followups_avg'))} |"
        )
    lines.append("")
    lines.append(
        f"Agentic one-tool-call questions: {agentic.get('one_tool_call', 0)}. "
        f"Questions with follow-ups: {agentic.get('followups', 0)}. "
        f"Neighborhood uses: {agentic.get('neighborhood', 0)}."
    )
    return "\n".join(lines)


def _failure_section(failure_counts: dict[str, Any], failures: list[dict[str, Any]]) -> str:
    lines = ["Per-system failure category counts:", ""]
    lines.append(f"`{json.dumps(failure_counts, sort_keys=True)}`")
    lines.append("")
    if not failures:
        lines.append("No compact failure rows.")
        return "\n".join(lines)
    lines.append("| qid | qtype | system | expected | predicted | status | category | gen class | stop |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in failures[:60]:
        lines.append(
            f"| {row['qid']} | {row['qtype']} | {row['system_name']} | {row['expected']} | "
            f"{row['predicted']} | {row['status']} | {row['failure_category']} | "
            f"{row.get('failure_class') or ''} | {row.get('stop_reason') or ''} |"
        )
    return "\n".join(lines)


def _reliability_section(reliability: dict[str, Any]) -> str:
    lines = [
        "| System | gen errors | retried | tokens unknown | tokens total | model calls | failure_class counts |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in SYSTEM_ORDER:
        stats = reliability.get(name)
        if not stats:
            continue
        lines.append(
            f"| {name} | {stats.get('generation_errors', 0)} | {stats.get('retried', 0)} | "
            f"{stats.get('tokens_unknown', 0)} | {_fmt(stats.get('tokens_total'))} | "
            f"{_fmt(stats.get('model_calls_total'))} | "
            f"`{json.dumps(stats.get('failure_class_counts') or {}, sort_keys=True)}` |"
        )
    if len(lines) == 2:
        return "No reliability rows."
    return "\n".join(lines)


def _agentic_section(agentic: dict[str, Any]) -> str:
    if not agentic:
        return "No Agentic rows."
    lines = [
        f"Stop reasons: `{json.dumps(agentic.get('stop_reasons') or {}, sort_keys=True)}`",
        "",
        f"- no_progress: {agentic.get('no_progress', 0)}",
        f"- budget_exhausted: {agentic.get('budget_exhausted', 0)}",
        "",
        "| qtype | n | one tool | follow-ups | neighborhood | temporal query | complete_set | truncated | avg tools | stop reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for qtype, stats in (agentic.get("by_qtype") or {}).items():
        lines.append(
            f"| {qtype} | {stats['n']} | {stats['one_tool_call']} | {stats['followups']} | "
            f"{stats['used_neighborhood']} | {stats['used_temporal']} | {stats['complete_set']} | "
            f"{stats['truncated_set']} | {_fmt(stats['tool_calls_avg'])} | "
            f"`{json.dumps(stats['stop_reasons'], sort_keys=True)}` |"
        )
    return "\n".join(lines)
