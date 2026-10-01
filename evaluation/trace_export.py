"""Submission-safe observable trace export. Gold and scores never enter this schema."""

from __future__ import annotations

import copy
import json
from typing import Any

from answering.models import Citation, PipelineTimings
from evaluation.harness import HarnessResult
from retrieval.agentic.pipeline import AgentResult
from retrieval.agentic.trace import AgentTrace

SCHEMA_VERSION = 1

# Evaluation-only keys. Never copy these into a submission payload.
EXCLUDED_KEYS = frozenset(
    {
        "expected",
        "gold",
        "gold_answer",
        "gold_answers",
        "gold_doc_ids",
        "gold_doc_recall",
        "gold_source",
        "correctness",
        "correctness_exact",
        "completeness",
        "completeness_reason",
        "grounding",
        "grounding_status",
        "citation_validity",
        "failure_category",
        "acceptable_answers",
        "expected_behavior",
        "answer_verified",
        "holdout_version",
        "robustness_category",
    }
)


def export_submission_record(
    result: HarnessResult | AgentResult | AgentTrace | dict[str, Any],
    *,
    question_id: str | None = None,
    pipeline: str | None = None,
) -> dict[str, Any]:
    """Build a versioned, JSON-safe record of system-observable behavior.

    Accepts a harness result, agent pipeline result, raw AgentTrace, or a scored
    dict. Evaluation-only fields are dropped. The input object is not mutated.
    """

    payload = _from_any(result, question_id=question_id, pipeline=pipeline)
    sanitized = _strip_excluded(payload)
    json.dumps(sanitized, default=str, sort_keys=True)
    return sanitized


def _from_any(
    result: HarnessResult | AgentResult | AgentTrace | dict[str, Any],
    *,
    question_id: str | None,
    pipeline: str | None,
) -> dict[str, Any]:
    if isinstance(result, AgentTrace):
        return _record(
            question_id=question_id,
            pipeline=pipeline or "agentic_graphrag",
            question=result.question,
            answer="",
            status="",
            latency_ms=_trace_latency(result),
            timings=dict(result.timings),
            citations=[],
            evidence=[],
            retrieval_methods=list(result.retrieval_methods),
            notes={},
            agent_trace=_agent_trace_payload(result),
        )
    if isinstance(result, AgentResult):
        harness = _harness_from_agent_result(result)
        return _from_harness(harness, question_id=question_id, pipeline=pipeline or harness.system_name)
    if isinstance(result, HarnessResult):
        return _from_harness(result, question_id=question_id, pipeline=pipeline)
    if isinstance(result, dict):
        return _from_mapping(result, question_id=question_id, pipeline=pipeline)
    raise TypeError(f"unsupported export source: {type(result)!r}")


def _from_harness(
    result: HarnessResult,
    *,
    question_id: str | None,
    pipeline: str | None,
) -> dict[str, Any]:
    meta = copy.deepcopy(dict(result.retrieval_metadata or {}))
    evidence = copy.deepcopy(list(result.evidence_metadata or []))
    notes = _generator_notes(meta)
    raw_trace = meta.get("trace")
    agent_trace = None
    if isinstance(raw_trace, dict) and raw_trace:
        agent_trace = _agent_trace_from_mapping(raw_trace)
    retrieval_methods = _retrieval_methods(meta, agent_trace)
    timings = result.timings.to_dict() if isinstance(result.timings, PipelineTimings) else dict(result.timings or {})
    return _record(
        question_id=question_id,
        pipeline=pipeline or result.system_name,
        question=result.question,
        answer=result.answer,
        status=str(result.status),
        latency_ms=float(result.latency_ms),
        timings=timings,
        citations=list(result.citations),
        evidence=evidence,
        retrieval_methods=retrieval_methods,
        notes=notes,
        agent_trace=agent_trace,
        warnings=list(result.warnings),
        errors=list(result.errors),
    )


def _from_mapping(
    row: dict[str, Any],
    *,
    question_id: str | None,
    pipeline: str | None,
) -> dict[str, Any]:
    cloned = copy.deepcopy(row)
    meta = cloned.get("retrieval_metadata") if isinstance(cloned.get("retrieval_metadata"), dict) else {}
    notes = _generator_notes(meta)
    if not notes:
        notes = {
            "model_calls": cloned.get("model_calls"),
            "tokens": cloned.get("tokens"),
            "tokens_unknown": cloned.get("tokens_unknown"),
            "failure_class": cloned.get("failure_class"),
            "attempts": cloned.get("attempts"),
            "model_accounting": cloned.get("model_accounting"),
        }
    raw_trace = cloned.get("trace")
    if not isinstance(raw_trace, dict):
        raw_trace = meta.get("trace") if isinstance(meta.get("trace"), dict) else None
    agent_trace = _agent_trace_from_mapping(raw_trace) if isinstance(raw_trace, dict) and raw_trace else None
    citations = cloned.get("citations") or []
    evidence = cloned.get("evidence_metadata") or []
    timings = cloned.get("timings") if isinstance(cloned.get("timings"), dict) else {}
    methods = cloned.get("retrieval_methods")
    if methods is None:
        methods = _retrieval_methods(meta, agent_trace)
    return _record(
        question_id=question_id or cloned.get("qid") or cloned.get("question_id"),
        pipeline=pipeline or cloned.get("system_name") or cloned.get("pipeline"),
        question=str(cloned.get("question") or ""),
        answer=str(cloned.get("answer") or ""),
        status=str(cloned.get("answer_status") or cloned.get("status") or ""),
        latency_ms=float(cloned.get("latency_ms") or 0.0),
        timings=dict(timings),
        citations=citations,
        evidence=evidence if isinstance(evidence, list) else [],
        retrieval_methods=list(methods or []),
        notes=notes,
        agent_trace=agent_trace,
        warnings=list(cloned.get("warnings") or []),
        errors=list(cloned.get("errors") or []),
    )


def _harness_from_agent_result(result: AgentResult) -> HarnessResult:
    answer = result.answer
    metadata = answer.retrieval_result.to_dict()
    metadata["pipeline_notes"] = dict(answer.notes)
    metadata["trace"] = result.trace.to_dict()
    metadata["stop_reason"] = result.trace.stop_reason
    metadata["total_tool_calls"] = result.trace.total_tool_calls
    return HarnessResult(
        system_name="agentic_graphrag",
        question=answer.question,
        answer=answer.answer_text,
        citations=list(answer.citations),
        latency_ms=answer.timings.total_ms,
        retrieval_metadata=metadata,
        evidence_metadata=[item.to_dict() for item in answer.retrieval_result.all_evidence()],
        status=str(answer.status),
        warnings=list(answer.warnings),
        timings=answer.timings,
    )


def _record(
    *,
    question_id: str | None,
    pipeline: str | None,
    question: str,
    answer: str,
    status: str,
    latency_ms: float,
    timings: dict[str, Any],
    citations: list[Any],
    evidence: list[dict[str, Any]],
    retrieval_methods: list[str],
    notes: dict[str, Any],
    agent_trace: dict[str, Any] | None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    citation_payload = [_citation_payload(item) for item in citations]
    citation_ids = [item["evidence_id"] for item in citation_payload if item.get("evidence_id")]
    evidence_ids, chunk_ids, document_ids, event_ids = _observable_ids(evidence, citation_payload, agent_trace)
    tokens = _int(notes.get("tokens"), default=0)
    calls = _int(notes.get("model_calls"), default=0)
    attempts = notes.get("attempts") if isinstance(notes.get("attempts"), list) else []
    return {
        "schema_version": SCHEMA_VERSION,
        "question_id": question_id,
        "question": question,
        "pipeline": pipeline,
        "answer": answer,
        "status": status,
        "citation_ids": citation_ids,
        "citations": citation_payload,
        "total_tokens": tokens,
        "model_calls": calls,
        "tokens_unknown": bool(notes.get("tokens_unknown")),
        "failure_class": notes.get("failure_class") or None,
        "attempts": copy.deepcopy(attempts),
        "model_accounting": notes.get("model_accounting") or None,
        "latency_ms": latency_ms,
        "timings": dict(timings),
        "evidence_ids": evidence_ids,
        "chunk_ids": chunk_ids,
        "document_ids": document_ids,
        "event_ids": event_ids,
        "retrieval_methods": list(retrieval_methods),
        "warnings": list(warnings or []),
        "errors": list(errors or []),
        "agent_trace": agent_trace,
    }


def _agent_trace_payload(trace: AgentTrace) -> dict[str, Any]:
    return _agent_trace_from_mapping(trace.to_dict())


def _agent_trace_from_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    tools = []
    observations = []
    for item in raw.get("tool_calls") or []:
        if not isinstance(item, dict):
            continue
        observation = {
            "tool_call_id": item.get("tool_call_id"),
            "tool": item.get("tool"),
            "arguments": copy.deepcopy(item.get("arguments") or {}),
            "reason": item.get("reason"),
            "success": item.get("success"),
            "result_status": item.get("result_status"),
            "elapsed_ms": item.get("elapsed_ms"),
            "started_ms": item.get("started_ms"),
            "retrieval_method": item.get("retrieval_method"),
            "evidence_ids": list(item.get("evidence_ids") or []),
            "event_ids": list(item.get("event_ids") or []),
            "error": item.get("error"),
            "observation_truncated": bool(item.get("observation_truncated")),
            "parallel_group": item.get("parallel_group"),
        }
        observations.append(observation)
        tools.append(
            {
                "tool": observation["tool"],
                "arguments": copy.deepcopy(observation["arguments"]),
                "success": observation["success"],
                "result_status": observation["result_status"],
                "elapsed_ms": observation["elapsed_ms"],
                "retrieval_method": observation["retrieval_method"],
                "evidence_ids": list(observation["evidence_ids"]),
            }
        )
    plan_steps = []
    for step in raw.get("plan_steps") or []:
        if not isinstance(step, dict):
            continue
        plan_steps.append(
            {
                "iteration": step.get("iteration"),
                "kind": step.get("kind"),
                "reason": step.get("reason"),
                "actions": copy.deepcopy(step.get("actions") or []),
            }
        )
    return {
        "interpreted": copy.deepcopy(raw.get("interpreted") or {}),
        "plan_steps": plan_steps,
        "tools": tools,
        "tool_observations": observations,
        "slot_updates": copy.deepcopy(raw.get("slot_updates") or []),
        "follow_up_decisions": list(raw.get("follow_up_decisions") or []),
        "strategy_changes": list(raw.get("strategy_changes") or []),
        "stop_reason": raw.get("stop_reason"),
        "total_tool_calls": _int(raw.get("total_tool_calls"), default=len(tools)),
        "total_steps": _int(raw.get("total_steps"), default=len(plan_steps)),
        "retrieval_methods": list(raw.get("retrieval_methods") or []),
        "timings": copy.deepcopy(raw.get("timings") or {}),
        "observation_chars": _int(raw.get("observation_chars"), default=0),
        "parallel_groups": list(raw.get("parallel_groups") or []),
    }


def _citation_payload(item: Any) -> dict[str, Any]:
    if isinstance(item, Citation):
        payload = item.to_dict()
    elif isinstance(item, dict):
        payload = dict(item)
    else:
        payload = {"evidence_id": str(item)}
    return {
        "evidence_id": payload.get("evidence_id"),
        "document_id": payload.get("document_id"),
        "chunk_id": payload.get("chunk_id"),
        "event_id": payload.get("event_id"),
        "source_url": payload.get("source_url"),
        "source_date": payload.get("source_date"),
    }


def _observable_ids(
    evidence: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    agent_trace: dict[str, Any] | None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    evidence_ids: list[str] = []
    chunk_ids: list[str] = []
    document_ids: list[str] = []
    event_ids: list[str] = []

    def add(bucket: list[str], value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in bucket:
            bucket.append(text)

    for item in list(evidence) + list(citations):
        if not isinstance(item, dict):
            continue
        add(evidence_ids, item.get("evidence_id"))
        add(chunk_ids, item.get("chunk_id") or item.get("source_chunk_id"))
        add(document_ids, item.get("document_id"))
        add(event_ids, item.get("event_id"))
        for event_id in item.get("event_ids") or []:
            add(event_ids, event_id)
    if agent_trace:
        for observation in agent_trace.get("tool_observations") or []:
            for evidence_id in observation.get("evidence_ids") or []:
                add(evidence_ids, evidence_id)
            for event_id in observation.get("event_ids") or []:
                add(event_ids, event_id)
        for update in agent_trace.get("slot_updates") or []:
            if not isinstance(update, dict):
                continue
            for evidence_id in update.get("evidence_ids") or []:
                add(evidence_ids, evidence_id)
    return evidence_ids, chunk_ids, document_ids, event_ids


def _generator_notes(meta: dict[str, Any]) -> dict[str, Any]:
    notes: dict[str, Any] = {}
    pipeline_notes = meta.get("pipeline_notes")
    if isinstance(pipeline_notes, dict):
        notes.update(pipeline_notes)
    generator_notes = meta.get("generator_notes")
    if isinstance(generator_notes, dict):
        notes.update(generator_notes)
    return notes


def _retrieval_methods(meta: dict[str, Any], agent_trace: dict[str, Any] | None) -> list[str]:
    if agent_trace and agent_trace.get("retrieval_methods"):
        return list(agent_trace["retrieval_methods"])
    if meta.get("retrieval_methods"):
        return list(meta["retrieval_methods"])
    if meta.get("method"):
        return [str(meta["method"])]
    if meta.get("retrieval_method"):
        return [str(meta["retrieval_method"])]
    return []


def _trace_latency(trace: AgentTrace) -> float:
    timings = trace.timings or {}
    if "total_ms" in timings:
        return float(timings["total_ms"])
    return float(sum(float(value) for value in timings.values() if isinstance(value, (int, float))))


def _strip_excluded(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_excluded(item)
            for key, item in value.items()
            if key not in EXCLUDED_KEYS
        }
    if isinstance(value, list):
        return [_strip_excluded(item) for item in value]
    return value


def _int(value: Any, *, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default
