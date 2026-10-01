"""Public three-way reliability metadata. Scoring values must stay unchanged."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from answering.models import Citation
from evaluation.benchmark import failure_table, score_harness_result, summarize_rows
from evaluation.harness import HarnessResult
from evaluation.three_way import CSV_COLUMNS, render_report, write_benchmark_artifacts


GOLD = {"qid": "pub-001", "qtype": "lookup", "question": "How many nations?", "answer": ["23"]}
SCORING_KEYS = (
    "correctness",
    "correctness_exact",
    "completeness",
    "completeness_reason",
    "grounding",
    "grounding_status",
    "citation_validity",
    "failure_category",
    "answer_status",
    "answer_suppressed",
)


def _attempts(*, retry: bool = False, unknown: bool = False) -> list[dict]:
    first = {
        "ok": False,
        "provider_error": "http_429",
        "tokens": None if unknown else 3,
        "usage_known": not unknown,
        "finish_reason": "ignored",
    }
    second = {
        "ok": True,
        "provider_error": None,
        "tokens": 11,
        "usage_known": True,
    }
    if retry:
        return [first, second]
    return [
        {
            "ok": True,
            "provider_error": None,
            "tokens": 9,
            "usage_known": True,
        }
    ]


def _rag_result(*, status: str = "answered", answer: str = "23", notes: dict | None = None) -> HarnessResult:
    return HarnessResult(
        system_name="rag",
        question=GOLD["question"],
        answer=answer,
        citations=[Citation(evidence_id="e1")] if status == "answered" else [],
        latency_ms=1.0,
        retrieval_metadata={
            "method": "sparse",
            "hit_count": 1,
            "generator_notes": dict(notes or {}),
        },
        evidence_metadata=[{"evidence_id": "e1", "document_id": "Q1"}],
        status=status,
        errors=["generation_error"] if status == "generation_error" else [],
    )


def _graph_result(
    system_name: str,
    *,
    status: str = "answered",
    answer: str = "23",
    notes: dict | None = None,
) -> HarnessResult:
    pipeline = {"fixed_policy": system_name == "graphrag", "graph_retriever_calls": 1}
    pipeline.update(notes or {})
    return HarnessResult(
        system_name=system_name,
        question=GOLD["question"],
        answer=answer,
        citations=[Citation(evidence_id="e1")] if status == "answered" else [],
        latency_ms=2.0,
        retrieval_metadata={
            "cardinality": "one",
            "event_ids": ["Q1"],
            "retrieval_method": "gsql:lookup_event",
            "pipeline_notes": pipeline,
            "stop_reason": "answered" if system_name == "agentic_graphrag" else None,
            "total_tool_calls": 1 if system_name == "agentic_graphrag" else None,
            "trace": {"total_steps": 1, "total_tool_calls": 1} if system_name == "agentic_graphrag" else None,
        },
        evidence_metadata=[{"evidence_id": "e1", "event_id": "Q1"}],
        status=status,
        errors=["generation_error"] if status == "generation_error" else [],
    )


class ReliabilityMetricsTests(unittest.TestCase):
    def test_successful_generation_has_empty_failure_class(self) -> None:
        notes = {
            "model_calls": 1,
            "tokens": 9,
            "tokens_unknown": False,
            "model_accounting": "provider_completion",
            "attempts": _attempts(),
        }
        row = score_harness_result(GOLD, _rag_result(notes=notes))
        self.assertTrue(row["correctness"])
        self.assertEqual(row["answer_status"], "answered")
        self.assertIsNone(row["failure_class"])
        self.assertFalse(row["tokens_unknown"])
        self.assertEqual(row["model_calls"], 1)
        self.assertEqual(row["tokens"], 9)
        self.assertEqual(len(row["attempts"]), 1)
        self.assertTrue(row["attempts"][0]["ok"])
        self.assertNotIn("finish_reason", row["attempts"][0])

    def test_transient_retry_records_multiple_attempts_and_sums_tokens_once(self) -> None:
        notes = {
            "model_calls": 2,
            "tokens": 14,
            "tokens_unknown": False,
            "model_accounting": "provider_completion_retry",
            "attempts": _attempts(retry=True),
        }
        row = score_harness_result(GOLD, _rag_result(notes=notes))
        self.assertEqual(row["model_calls"], 2)
        self.assertEqual(row["tokens"], 14)
        self.assertEqual(len(row["attempts"]), 2)
        self.assertEqual(row["attempts"][0]["tokens"], 3)
        self.assertEqual(row["attempts"][1]["tokens"], 11)
        self.assertNotEqual(row["tokens"], 3 + 11 + 14)
        self.assertFalse(row["tokens_unknown"])
        self.assertIsNone(row["failure_class"])

    def test_tokens_unknown_and_failure_class_are_propagated(self) -> None:
        notes = {
            "model_calls": 3,
            "tokens": 0,
            "tokens_unknown": True,
            "failure_class": "timeout",
            "model_accounting": "provider_completion_retry",
            "attempts": [
                {"ok": False, "provider_error": "timeout", "tokens": None, "usage_known": False},
                {"ok": False, "provider_error": "timeout", "tokens": None, "usage_known": False},
                {"ok": False, "provider_error": "timeout", "tokens": None, "usage_known": False},
            ],
        }
        row = score_harness_result(
            GOLD,
            _rag_result(status="generation_error", answer="", notes=notes),
        )
        self.assertEqual(row["answer_status"], "generation_error")
        self.assertEqual(row["failure_class"], "timeout")
        self.assertTrue(row["tokens_unknown"])
        self.assertEqual(row["tokens"], 0)
        self.assertEqual(row["model_calls"], 3)
        self.assertEqual(row["failure_category"], "generation_failure")
        self.assertFalse(row["correctness"])

    def test_known_zero_tokens_is_not_tokens_unknown(self) -> None:
        notes = {
            "model_calls": 0,
            "tokens": 0,
            "tokens_unknown": False,
            "model_accounting": "semantic_no_provider_call",
        }
        row = score_harness_result(GOLD, _rag_result(status="abstained", answer="", notes=notes))
        self.assertEqual(row["tokens"], 0)
        self.assertFalse(row["tokens_unknown"])
        self.assertEqual(row["model_calls"], 0)
        self.assertIsNone(row["failure_class"])

    def test_three_way_rows_preserve_the_same_metadata(self) -> None:
        notes = {
            "model_calls": 2,
            "tokens": 14,
            "tokens_unknown": True,
            "failure_class": "timeout",
            "model_accounting": "provider_completion_retry",
            "attempts": _attempts(retry=True, unknown=True),
        }
        rows = [
            score_harness_result(
                GOLD,
                _rag_result(status="generation_error", answer="", notes=notes),
            ),
            score_harness_result(
                GOLD,
                _graph_result("graphrag", status="generation_error", answer="", notes=notes),
            ),
            score_harness_result(
                GOLD,
                _graph_result("agentic_graphrag", status="generation_error", answer="", notes=notes),
            ),
        ]
        self.assertEqual({row["system_name"] for row in rows}, {"rag", "graphrag", "agentic_graphrag"})
        for row in rows:
            self.assertEqual(row["failure_class"], "timeout")
            self.assertTrue(row["tokens_unknown"])
            self.assertEqual(row["model_calls"], 2)
            self.assertEqual(row["tokens"], 14)
            self.assertEqual(len(row["attempts"]), 2)
            self.assertIsNone(row["attempts"][0]["tokens"])
            self.assertEqual(row["attempts"][1]["tokens"], 11)
            self.assertEqual(row["failure_category"], "generation_failure")
            self.assertFalse(row["correctness"])

    def test_scoring_output_is_unchanged_for_gold_match_and_mismatch(self) -> None:
        gold = {"qid": "pub-005", "qtype": "multi_hop", "question": "q", "answer": ["Naim Süleymanoğlu"]}
        graph = HarnessResult(
            system_name="graphrag",
            question="q",
            answer="Naim Süleymanoğlu",
            citations=[Citation(evidence_id="e1")],
            latency_ms=1.0,
            retrieval_metadata={"cardinality": "one", "event_ids": ["Q1"], "notes": {}},
            evidence_metadata=[{"evidence_id": "e1", "event_id": "Q1"}],
            status="answered",
        )
        duplicated = HarnessResult(
            system_name="agentic_graphrag",
            question="q",
            answer="Naim Süleymanoğlu | Naim Süleymanoğlu",
            citations=[Citation(evidence_id="e1"), Citation(evidence_id="e2")],
            latency_ms=1.0,
            retrieval_metadata={"cardinality": "one", "event_ids": ["Q1"], "notes": {}, "stop_reason": "answered"},
            evidence_metadata=[{"evidence_id": "e1", "event_id": "Q1"}, {"evidence_id": "e2", "event_id": "Q1"}],
            status="answered",
        )
        graph_row = score_harness_result(gold, graph)
        dup_row = score_harness_result(gold, duplicated)
        self.assertTrue(graph_row["correctness"])
        self.assertTrue(graph_row["correctness_exact"])
        self.assertTrue(graph_row["grounding"])
        self.assertTrue(graph_row["citation_validity"])
        self.assertIsNone(graph_row["failure_category"])
        self.assertFalse(dup_row["correctness"])
        self.assertFalse(dup_row["correctness_exact"])
        self.assertEqual(dup_row["failure_category"], "incorrect")
        self.assertEqual(graph_row["completeness"], None)
        self.assertEqual(dup_row["completeness"], None)

    def test_csv_json_and_report_include_reliability_without_changing_primary_table(self) -> None:
        success_notes = {
            "model_calls": 1,
            "tokens": 9,
            "tokens_unknown": False,
            "model_accounting": "provider_completion",
            "attempts": _attempts(),
        }
        fail_notes = {
            "model_calls": 3,
            "tokens": 0,
            "tokens_unknown": True,
            "failure_class": "timeout",
            "model_accounting": "provider_completion_retry",
            "attempts": [
                {"ok": False, "provider_error": "timeout", "tokens": None, "usage_known": False},
            ],
        }
        rows = [
            score_harness_result(GOLD, _rag_result(notes=success_notes)),
            score_harness_result(
                GOLD,
                _graph_result("graphrag", status="generation_error", answer="", notes=fail_notes),
            ),
        ]
        summary = summarize_rows(rows)
        self.assertIn("reliability", summary)
        self.assertEqual(summary["reliability"]["rag"]["generation_errors"], 0)
        self.assertEqual(summary["reliability"]["graphrag"]["generation_errors"], 1)
        self.assertEqual(summary["reliability"]["graphrag"]["failure_class_counts"], {"timeout": 1})
        self.assertEqual(summary["reliability"]["graphrag"]["tokens_unknown"], 1)
        self.assertEqual(summary["overall"]["rag"]["correctness"], 100.0)
        compact = failure_table(rows)
        self.assertEqual(compact[0]["failure_class"], "timeout")
        payload = {
            "run_id": "test",
            "n_questions": 1,
            "n_rows": 2,
            "config": {"systems": ["rag", "graphrag"], "generator": "SemanticGenerator"},
            "rows": rows,
            "summary": summary,
            "failures": compact,
        }
        report = render_report(payload)
        self.assertIn("## 1. Overall comparison", report)
        self.assertIn("correctness %", report)
        self.assertNotIn("tokens_unknown", report.split("## 1. Overall comparison", 1)[1].split("## 2.", 1)[0])
        self.assertIn("## 11. Generation reliability", report)
        self.assertIn("timeout", report)
        self.assertIn("tokens_unknown", CSV_COLUMNS)
        self.assertIn("failure_class", CSV_COLUMNS)
        self.assertIn("attempts", CSV_COLUMNS)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_benchmark_artifacts(payload, Path(tmp))
            csv_header = paths["csv"].read_text(encoding="utf-8").splitlines()[0]
            loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
        self.assertIn("failure_class", csv_header)
        self.assertIn("tokens_unknown", csv_header)
        self.assertIn("attempts", csv_header)
        self.assertEqual(loaded["rows"][0]["failure_class"], None)
        self.assertEqual(loaded["rows"][1]["failure_class"], "timeout")
        self.assertTrue(loaded["rows"][1]["tokens_unknown"])
        self.assertEqual(loaded["summary"]["overall"]["rag"]["correctness"], 100.0)

    def test_scoring_keys_do_not_depend_on_reliability_fields(self) -> None:
        bare = score_harness_result(GOLD, _rag_result(notes={}))
        noted = score_harness_result(
            GOLD,
            _rag_result(
                notes={
                    "model_calls": 2,
                    "tokens": 14,
                    "tokens_unknown": True,
                    "failure_class": None,
                    "attempts": _attempts(retry=True),
                }
            ),
        )
        for key in SCORING_KEYS:
            self.assertEqual(bare[key], noted[key], key)


if __name__ == "__main__":
    unittest.main()
