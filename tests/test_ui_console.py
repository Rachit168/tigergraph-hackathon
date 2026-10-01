"""Judge console view-models, live runtime, metrics, and public-data safety."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from answering.models import Citation, PipelineTimings
from evaluation.harness import HarnessResult, ThreeWayEvaluationHarness
from evaluation.trace_export import EXCLUDED_KEYS, SCHEMA_VERSION
from ui.catalog import AGENT_BEHAVIOR, PIPELINES, SYSTEM_STATUS, canonical_benchmark
from ui.data import bootstrap, public_questions
from ui.demo_records import recorded_export_records
from ui.runtime import ConsoleRuntime, investigation_from_result, reset_runtime_for_tests
from ui.viewmodels import (
    empty_investigation,
    graph_from_evidence,
    investigation_from_export,
    record_is_submission_safe,
    status_label,
)

UI_ROOT = Path(__file__).resolve().parents[1] / "ui"
FORBIDDEN = (
    "eval_hidden",
    "phase11_holdout",
    "eval_holdout",
    "official_hidden",
    "hold-",
    "hidden50",
    "hidden_submission",
)
UI_INTERNAL = ("Phase 10", "Phase 13", "Phase 14", "recorded public example", "Recorded public example")


class FakeSystem:
    def __init__(self, name: str, answer: str, status: str = "answered", **extra: object) -> None:
        self.system_name = name
        self.answer = answer
        self.status = status
        self.extra = extra

    def run(self, question: str, qtype: str | None = None) -> HarnessResult:
        del qtype
        evidence = list(self.extra.get("evidence") or [])
        citations = list(self.extra.get("citations") or [])
        metadata = dict(self.extra.get("metadata") or {})
        metadata.setdefault("generator_notes", {"tokens": 12, "model_calls": 1, "tokens_unknown": False})
        return HarnessResult(
            system_name=self.system_name,
            question=question,
            answer=self.answer,
            citations=citations,
            latency_ms=float(self.extra.get("latency_ms") or 25.0),
            retrieval_metadata=metadata,
            evidence_metadata=evidence,
            status=self.status,
            timings=PipelineTimings(retrieval_ms=10.0, packing_ms=1.0, generation_ms=14.0, total_ms=25.0),
        )


def _live_runtime() -> ConsoleRuntime:
    harness = ThreeWayEvaluationHarness(
        [
            FakeSystem(
                "rag",
                "23",
                evidence=[
                    {
                        "evidence_id": "rag:chunk:Q1::c000",
                        "evidence_type": "chunk",
                        "retrieval_method": "bm25",
                        "chunk_id": "Q1::c000",
                        "document_id": "Q1",
                        "text": "Nations: 23",
                    }
                ],
                citations=[Citation(evidence_id="rag:chunk:Q1::c000", document_id="Q1", chunk_id="Q1::c000")],
                metadata={"method": "sparse", "retrieval_methods": ["bm25"]},
            ),
            FakeSystem(
                "graphrag",
                "23",
                evidence=[
                    {
                        "evidence_id": "g1:entity:Q1",
                        "evidence_type": "entity",
                        "event_id": "Q1",
                        "value": "Event",
                        "retrieval_method": "gsql:lookup_event",
                    },
                    {
                        "evidence_id": "g1:edge:HELD_AT:venue",
                        "evidence_type": "edge",
                        "field_name": "HELD_AT",
                        "value": "venue",
                        "event_id": "Q1",
                        "retrieval_method": "gsql:lookup_event",
                    },
                ],
                citations=[Citation(evidence_id="g1:entity:Q1", event_id="Q1", document_id="Q1")],
                metadata={"retrieval_method": "gsql:lookup_event", "retrieval_methods": ["gsql:lookup_event"]},
            ),
            FakeSystem(
                "agentic_graphrag",
                "Yi Siling",
                evidence=[
                    {
                        "evidence_id": "a1:fact:gold_raw:Q1",
                        "evidence_type": "fact",
                        "field_name": "gold_raw",
                        "value": "Yi Siling",
                        "event_id": "Q1",
                    }
                ],
                citations=[Citation(evidence_id="a1:fact:gold_raw:Q1", event_id="Q1")],
                metadata={
                    "trace": {
                        "question": "q",
                        "interpreted": {"qtype": "multi_hop"},
                        "plan_steps": [
                            {"iteration": 0, "kind": "primary", "reason": "typed hops", "actions": [{"tool": "retrieve_spec"}]},
                            {"iteration": 1, "kind": "follow_up", "reason": "gap", "actions": [{"tool": "event_neighborhood"}]},
                        ],
                        "tool_calls": [
                            {
                                "tool_call_id": "a1",
                                "tool": "retrieve_spec",
                                "arguments": {"operation": "events_at_venue_date"},
                                "reason": "typed hops",
                                "success": True,
                                "elapsed_ms": 11,
                                "retrieval_method": "gsql:events_at_venue_date",
                                "evidence_ids": ["a1:fact:gold_raw:Q1"],
                            },
                            {
                                "tool_call_id": "a2",
                                "tool": "event_neighborhood",
                                "arguments": {"event_id": "Q1"},
                                "reason": "relational gap",
                                "success": True,
                                "elapsed_ms": 8,
                                "retrieval_method": "gsql:event_neighborhood",
                                "evidence_ids": [],
                                "parallel_group": "repair",
                            },
                        ],
                        "stop_reason": "answered",
                        "follow_up_decisions": [
                            "primary venue/date hit left a relational gap: typed HELD_AT/IN_GAMES hops are still missing",
                            "stop:answered",
                        ],
                        "slot_updates": [
                            {"iteration": 0, "slot_id": "target_event", "status": "resolved", "evidence_ids": ["a1:fact:gold_raw:Q1"]},
                            {"iteration": 0, "slot_id": "graph_relation", "status": "searching", "evidence_ids": []},
                            {"iteration": 1, "slot_id": "graph_relation", "status": "resolved", "evidence_ids": []},
                        ],
                        "strategy_changes": ["primary_to_neighborhood"],
                        "total_tool_calls": 2,
                        "total_steps": 2,
                        "retrieval_methods": ["gsql:events_at_venue_date", "gsql:event_neighborhood"],
                    },
                    "stop_reason": "answered",
                    "generator_notes": {"tokens": 40, "model_calls": 1},
                },
            ),
        ]
    )
    return ConsoleRuntime(harness=harness)


class CanonicalMetricsTests(unittest.TestCase):
    def test_published_headline_from_single_source(self) -> None:
        bench = canonical_benchmark()
        payload = bootstrap()["benchmark"]
        self.assertEqual(bench["label"], "Published public benchmark")
        self.assertEqual(payload["label"], bench["label"])
        self.assertTrue(payload["replaceable"])
        self.assertEqual(PIPELINES["rag"]["correctness"], 67.0)
        self.assertEqual(PIPELINES["graphrag"]["correctness"], 98.0)
        self.assertEqual(PIPELINES["agentic_graphrag"]["correctness"], 97.0)
        self.assertEqual(payload["headline"]["rag"], 67.0)
        self.assertEqual(payload["headline"]["graphrag"], 98.0)
        self.assertEqual(payload["headline"]["agentic_graphrag"], 97.0)
        self.assertTrue(payload["vector_experiments_excluded"])
        self.assertIsNone(payload["pipelines"]["rag"]["completeness"])
        self.assertEqual(AGENT_BEHAVIOR["follow_ups"], 28)
        blob = json.dumps(payload)
        self.assertNotIn("Phase 10", blob)
        self.assertNotIn("final", blob.casefold())

    def test_canonical_numbers_not_duplicated_inconsistently(self) -> None:
        bench = bootstrap()["benchmark"]
        for key, row in PIPELINES.items():
            self.assertEqual(bench["pipelines"][key]["correctness"], row["correctness"])
            self.assertEqual(bench["pipelines"][key]["exact"], row["exact"])
            self.assertIsNone(bench["pipelines"][key]["tokens"])
            self.assertIsNone(bench["pipelines"][key]["latency_p50_ms"])
            self.assertIsNone(bench["pipelines"][key]["errors"])


class LiveRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = _live_runtime()
        reset_runtime_for_tests(self.runtime)

    def tearDown(self) -> None:
        reset_runtime_for_tests(None)

    def test_live_investigate_uses_backend_answer(self) -> None:
        view = self.runtime.investigate("How many nations?", "rag")
        self.assertEqual(view["answer"], "23")
        self.assertEqual(view["status"], "answered")
        self.assertEqual(view["status_label"], "ANSWERED")
        self.assertFalse(view["recorded"])
        self.assertTrue(view["evidence"])
        self.assertTrue(record_is_submission_safe(view["export_record"]))
        for key in EXCLUDED_KEYS:
            self.assertNotIn(key, view["export_record"])

    def test_compare_all_runs_three_pipelines(self) -> None:
        payload = self.runtime.compare("Who won?")
        self.assertEqual(len(payload["results"]), 3)
        self.assertEqual(len(payload["comparison"]), 3)
        by_name = {row["pipeline"]: row for row in payload["comparison"]}
        self.assertEqual(by_name["rag"]["answer"], "23")
        self.assertEqual(by_name["graphrag"]["answer"], "23")
        self.assertEqual(by_name["agentic_graphrag"]["answer"], "Yi Siling")
        self.assertEqual(by_name["agentic_graphrag"]["citation_count"], 1)

    def test_agentic_live_trace_is_real(self) -> None:
        view = self.runtime.investigate("Who won?", "agentic_graphrag")
        self.assertIsNotNone(view["agent"])
        self.assertTrue(view["agent"]["follow_up_occurred"])
        self.assertEqual(view["agent"]["stop_reason"], "answered")
        self.assertIn("Investigation stopped because the required evidence was sufficient", view["stop_explanation"])
        tools = [step["action"] for step in view["steps"]]
        self.assertIn("retrieve_spec", tools)
        self.assertIn("event_neighborhood", tools)
        self.assertTrue(view["why_continued"])
        self.assertIn("relationship", view["continue_explanation"].casefold())
        self.assertIsNotNone(view["evidence_diff"])
        self.assertGreaterEqual(len(view["evidence_diff"]["steps"]), 2)
        self.assertEqual(view["runtime_mode"], "live")
        self.assertTrue(any("graph_relation" in (step.get("remaining") or []) for step in view["evidence_diff"]["steps"]))

    def test_empty_question_does_not_fake_success(self) -> None:
        view = self.runtime.investigate("", "rag")
        self.assertEqual(view["status"], "error")
        self.assertEqual(view["status_label"], "ERROR")
        self.assertFalse(view["answer"])
        self.assertEqual(view["steps"], [])
        self.assertIn("failed", view["stop_explanation"].casefold())
        self.assertIn("failed", view["answer_display"].casefold())

    def test_unavailable_pipeline_has_no_invented_trace(self) -> None:
        runtime = ConsoleRuntime(harness=ThreeWayEvaluationHarness([FakeSystem("rag", "1")]))
        view = runtime.investigate("Who won?", "graphrag")
        self.assertEqual(view["status"], "unavailable")
        self.assertEqual(view["status_label"], "UNAVAILABLE")
        self.assertEqual(view["steps"], [])
        self.assertFalse(view["answer"])
        self.assertIsNone(view["agent"])
        self.assertIn("unavailable", view["stop_explanation"].casefold())
        self.assertIn("unavailable", view["answer_display"].casefold())
        self.assertNotIn("(empty)", view["answer_display"])


class CompareAndHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = _live_runtime()
        reset_runtime_for_tests(self.runtime)

    def tearDown(self) -> None:
        reset_runtime_for_tests(None)

    def test_compare_includes_what_changed_without_winner(self) -> None:
        payload = self.runtime.compare("Who won?")
        blob = json.dumps(payload)
        self.assertNotIn("best", blob.casefold())
        self.assertTrue(payload["what_changed"])
        self.assertIsNotNone(payload["fixed_vs_agentic"])
        self.assertIn("tool call", " ".join(payload["what_changed"]).casefold())
        self.assertEqual(payload["comparison"][2]["tool_calls"], 2)
        self.assertNotIn("(empty)", json.dumps(payload["comparison"]))
        self.assertTrue(all(row.get("answer_display") for row in payload["comparison"]))

    def test_partial_compare_stays_unavailable_without_winner(self) -> None:
        runtime = ConsoleRuntime(harness=ThreeWayEvaluationHarness([FakeSystem("rag", "23")]))
        payload = runtime.compare("Who won?")
        statuses = {row["pipeline"]: row["status"] for row in payload["comparison"]}
        self.assertEqual(statuses["rag"], "answered")
        self.assertEqual(statuses["graphrag"], "unavailable")
        self.assertEqual(statuses["agentic_graphrag"], "unavailable")
        blob = json.dumps(payload).casefold()
        self.assertNotIn("best", blob)
        self.assertTrue(any("unavailable" in line.casefold() for line in payload["what_changed"]))

    def test_health_checks_are_structured_and_read_only(self) -> None:
        from ui.runtime import health_payload

        payload = health_payload()
        self.assertTrue(payload["read_only"])
        self.assertTrue(payload["checks"])
        ids = {item["id"] for item in payload["checks"]}
        self.assertIn("tigergraph", ids)
        self.assertIn("corpus", ids)
        self.assertIn("export", ids)
        self.assertIn(payload["overall"], {"PASS", "WARN", "UNAVAILABLE", "ERROR"})
        self.assertNotIn("gold_answer", json.dumps(payload).casefold())


class TraceRenderingTests(unittest.TestCase):
    def test_records_use_exporter_schema_and_strip_eval_fields(self) -> None:
        records = recorded_export_records()
        self.assertEqual(len(records), 12)
        for record in records:
            self.assertEqual(record["schema_version"], SCHEMA_VERSION)
            self.assertTrue(record_is_submission_safe(record))
            for key in EXCLUDED_KEYS:
                self.assertNotIn(key, record)

    def test_agentic_trace_renders_follow_up_steps(self) -> None:
        record = next(
            item
            for item in recorded_export_records()
            if item["question_id"] == "pub-017" and item["pipeline"] == "agentic_graphrag"
        )
        view = investigation_from_export(record)
        self.assertTrue(view["agent"]["follow_up_occurred"])
        tools = [step["action"] for step in view["steps"]]
        self.assertIn("retrieve_spec", tools)
        self.assertIn("event_neighborhood", tools)

    def test_rag_and_graphrag_work_without_agent_trace(self) -> None:
        records = { (item["question_id"], item["pipeline"]): item for item in recorded_export_records() }
        rag = investigation_from_export(records[("pub-025", "rag")])
        graph = investigation_from_export(records[("pub-025", "graphrag")])
        self.assertIsNone(rag["agent"])
        self.assertIsNone(graph["agent"])
        self.assertTrue(any(step["action"] == "BM25" for step in rag["steps"]))
        self.assertTrue(any(step["action"] == "typed GSQL" for step in graph["steps"]))

    def test_evidence_ids_render(self) -> None:
        record = next(
            item
            for item in recorded_export_records()
            if item["question_id"] == "pub-017" and item["pipeline"] == "graphrag"
        )
        view = investigation_from_export(record)
        self.assertTrue(view["evidence_ids"])
        self.assertTrue(any(item["evidence_id"] for item in view["evidence"]))
        self.assertIn("Q1137721", view["event_ids"])

    def test_missing_evidence_does_not_crash(self) -> None:
        view = empty_investigation(question="?", pipeline="rag")
        self.assertEqual(view["evidence"], [])
        self.assertEqual(view["graph"]["nodes"], [])
        self.assertEqual(view["steps"], [])
        self.assertIsNone(view["agent"])
        graph = graph_from_evidence([], [])
        self.assertEqual(graph["nodes"], [])
        self.assertFalse(graph["truncated"])

    def test_empty_stop_reason_does_not_crash(self) -> None:
        view = investigation_from_export(
            {
                "schema_version": 1,
                "question": "q",
                "pipeline": "agentic_graphrag",
                "answer": "",
                "status": "",
                "citations": [],
                "evidence": [],
                "agent_trace": {"stop_reason": None, "tools": [], "plan_steps": []},
                "total_tokens": 0,
                "model_calls": 0,
                "latency_ms": 0,
            }
        )
        self.assertEqual(view["stop_reason"], None)
        self.assertIsNone(view["agent"])
        self.assertEqual(view["steps"], [])

    def test_timeout_status_label(self) -> None:
        self.assertEqual(status_label("generation_error", "timeout"), "TIMEOUT")
        self.assertEqual(status_label("not_found"), "NOT FOUND")
        self.assertEqual(status_label("ambiguous"), "AMBIGUOUS")

    def test_graph_truncates_pathological_results(self) -> None:
        evidence = [
            {
                "evidence_id": f"e{i}",
                "evidence_type": "entity",
                "event_id": f"Q{i}",
                "value": f"Event {i}",
            }
            for i in range(200)
        ]
        graph = graph_from_evidence(evidence, [])
        self.assertTrue(graph["truncated"])
        self.assertLessEqual(len(graph["nodes"]), 64)

    def test_graph_edges_only_from_returned_evidence(self) -> None:
        records = { (item["question_id"], item["pipeline"]): item for item in recorded_export_records() }
        rag = investigation_from_export(records[("pub-017", "rag")])
        graph = investigation_from_export(records[("pub-017", "graphrag")])
        self.assertNotIn("HELD_AT", {edge["label"] for edge in rag["graph"]["edges"]})
        self.assertIn("HELD_AT", {edge["label"] for edge in graph["graph"]["edges"]})

    def test_investigation_from_result_keeps_packed_evidence(self) -> None:
        result = HarnessResult(
            system_name="rag",
            question="q",
            answer="23",
            citations=[Citation(evidence_id="e1", document_id="Q1", chunk_id="c1")],
            latency_ms=10,
            retrieval_metadata={"method": "sparse", "generator_notes": {"tokens": 3, "model_calls": 1}},
            evidence_metadata=[{"evidence_id": "e1", "evidence_type": "chunk", "text": "Nations: 23", "chunk_id": "c1", "document_id": "Q1"}],
            status="answered",
        )
        view = investigation_from_result(result)
        self.assertEqual(view["evidence"][0]["text"], "Nations: 23")
        self.assertNotIn("gold", json.dumps(view["export_record"]))


class VectorStatusTests(unittest.TestCase):
    def test_vector_panel_uses_verified_static_values(self) -> None:
        vector = bootstrap()["system"]["vector_status"]
        self.assertEqual(SYSTEM_STATUS["vector"], vector)
        self.assertEqual(vector["chunk_embeddings_indexed"], 28905)
        self.assertFalse(vector["production"])
        self.assertEqual(bootstrap()["system"]["graph_status"]["graph"], "OlympicGraph")


class PublicSafetyTests(unittest.TestCase):
    def test_public_questions_are_pub_only_without_gold_fields(self) -> None:
        rows = public_questions()
        for row in rows:
            self.assertTrue(row["qid"].startswith("pub-"))
            self.assertEqual(set(row), {"qid", "question", "qtype"})

    def test_bootstrap_has_no_demo_or_hidden_or_phase_language(self) -> None:
        payload = bootstrap()
        self.assertNotIn("demo_questions", payload)
        self.assertNotIn("investigations", payload)
        blob = json.dumps(payload)
        for token in FORBIDDEN + UI_INTERNAL:
            self.assertNotIn(token, blob)

    def test_ui_assets_have_no_hidden_or_internal_stage_strings(self) -> None:
        for path in UI_ROOT.rglob("*"):
            if path.suffix.lower() not in {".py", ".js", ".css", ".html"}:
                continue
            if path.name == "demo_records.py":
                continue
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN:
                self.assertNotIn(token, text, msg=f"{token} in {path}")
            if path.suffix.lower() in {".js", ".css", ".html"} or (
                path.suffix.lower() == ".py" and path.name != "demo_records.py"
            ):
                for token in UI_INTERNAL:
                    self.assertNotIn(token, text, msg=f"{token} in {path}")
            if path.suffix.lower() in {".js", ".html"}:
                self.assertNotIn("(empty)", text, msg=f"placeholder empty token in {path}")


if __name__ == "__main__":
    unittest.main()
