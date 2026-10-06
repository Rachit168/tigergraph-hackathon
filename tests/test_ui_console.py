"""Judge console view-models, live runtime, metrics, and public-data safety."""

from __future__ import annotations

import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

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
    health_checks,
    investigation_from_export,
    record_is_submission_safe,
    runtime_mode,
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
        self.assertEqual(PIPELINES["rag"]["correctness"], 65.0)
        self.assertEqual(PIPELINES["graphrag"]["correctness"], 98.0)
        self.assertEqual(PIPELINES["agentic_graphrag"]["correctness"], 99.0)
        self.assertEqual(payload["headline"]["rag"], 65.0)
        self.assertEqual(payload["headline"]["graphrag"], 98.0)
        self.assertEqual(payload["headline"]["agentic_graphrag"], 99.0)
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
            self.assertIsNotNone(bench["pipelines"][key]["tokens"])
            self.assertIsNone(bench["pipelines"][key]["latency_p50_ms"])
            self.assertIsNotNone(bench["pipelines"][key]["errors"])


    def test_family_metrics_match_published_summary(self) -> None:
        document = (UI_ROOT.parent / "docs" / "public_benchmark.md").read_text(encoding="utf-8")
        section = document.split("## Question-family breakdown", 1)[1].split("## Agentic behavior", 1)[0]
        published = {}
        for line in section.splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 5 or not cells[1].isdigit():
                continue
            family = cells[0].rsplit("/", 1)[-1].strip().casefold()
            if family == "complete-set aggregation":
                family = "aggregation"
            family = family.replace("-", "_")
            published[family] = (int(cells[1]), [float(value.rstrip("%")) for value in cells[2:]])
        families = canonical_benchmark()["families"]
        self.assertEqual(set(published), {row["id"] for row in families})
        self.assertEqual(bootstrap()["benchmark"]["families"], families)
        self.assertEqual(sum(row["n"] for row in families), canonical_benchmark()["n"])
        for row in families:
            with self.subTest(family=row["id"]):
                self.assertEqual((row["n"], [row[key] for key in PIPELINES]), published[row["id"]])
        for pipeline in PIPELINES:
            correct = sum(round(row[pipeline] * row["n"] / 100) for row in families)
            self.assertEqual(correct, PIPELINES[pipeline]["correctness"])


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


    def test_system_failure_is_redacted_in_investigation_and_export(self) -> None:
        from unittest.mock import Mock

        message = (
            'https://private.example.invalid/api?token=URL_VALUE '
            'api_key=KEY_VALUE token=TOKEN_VALUE Authorization: Bearer AUTH_VALUE; '
            'password=PASSWORD_VALUE secret="SECRET VALUE"; '
            'path="D:\\private folder\\config.txt" /srv/private/config.txt'
        )
        for pipeline in PIPELINES:
            with self.subTest(pipeline=pipeline):
                system = Mock(system_name=pipeline)
                system.run.side_effect = ValueError(message)
                runtime = ConsoleRuntime(harness=ThreeWayEvaluationHarness([system]))
                view = runtime.investigate("An arbitrary public question", pipeline)
                self.assertEqual(view["status"], "error")
                self.assertEqual(view["answer"], "")
                self.assertEqual(view["citations"], [])
                self.assertIn("system_error:ValueError", view["errors"][0])
                self.assertIn("operation=system_run", view["errors"][0])
                self.assertTrue(record_is_submission_safe(view["export_record"]))
                blob = json.dumps(view)
                for value in ("private.example.invalid", "URL_VALUE", "KEY_VALUE", "TOKEN_VALUE",
                              "AUTH_VALUE", "PASSWORD_VALUE", "SECRET VALUE", "private folder",
                              "/srv/private/config.txt"):
                    self.assertNotIn(value, blob)
                self.assertEqual(view["export_record"]["errors"], view["errors"])

    def test_runtime_initialization_and_health_errors_use_same_redaction(self) -> None:
        from unittest.mock import Mock, patch
        from ui.runtime import health_payload

        message = "api_key=KEY_VALUE password=PASSWORD_VALUE token=TOKEN_VALUE"
        runtime = ConsoleRuntime()
        with patch.object(runtime, "_ensure_harness", side_effect=RuntimeError(message)):
            view = runtime.investigate("An arbitrary public question", "rag")
        self.assertEqual(view["status"], "unavailable")
        with (
            patch("ui.runtime.get_runtime", return_value=Mock(capabilities=lambda: {})),
            patch("ui.runtime.run_verify", side_effect=RuntimeError(message)),
        ):
            health = health_payload()
        for payload in (view, health):
            for value in ("KEY_VALUE", "PASSWORD_VALUE", "TOKEN_VALUE"):
                self.assertNotIn(value, json.dumps(payload))

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

    def test_rag_source_links_do_not_claim_graph_retrieval(self) -> None:
        record = {
            "schema_version": 1, "pipeline": "rag", "question": "q", "status": "answered",
            "answer": "answer", "retrieval_methods": ["sparse"],
            "evidence": [{"evidence_id": "c1", "evidence_type": "chunk", "chunk_id": "c1",
                          "document_id": "doc1", "event_id": "Q1", "text": "source"}],
        }
        view = investigation_from_export(record)
        self.assertTrue(view["graph"]["edges"])
        self.assertTrue(view["evidence_status"]["source_metadata_present"])
        self.assertFalse(view["evidence_status"]["graph_present"])
        self.assertEqual(view["export_record"], record)
        from ui.viewmodels import what_changed
        self.assertIn("text-only retrieval", " ".join(what_changed([view])))

    def test_fixed_graph_evidence_still_claims_graph_context(self) -> None:
        view = investigation_from_export({
            "schema_version": 1, "pipeline": "graphrag", "status": "answered",
            "evidence": [{"evidence_id": "event1", "evidence_type": "entity", "event_id": "Q1"}],
        })
        self.assertTrue(view["evidence_status"]["graph_present"])
        self.assertFalse(view["evidence_status"]["source_metadata_present"])

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

    def test_malformed_generation_is_visible_as_failure(self) -> None:
        view = investigation_from_export(
            {
                "schema_version": 1,
                "question": "q",
                "pipeline": "rag",
                "answer": "",
                "status": "generation_error",
                "failure_class": "malformed_output",
                "citations": [],
                "evidence": [],
                "total_tokens": 4,
                "model_calls": 1,
                "latency_ms": 10,
            }
        )
        self.assertEqual(view["status_label"], "GENERATION ERROR")
        self.assertEqual(view["runtime_mode"], "error")
        self.assertIn("malformed output", view["answer_display"])
        self.assertTrue(view["reliability"]["malformed"])

    def test_recorded_mode_is_preview(self) -> None:
        self.assertEqual(runtime_mode("answered", recorded=True), "preview")

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
    def test_vector_panel_uses_static_metadata_until_live_health_arrives(self) -> None:
        vector = bootstrap()["system"]["vector_status"]
        self.assertEqual(SYSTEM_STATUS["vector"], vector)
        self.assertEqual(vector["chunk_embeddings_indexed"], 28905)
        self.assertTrue(vector["production"])
        self.assertEqual(vector["status"], "NOT_CHECKED")
        self.assertIn("historical", vector["counts_source"])
        self.assertIn("optional bounded tigergraph vector follow-up", bootstrap()["system"]["production_routing"]["vector"].casefold())
        self.assertEqual(bootstrap()["system"]["graph_status"]["graph"], "OlympicGraph")

    def test_vector_health_check_states_are_truthful(self) -> None:
        base = {"capabilities": {}, "connection": {"ok": True}, "blockers": []}
        healthy = health_checks({
            **base,
            "vector": {
                "status": "READY",
                "ok": True,
                "schema_ok": True,
                "search_query_ok": True,
                "index_ok": True,
            },
        })
        self.assertEqual(next(item for item in healthy if item["id"] == "vector")["status"], "PASS")
        failed = health_checks({
            **base,
            "vector": {
                "status": "NOT_READY",
                "ok": False,
                "error": "search query is not installed",
            },
        })
        vector_check = next(item for item in failed if item["id"] == "vector")
        self.assertEqual(vector_check["status"], "WARN")
        self.assertIn("not installed", vector_check["detail"])


class LiveHarnessVectorWiringTests(unittest.TestCase):
    def _build_harness(self, *, vector_fails: bool):
        from unittest.mock import patch

        import ui.runtime as runtime

        class FakeClient:
            def __init__(self, settings) -> None:
                self.settings = settings

            def connect(self) -> bool:
                return True

        class FakeGraphRetriever:
            @classmethod
            def from_client(cls, client):
                return ("graph-retriever", client)

        class RecordingAgenticPipeline:
            instances = []

            def __init__(self, *args, **kwargs) -> None:
                self.args = args
                self.kwargs = kwargs
                type(self).instances.append(self)

        class Harness:
            def __init__(self, systems) -> None:
                self.systems = systems

        vector = object()

        def build_vector(*args, **kwargs):
            del args, kwargs
            if vector_fails:
                raise RuntimeError("embedding unavailable")
            return vector

        settings = SimpleNamespace(configured=True, graphname="OlympicGraph")
        llm = SimpleNamespace(configured=False)
        patches = [
            patch.object(runtime, "load_settings", return_value=settings),
            patch.object(runtime, "load_llm_settings", return_value=llm),
            patch.object(runtime, "build_generator", return_value="generator"),
            patch.object(runtime, "DEFAULT_CORPUS_PATH", SimpleNamespace(is_file=lambda: True)),
            patch.object(runtime, "parse_corpus", return_value=SimpleNamespace(documents=["doc"])),
            patch.object(runtime, "chunk_corpus", return_value=["chunk"]),
            patch("evaluation.retrieval_gold.build_parser", return_value=("index", "question-parser")),
            patch.object(runtime, "TigerGraphClient", FakeClient),
            patch.object(runtime, "TextRetriever", return_value="sparse-retriever"),
            patch.object(runtime, "GraphRAGQuestionParser", return_value="graph-parser"),
            patch.object(runtime, "GraphRetriever", FakeGraphRetriever),
            patch.object(runtime, "FixedGraphRAGPipeline", return_value="fixed-pipeline"),
            patch.object(runtime, "RAGAdapter", side_effect=lambda *args, **kwargs: ("rag", args, kwargs)),
            patch.object(runtime, "GraphRAGAdapter", side_effect=lambda *args, **kwargs: ("graph", args, kwargs)),
            patch.object(runtime, "AgenticGraphRAGAdapter", side_effect=lambda *args, **kwargs: ("agentic", args, kwargs)),
            patch.object(runtime, "AgenticGraphRAGPipeline", RecordingAgenticPipeline),
            patch.object(runtime, "load_embedding_settings", return_value="embedding-settings"),
            patch.object(runtime, "build_embedder", return_value=SimpleNamespace(dimension=384)),
            patch.object(runtime, "TigerGraphVectorStore", return_value="vector-store"),
            patch.object(runtime, "TigerGraphVectorRetriever", side_effect=build_vector),
            patch("evaluation.harness.ThreeWayEvaluationHarness", Harness),
        ]
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            harness = runtime.build_live_harness()
        return harness, RecordingAgenticPipeline.instances[-1], vector

    def test_agentic_receives_vector_retriever_when_construction_succeeds(self) -> None:
        _harness, pipeline, vector = self._build_harness(vector_fails=False)
        self.assertIs(pipeline.kwargs["vector_retriever"], vector)

    def test_agentic_falls_back_to_no_vector_retriever(self) -> None:
        _harness, pipeline, _vector = self._build_harness(vector_fails=True)
        self.assertIsNone(pipeline.kwargs["vector_retriever"])
class JavaScriptProcessTests(unittest.TestCase):
    def test_rendering_tests_exit_without_stdin(self) -> None:
        import shutil
        import subprocess

        node = shutil.which("node")
        if not node:
            self.skipTest("Node is not available")
        result = subprocess.run(
            [node, "tests/test_ui_rendering.js"], cwd=UI_ROOT.parent,
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("fail 0", result.stdout)
        self.assertNotIn("process.exit(0)", (UI_ROOT.parent / "tests/test_ui_rendering.js").read_text(encoding="utf-8"))


class PublicSafetyTests(unittest.TestCase):
    def test_bootstrap_does_not_claim_live_before_health_probe(self) -> None:
        self.assertEqual(bootstrap()["mode"], "unavailable")

    def test_console_replay_forces_preview_presentation(self) -> None:
        text = (UI_ROOT / "static" / "console.js").read_text(encoding="utf-8")
        self.assertIn('recorded_label: "SESSION REPLAY"', text)
        self.assertIn('runtime_mode: "preview"', text)

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
