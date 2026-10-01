"""Submission-safe trace export. No hidden evaluation imports or gold leakage."""

from __future__ import annotations

import copy
import inspect
import json
import unittest

from answering.models import Citation, PipelineTimings
from evaluation.harness import HarnessResult
from evaluation.trace_export import EXCLUDED_KEYS, SCHEMA_VERSION, export_submission_record
from retrieval.agentic.trace import AgentTrace, PlanStep, SlotUpdate
from retrieval.agentic.tools import ToolAction, ToolObservation


def _trace() -> AgentTrace:
    action = ToolAction(
        tool="retrieve_spec",
        arguments={"operation": "lookup_event", "include_chunks": False},
        reason="initial lookup",
    )
    observation = ToolObservation(
        tool_call_id="call-1",
        tool="retrieve_spec",
        arguments={"operation": "lookup_event", "include_chunks": False},
        reason="initial lookup",
        success=True,
        started_ms=1.5,
        elapsed_ms=12.25,
        fingerprint="fp-1",
        evidence_ids=["call-1:fact:nations:Q1"],
        retrieval_method="gsql:lookup_event",
    )
    return AgentTrace(
        question="How many nations?",
        interpreted={"qtype": "lookup", "operation": "lookup_event"},
        plan_steps=[
            PlanStep(iteration=0, kind="initial", reason="typed lookup", actions=[action]),
            PlanStep(iteration=1, kind="follow_up", reason="neighborhood", actions=[]),
        ],
        tool_calls=[observation],
        slot_updates=[
            SlotUpdate(iteration=0, slot_id="target_event", status="filled", evidence_ids=["call-1:fact:nations:Q1"])
        ],
        follow_up_decisions=["neighborhood"],
        strategy_changes=["lookup_to_neighborhood"],
        stop_reason="answered",
        total_steps=2,
        total_tool_calls=1,
        retrieval_methods=["gsql:lookup_event", "gsql:event_neighborhood"],
        timings={"retrieval_ms": 12.25, "generation_ms": 40.0, "total_ms": 55.5},
        observation_chars=120,
    )


class SubmissionTraceExportTests(unittest.TestCase):
    def test_full_agent_trace_export_is_rich_and_serializable(self) -> None:
        exported = export_submission_record(_trace(), question_id="pub-001")
        self.assertEqual(exported["schema_version"], SCHEMA_VERSION)
        self.assertEqual(exported["question_id"], "pub-001")
        self.assertEqual(exported["pipeline"], "agentic_graphrag")
        trace = exported["agent_trace"]
        self.assertEqual(trace["stop_reason"], "answered")
        self.assertEqual(trace["strategy_changes"], ["lookup_to_neighborhood"])
        self.assertEqual(trace["total_tool_calls"], 1)
        self.assertEqual(trace["total_steps"], 2)
        self.assertEqual(trace["tools"][0]["tool"], "retrieve_spec")
        self.assertEqual(trace["tools"][0]["arguments"]["operation"], "lookup_event")
        self.assertEqual(trace["tool_observations"][0]["elapsed_ms"], 12.25)
        self.assertIn("call-1:fact:nations:Q1", trace["tool_observations"][0]["evidence_ids"])
        json.dumps(exported)

    def test_harness_preserves_tokens_latency_and_citations(self) -> None:
        result = HarnessResult(
            system_name="agentic_graphrag",
            question="How many nations?",
            answer="23",
            citations=[Citation(evidence_id="e1", document_id="Q1", chunk_id="Q1::c000", event_id="Q1")],
            latency_ms=55.5,
            retrieval_metadata={
                "trace": _trace().to_dict(),
                "pipeline_notes": {
                    "model_calls": 2,
                    "tokens": 14,
                    "tokens_unknown": True,
                    "failure_class": None,
                    "attempts": [
                        {"ok": False, "tokens": 3, "usage_known": True},
                        {"ok": True, "tokens": 11, "usage_known": True},
                    ],
                },
                "stop_reason": "answered",
                "total_tool_calls": 1,
            },
            evidence_metadata=[
                {
                    "evidence_id": "e1",
                    "chunk_id": "Q1::c000",
                    "document_id": "Q1",
                    "event_id": "Q1",
                }
            ],
            status="answered",
            timings=PipelineTimings(retrieval_ms=12.25, generation_ms=40.0, total_ms=55.5),
        )
        original = copy.deepcopy(result.to_dict())
        exported = export_submission_record(result, question_id="pub-001")
        self.assertEqual(exported["answer"], "23")
        self.assertEqual(exported["total_tokens"], 14)
        self.assertEqual(exported["model_calls"], 2)
        self.assertTrue(exported["tokens_unknown"])
        self.assertEqual(exported["latency_ms"], 55.5)
        self.assertEqual(exported["citation_ids"], ["e1"])
        self.assertEqual(exported["chunk_ids"], ["Q1::c000"])
        self.assertEqual(exported["document_ids"], ["Q1"])
        self.assertEqual(exported["event_ids"], ["Q1"])
        self.assertIsNotNone(exported["agent_trace"])
        self.assertEqual(result.to_dict(), original)
        exported["answer"] = "mutated"
        self.assertEqual(result.answer, "23")

    def test_scored_row_drops_gold_and_correctness(self) -> None:
        row = {
            "qid": "pub-001",
            "system_name": "graphrag",
            "question": "How many nations?",
            "answer": "23",
            "expected": ["23"],
            "gold_doc_ids": ["Q1"],
            "correctness": True,
            "correctness_exact": True,
            "completeness": True,
            "grounding": True,
            "citation_validity": True,
            "failure_category": None,
            "latency_ms": 10.0,
            "tokens": 9,
            "model_calls": 1,
            "citations": [{"evidence_id": "e1", "event_id": "Q1"}],
            "retrieval_methods": ["gsql:lookup_event"],
        }
        exported = export_submission_record(row)
        blob = json.dumps(exported)
        self.assertEqual(exported["question_id"], "pub-001")
        self.assertEqual(exported["pipeline"], "graphrag")
        for key in EXCLUDED_KEYS:
            self.assertNotIn(key, exported)
            self.assertNotIn(f'"{key}"', blob)
        self.assertNotIn("23", json.dumps({k: exported[k] for k in exported if k != "answer"}))
        self.assertEqual(exported["answer"], "23")
        self.assertEqual(exported["total_tokens"], 9)
        self.assertIsNone(exported["agent_trace"])

    def test_rag_export_works_without_agent_trace(self) -> None:
        result = HarnessResult(
            system_name="rag",
            question="How many nations?",
            answer="23",
            citations=[Citation(evidence_id="rag:sparse:Q1::c000", chunk_id="Q1::c000", document_id="Q1")],
            latency_ms=8.0,
            retrieval_metadata={
                "method": "sparse",
                "hit_count": 1,
                "generator_notes": {"model_calls": 1, "tokens": 7, "tokens_unknown": False},
            },
            evidence_metadata=[{"evidence_id": "rag:sparse:Q1::c000", "chunk_id": "Q1::c000", "document_id": "Q1"}],
            status="answered",
        )
        exported = export_submission_record(result, question_id="pub-010")
        self.assertEqual(exported["pipeline"], "rag")
        self.assertIsNone(exported["agent_trace"])
        self.assertEqual(exported["retrieval_methods"], ["sparse"])
        self.assertEqual(exported["total_tokens"], 7)
        self.assertEqual(exported["latency_ms"], 8.0)
        self.assertIn("Q1::c000", exported["chunk_ids"])

    def test_graphrag_export_works_without_agent_trace(self) -> None:
        result = HarnessResult(
            system_name="graphrag",
            question="How many nations?",
            answer="23",
            citations=[Citation(evidence_id="call-1:fact:nations:Q1", event_id="Q1")],
            latency_ms=6.0,
            retrieval_metadata={
                "retrieval_method": "gsql:lookup_event",
                "pipeline_notes": {"fixed_policy": True, "graph_retriever_calls": 1, "tokens": 5, "model_calls": 1},
            },
            evidence_metadata=[{"evidence_id": "call-1:fact:nations:Q1", "event_id": "Q1"}],
            status="answered",
        )
        exported = export_submission_record(result, question_id="pub-010")
        self.assertEqual(exported["pipeline"], "graphrag")
        self.assertIsNone(exported["agent_trace"])
        self.assertEqual(exported["retrieval_methods"], ["gsql:lookup_event"])
        self.assertEqual(exported["event_ids"], ["Q1"])

    def test_agentic_export_contains_rich_trace(self) -> None:
        result = HarnessResult(
            system_name="agentic_graphrag",
            question="How many nations?",
            answer="23",
            citations=[Citation(evidence_id="e1")],
            latency_ms=20.0,
            retrieval_metadata={"trace": _trace().to_dict()},
            evidence_metadata=[{"evidence_id": "e1"}],
            status="answered",
        )
        exported = export_submission_record(result, question_id="pub-002")
        trace = exported["agent_trace"]
        self.assertGreaterEqual(len(trace["plan_steps"]), 1)
        self.assertGreaterEqual(len(trace["tools"]), 1)
        self.assertGreaterEqual(len(trace["tool_observations"]), 1)
        self.assertTrue(trace["strategy_changes"])
        self.assertEqual(trace["stop_reason"], "answered")

    def test_exporter_has_no_hidden_data_dependency(self) -> None:
        import evaluation.trace_export as module

        source = inspect.getsource(module)
        for token in (
            "eval_hidden",
            "phase11_holdout",
            "eval_holdout",
            "official_hidden",
            "hidden50",
            "hidden_submission",
            "evaluation.holdout",
        ):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
