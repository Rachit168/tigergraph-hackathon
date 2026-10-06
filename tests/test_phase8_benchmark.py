"""Phase 8 public three-way benchmark tests. No live graph mutation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from answering.models import (
    Citation,
    CitedAnswer,
    GeneratorResult,
    PackedContextItem,
    PipelineTimings,
)
from answering.packer import ContextPacker
from evaluation.benchmark import score_harness_result
from evaluation.harness import (
    AgenticGraphRAGAdapter,
    AgenticGraphRAGPlaceholder,
    GraphRAGAdapter,
    HarnessResult,
    RAGAdapter,
)
from evaluation.three_way import (
    evaluate_three_way,
    load_public_questions,
    make_three_way_harness,
    write_benchmark_artifacts,
)
from ingestion.paths import DEFAULT_PUBLIC_QUESTIONS_PATH
from retrieval.agentic.pipeline import AgentResult
from retrieval.agentic.trace import AgentTrace
from retrieval.graphrag.models import GraphEvidence, GraphRef, GraphRetrievalResult
from retrieval.rag.models import RetrievalHit, RetrievalResult


class FakeTextRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        self.calls.append((query, kwargs))
        return RetrievalResult(
            query=query,
            method=str(kwargs.get("method") or "sparse"),
            hits=[
                RetrievalHit(
                    chunk_id="c1",
                    document_id="Q1",
                    document_title="doc",
                    score=1.0,
                    rank=1,
                    text="nations: 23",
                    section="infobox",
                    kind="infobox",
                    event_id="Q1",
                    retrieval_method="bm25",
                )
            ],
            params={"top_k": kwargs.get("top_k")},
        )


class SilentGenerator:
    def generate(self, request):
        return GeneratorResult(answer_text="", status="abstained", warnings=("no_explicit_structured_answer_fact",))


class OrphanGenerator:
    def generate(self, request):
        return GeneratorResult(answer_text="guess", citation_ids=("missing",))


class MappingGraphPipeline:
    def __init__(self, answers: dict[str, CitedAnswer]) -> None:
        self.answers = answers
        self.calls: list[tuple] = []

    def run(self, question: str, qtype: str | None = None) -> CitedAnswer:
        self.calls.append((question, qtype))
        return self.answers[question]


class MappingAgentPipeline:
    def __init__(self, results: dict[str, AgentResult]) -> None:
        self.results = results
        self.calls: list[tuple] = []

    def run(self, question: str, qtype: str | None = None) -> AgentResult:
        self.calls.append((question, qtype))
        return self.results[question]


def _graph_cited(
    question: str,
    *,
    answer: str,
    status: str,
    operation: str = "lookup_event",
    cardinality: str = "one",
    event_ids: list[str] | None = None,
    notes: dict | None = None,
    pipeline_notes: dict | None = None,
    facts: list[GraphEvidence] | None = None,
    warnings: list[str] | None = None,
) -> CitedAnswer:
    ids = event_ids or (["Q1"] if status == "supported" or status == "answered" else [])
    retrieval_status = {
        "answered": "supported",
        "ambiguous": "ambiguous",
        "not_found": "not_found",
        "invalid_citations": "supported",
    }.get(status, "supported")
    evidence = facts or []
    if answer and not evidence:
        evidence = [
            GraphEvidence(
                evidence_id="e1",
                evidence_type="fact",
                retrieval_method=f"gsql:{operation}",
                tool_call_id="call",
                graph_refs=[GraphRef(vertex_type="Event", vertex_id=ids[0] if ids else None)],
                event_id=ids[0] if ids else None,
                field_name="nations",
                value=answer,
            )
        ]
    retrieval_notes = dict(notes or {})
    retrieval = GraphRetrievalResult(
        operation=operation,
        query_name=operation,
        status=retrieval_status,
        cardinality=cardinality,
        tool_call_id="call",
        retrieval_method=f"gsql:{operation}",
        event_ids=list(ids),
        facts=evidence,
        notes=retrieval_notes,
    )
    citations = []
    if status == "answered" and evidence:
        packed = PackedContextItem(
            evidence_id=evidence[0].evidence_id,
            evidence_type="fact",
            retrieval_method=evidence[0].retrieval_method,
            tool_call_id="call",
            event_id=evidence[0].event_id,
            field_name=evidence[0].field_name,
            value=evidence[0].value,
        )
        citations = [Citation.from_context_item(packed)]
    if status == "invalid_citations":
        citations = [Citation(evidence_id="not-retrieved")]
    return CitedAnswer(
        question=question,
        answer_text=answer if status == "answered" else "",
        citations=citations,
        retrieval_result=retrieval,
        evidence_used=[],
        status=status,
        warnings=list(warnings or []),
        notes=dict(pipeline_notes or {"fixed_policy": True, "graph_retriever_calls": 1, "followup_retrievals": 0}),
        timings=PipelineTimings(total_ms=4.0),
    )


def _agent_result(
    question: str,
    cited: CitedAnswer,
    *,
    tool_calls: int = 1,
    steps: int = 1,
    followups: int = 0,
    stop_reason: str = "answered",
    methods: list[str] | None = None,
    tools: list[str] | None = None,
) -> AgentResult:
    cited.notes = {
        **dict(cited.notes),
        "agentic_policy": True,
        "graph_retriever_calls": tool_calls,
        "followup_retrievals": followups,
        "stop_reason": stop_reason,
    }
    from retrieval.agentic.tools import ToolObservation

    observations = [
        ToolObservation(
            tool_call_id=f"agent:{name}:001",
            tool=name,
            arguments={},
            reason="test",
            success=True,
            started_ms=0.0,
            elapsed_ms=1.0,
            fingerprint=name,
            retrieval_method=(methods or ["gsql:lookup_event"])[0],
        )
        for name in (tools or ["retrieve_spec"])
    ]
    trace = AgentTrace(
        question=question,
        interpreted={"qtype": "lookup"},
        tool_calls=observations,
        stop_reason=stop_reason,
        total_steps=steps,
        total_tool_calls=tool_calls,
        retrieval_methods=list(methods or ["gsql:lookup_event"]),
    )
    return AgentResult(answer=cited, trace=trace, state=None)  # type: ignore[arg-type]


class LoadPublicQuestionsTests(unittest.TestCase):
    def test_all_100_public_questions_load(self) -> None:
        records = load_public_questions()
        self.assertEqual(len(records), 100)
        self.assertTrue(DEFAULT_PUBLIC_QUESTIONS_PATH.exists())
        qids = [row["qid"] for row in records]
        self.assertEqual(len(qids), len(set(qids)))
        self.assertEqual(qids, sorted(qids))
        qtypes = {row["qtype"] for row in records}
        self.assertEqual(qtypes, {"lookup", "aggregation", "superlative", "temporal", "multi_hop"})
        for row in records:
            self.assertIn("question", row)
            self.assertIn("answer", row)


class BenchmarkFairnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lookup_q = "How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?"
        self.agg_q = "According to the provided corpus, how many cycling events at the 2000 Summer Olympics had more than 30 competitors?"
        self.laura_q = "Who won the gold medal in the event held at Laura Biathlon & Ski Complex on 22 February 2014?"
        event_ids = [f"Q{i}" for i in range(4)]
        graph_answers = {
            self.lookup_q: _graph_cited(self.lookup_q, answer="23", status="answered"),
            self.agg_q: _graph_cited(
                self.agg_q,
                answer="4",
                status="answered",
                operation="count_over_threshold",
                cardinality="complete_set",
                event_ids=event_ids,
                notes={"truncated_set": False, "complete_set": True},
                facts=[
                    GraphEvidence(
                        evidence_id="count",
                        evidence_type="fact",
                        retrieval_method="gsql:count_over_threshold",
                        tool_call_id="call",
                        field_name="count",
                        value="4",
                    )
                ],
            ),
            self.laura_q: _graph_cited(
                self.laura_q,
                answer="",
                status="ambiguous",
                operation="events_at_venue_date",
                cardinality="ambiguous",
                event_ids=["Qa", "Qb"],
                notes={"answer_suppressed": True},
            ),
        }
        agent_results = {
            self.lookup_q: _agent_result(
                self.lookup_q,
                _graph_cited(self.lookup_q, answer="23", status="answered"),
                stop_reason="answered",
            ),
            self.agg_q: _agent_result(
                self.agg_q,
                _graph_cited(
                    self.agg_q,
                    answer="4",
                    status="answered",
                    operation="count_over_threshold",
                    cardinality="complete_set",
                    event_ids=event_ids,
                    notes={"truncated_set": False, "complete_set": True},
                    facts=[
                        GraphEvidence(
                            evidence_id="count",
                            evidence_type="fact",
                            retrieval_method="gsql:count_over_threshold",
                            tool_call_id="call",
                            field_name="count",
                            value="4",
                        )
                    ],
                ),
                stop_reason="answered",
            ),
            self.laura_q: _agent_result(
                self.laura_q,
                _graph_cited(
                    self.laura_q,
                    answer="",
                    status="ambiguous",
                    operation="events_at_venue_date",
                    cardinality="ambiguous",
                    event_ids=["Qa", "Qb"],
                    notes={"answer_suppressed": True},
                ),
                tool_calls=2,
                steps=2,
                followups=1,
                stop_reason="ambiguous",
                methods=["gsql:events_at_venue_date", "gsql:chunks_for_events"],
                tools=["retrieve_spec", "supporting_chunks"],
            ),
        }
        self.rag_retriever = FakeTextRetriever()
        self.graph_pipeline = MappingGraphPipeline(graph_answers)
        self.agent_pipeline = MappingAgentPipeline(agent_results)
        self.harness = make_three_way_harness(
            rag=RAGAdapter(self.rag_retriever, ContextPacker(), SilentGenerator()),
            graphrag=GraphRAGAdapter(self.graph_pipeline),
            agentic=AgenticGraphRAGAdapter(self.agent_pipeline),
        )

    def _records(self) -> list[dict]:
        return [
            {
                "qid": "pub-001",
                "qtype": "lookup",
                "question": self.lookup_q,
                "answer": ["23"],
                "gold_doc_ids": ["Q1"],
            },
            {
                "qid": "pub-agg",
                "qtype": "aggregation",
                "question": self.agg_q,
                "answer": ["4"],
                "gold_doc_ids": ["Q0", "Q1", "Q2", "Q3"],
            },
            {
                "qid": "pub-laura",
                "qtype": "multi_hop",
                "question": self.laura_q,
                "answer": ["Someone"],
                "gold_doc_ids": ["Qa"],
            },
        ]

    def test_placeholder_is_rejected_for_public_benchmark(self) -> None:
        with self.assertRaises(TypeError):
            make_three_way_harness(
                rag=RAGAdapter(FakeTextRetriever(), ContextPacker(), SilentGenerator()),
                graphrag=GraphRAGAdapter(self.graph_pipeline),
                agentic=AgenticGraphRAGPlaceholder(),  # type: ignore[arg-type]
            )
        self.assertIsInstance(self.harness.systems["agentic_graphrag"], AgenticGraphRAGAdapter)
        self.assertNotIsInstance(self.harness.systems["agentic_graphrag"], AgenticGraphRAGPlaceholder)

    def test_gold_is_not_passed_into_systems(self) -> None:
        evaluate_three_way(self._records(), self.harness, config={"questions_name": "eval_public.jsonl"})
        for question, qtype in self.graph_pipeline.calls:
            self.assertIsInstance(question, str)
            self.assertIn(qtype, {"lookup", "aggregation", "multi_hop"})
        for args in self.rag_retriever.calls:
            self.assertEqual(set(args[1]), {"method", "top_k"})
        joined = json.dumps(self.graph_pipeline.calls) + json.dumps(self.agent_pipeline.calls)
        self.assertNotIn("Someone", joined)
        self.assertNotIn("gold_doc_ids", joined)

    def test_qid_and_system_are_associated(self) -> None:
        payload = evaluate_three_way(self._records(), self.harness)
        by_key = {(row["qid"], row["system_name"]): row for row in payload["rows"]}
        self.assertEqual(len(payload["rows"]), 9)
        self.assertEqual(by_key[("pub-001", "graphrag")]["answer"], "23")
        self.assertTrue(by_key[("pub-001", "graphrag")]["correctness"])
        self.assertEqual(by_key[("pub-001", "agentic_graphrag")]["answer"], "23")
        self.assertEqual(by_key[("pub-001", "rag")]["answer"], "")
        self.assertFalse(by_key[("pub-001", "rag")]["correctness"])

    def test_aggregation_completeness_is_preserved(self) -> None:
        payload = evaluate_three_way(self._records(), self.harness)
        graph = next(row for row in payload["rows"] if row["qid"] == "pub-agg" and row["system_name"] == "graphrag")
        agent = next(row for row in payload["rows"] if row["qid"] == "pub-agg" and row["system_name"] == "agentic_graphrag")
        rag = next(row for row in payload["rows"] if row["qid"] == "pub-agg" and row["system_name"] == "rag")
        self.assertTrue(graph["completeness"])
        self.assertEqual(graph["completeness_reason"], "complete_set_preserved")
        self.assertFalse(graph["truncated_set"])
        self.assertTrue(agent["completeness"])
        self.assertIsNone(rag["completeness"])
        self.assertEqual(rag["completeness_reason"], "rag_top_k_has_no_complete_set_semantics")
        self.assertIsNone(rag["tool_calls"])
        self.assertIsNone(rag["stop_reason"])

    def test_ambiguity_remains_suppressed(self) -> None:
        payload = evaluate_three_way(self._records(), self.harness)
        for system in ("graphrag", "agentic_graphrag"):
            row = next(item for item in payload["rows"] if item["qid"] == "pub-laura" and item["system_name"] == system)
            self.assertEqual(row["answer_status"], "ambiguous")
            self.assertEqual(row["answer"], "")
            self.assertTrue(row["answer_suppressed"])
            self.assertFalse(row["correctness"])
            self.assertEqual(row["failure_category"], "ambiguous_suppressed")
            self.assertTrue(row["grounding"])
            self.assertEqual(row["grounding_status"], "faithful_abstention")
        agent = next(item for item in payload["rows"] if item["qid"] == "pub-laura" and item["system_name"] == "agentic_graphrag")
        self.assertEqual(agent["stop_reason"], "ambiguous")
        self.assertEqual(agent["followups"], 1)

    def test_citation_validity_fail_closed(self) -> None:
        rag = RAGAdapter(FakeTextRetriever(), ContextPacker(), OrphanGenerator())
        cited = _graph_cited(self.lookup_q, answer="23", status="invalid_citations")
        harness = make_three_way_harness(
            rag=rag,
            graphrag=GraphRAGAdapter(MappingGraphPipeline({self.lookup_q: cited})),
            agentic=AgenticGraphRAGAdapter(
                MappingAgentPipeline(
                    {
                        self.lookup_q: _agent_result(
                            self.lookup_q,
                            _graph_cited(self.lookup_q, answer="23", status="invalid_citations"),
                            stop_reason="answered",
                        )
                    }
                )
            ),
        )
        payload = evaluate_three_way(self._records()[:1], harness)
        for row in payload["rows"]:
            self.assertEqual(row["answer_status"], "invalid_citations", row)
            self.assertFalse(row["citation_validity"], row)
            self.assertEqual(row["answer"], "")
            self.assertEqual(row["failure_category"], "citation_failure")

    def test_one_system_failure_does_not_abort_others(self) -> None:
        class BoomRetriever(FakeTextRetriever):
            def retrieve(self, query: str, **kwargs):
                raise RuntimeError("rag-down")

        harness = make_three_way_harness(
            rag=RAGAdapter(BoomRetriever(), ContextPacker(), SilentGenerator()),
            graphrag=GraphRAGAdapter(self.graph_pipeline),
            agentic=AgenticGraphRAGAdapter(self.agent_pipeline),
        )
        payload = evaluate_three_way(self._records()[:1], harness)
        by_system = {row["system_name"]: row for row in payload["rows"]}
        self.assertEqual(by_system["rag"]["answer_status"], "error")
        self.assertEqual(by_system["rag"]["failure_category"], "system_failure")
        self.assertTrue(by_system["graphrag"]["correctness"])
        self.assertTrue(by_system["agentic_graphrag"]["correctness"])

    def test_output_is_serializable_and_run_id_is_stable(self) -> None:
        first = evaluate_three_way(
            self._records(),
            self.harness,
            config={"questions_name": "eval_public.jsonl", "generator": "DeterministicGroundedGenerator"},
        )
        second = evaluate_three_way(
            self._records(),
            self.harness,
            config={"questions_name": "eval_public.jsonl", "generator": "DeterministicGroundedGenerator"},
        )
        json.dumps(first, sort_keys=True, default=str)
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["n_questions"], 3)
        stripped = []
        for payload in (first, second):
            rows = []
            for row in payload["rows"]:
                copy = dict(row)
                copy.pop("latency_ms", None)
                rows.append(copy)
            stripped.append(rows)
        self.assertEqual(stripped[0], stripped[1])

    def test_public_questions_all_run_through_three_systems(self) -> None:
        records = load_public_questions()
        default = _graph_cited("q", answer="", status="not_found")

        class AnyGraph:
            def run(self, question: str, qtype: str | None = None) -> CitedAnswer:
                return default

        class AnyAgent:
            def run(self, question: str, qtype: str | None = None) -> AgentResult:
                return _agent_result(question, _graph_cited(question, answer="", status="not_found"), stop_reason="not_found")

        harness = make_three_way_harness(
            rag=RAGAdapter(FakeTextRetriever(), ContextPacker(), SilentGenerator()),
            graphrag=GraphRAGAdapter(AnyGraph()),
            agentic=AgenticGraphRAGAdapter(AnyAgent()),
        )
        payload = evaluate_three_way(
            records,
            harness,
            config={"questions_name": "eval_public.jsonl"},
        )
        self.assertEqual(payload["n_questions"], 100)
        self.assertEqual(payload["n_rows"], 300)
        systems = {row["system_name"] for row in payload["rows"]}
        self.assertEqual(systems, {"rag", "graphrag", "agentic_graphrag"})
        counts = {}
        for row in payload["rows"]:
            counts.setdefault(row["system_name"], set()).add(row["qid"])
        for name in systems:
            self.assertEqual(len(counts[name]), 100)

    def test_write_artifacts_round_trip(self) -> None:
        payload = evaluate_three_way(self._records(), self.harness)
        out = Path("tests") / "_tmp_phase8_artifacts"
        try:
            paths = write_benchmark_artifacts(payload, out)
            loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(loaded["run_id"], payload["run_id"])
            self.assertIn("Public three-way benchmark", paths["report"].read_text(encoding="utf-8"))
            csv_text = paths["csv"].read_text(encoding="utf-8")
            self.assertIn("qid,qtype,system_name", csv_text)
        finally:
            for path in out.glob("*"):
                path.unlink()
            if out.exists():
                out.rmdir()


class ScoreSchemaTests(unittest.TestCase):
    def test_missing_agentic_metrics_are_null_for_rag(self) -> None:
        result = HarnessResult(
            system_name="rag",
            question="q",
            answer="",
            citations=[],
            latency_ms=1.0,
            retrieval_metadata={"method": "sparse", "hit_count": 10},
            evidence_metadata=[{"evidence_id": "rag:sparse:c1", "document_id": "Q1"}],
            status="abstained",
        )
        row = score_harness_result(
            {"qid": "pub-001", "qtype": "lookup", "question": "q", "answer": ["23"], "gold_doc_ids": ["Q1"]},
            result,
        )
        self.assertIsNone(row["tool_calls"])
        self.assertIsNone(row["agent_steps"])
        self.assertIsNone(row["followups"])
        self.assertIsNone(row["stop_reason"])
        self.assertIsNone(row["completeness"])
        self.assertEqual(row["model_calls"], 0)
        self.assertEqual(row["tokens"], 0)
        self.assertTrue(row["citation_validity"])
        self.assertEqual(row["gold_doc_recall"], 1.0)


class ScoringRepresentationTests(unittest.TestCase):
    def test_gold_list_matches_scalar_string_but_not_duplicated_join(self) -> None:
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
        self.assertFalse(dup_row["correctness"])
        self.assertFalse(dup_row["correctness_exact"])
        self.assertEqual(dup_row["failure_category"], "incorrect")


if __name__ == "__main__":
    unittest.main()
