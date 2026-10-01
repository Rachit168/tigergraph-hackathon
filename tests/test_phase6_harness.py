"""Phase 6 common three-way evaluation harness."""

from __future__ import annotations

import unittest

from answering.models import (
    Citation,
    CitedAnswer,
    GeneratorResult,
    PackedContextItem,
    PipelineTimings,
)
from answering.packer import ContextPacker
from evaluation.harness import (
    AgenticGraphRAGPlaceholder,
    GraphRAGAdapter,
    HarnessResult,
    RAGAdapter,
    ThreeWayEvaluationHarness,
)
from retrieval.graphrag.models import GraphEvidence, GraphRef, GraphRetrievalResult
from retrieval.rag.models import RetrievalHit, RetrievalResult


class FakeTextRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls = []

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        self.calls.append((query, kwargs))
        return self.result


class FirstEvidenceGenerator:
    def generate(self, request):
        if not request.context.items:
            return GeneratorResult(answer_text="", status="abstained")
        item = request.context.items[0]
        return GeneratorResult(
            answer_text="23",
            citation_ids=(item.evidence_id,),
        )


class OrphanGenerator:
    def generate(self, request):
        return GeneratorResult(answer_text="unsupported", citation_ids=("orphan",))


class FakeGraphPipeline:
    def __init__(self, answer: CitedAnswer) -> None:
        self.answer = answer
        self.calls = []

    def run(self, question: str, qtype: str | None = None) -> CitedAnswer:
        self.calls.append((question, qtype))
        return self.answer


def _text_result() -> RetrievalResult:
    return RetrievalResult(
        query="Judo question",
        method="sparse",
        hits=[
            RetrievalHit(
                chunk_id="Q26217865::c000",
                document_id="Q26217865",
                document_title="Judo at the 2016 Summer Olympics – Women's 57 kg",
                score=4.2,
                rank=1,
                text="nations: 23",
                section="infobox",
                kind="infobox",
                event_id="Q26217865",
                retrieval_method="bm25",
                source_url="https://example.test/judo",
            )
        ],
        elapsed_ms=1.2,
        params={"top_k": 10},
    )


def _graph_answer() -> CitedAnswer:
    evidence = GraphEvidence(
        evidence_id="call:fact:nations:Q26217865",
        evidence_type="fact",
        retrieval_method="gsql:lookup_event",
        tool_call_id="call",
        graph_refs=[
            GraphRef(
                vertex_type="Event",
                vertex_id="Q26217865",
                attribute="nations",
            )
        ],
        event_id="Q26217865",
        document_id="Q26217865",
        field_name="nations",
        value="23",
    )
    retrieval = GraphRetrievalResult(
        operation="lookup_event",
        query_name="lookup_event",
        status="supported",
        cardinality="one",
        tool_call_id="call",
        retrieval_method="gsql:lookup_event",
        event_ids=["Q26217865"],
        facts=[evidence],
    )
    packed = PackedContextItem(
        evidence_id=evidence.evidence_id,
        evidence_type=evidence.evidence_type,
        retrieval_method=evidence.retrieval_method,
        tool_call_id=evidence.tool_call_id,
        graph_refs=tuple(evidence.graph_refs),
        event_id=evidence.event_id,
        document_id=evidence.document_id,
        field_name=evidence.field_name,
        value=evidence.value,
    )
    citation = Citation.from_context_item(packed)
    return CitedAnswer(
        question="Judo question",
        answer_text="23",
        citations=[citation],
        retrieval_result=retrieval,
        evidence_used=[packed],
        status="answered",
        timings=PipelineTimings(total_ms=5.0),
    )


class RAGAdapterTests(unittest.TestCase):
    def test_existing_text_retrieval_path_maps_to_common_schema(self) -> None:
        retriever = FakeTextRetriever(_text_result())
        adapter = RAGAdapter(
            retriever,
            ContextPacker(),
            FirstEvidenceGenerator(),
            method="sparse",
            top_k=10,
        )
        result = adapter.run("Judo question")
        self.assertEqual(result.system_name, "rag")
        self.assertEqual(result.answer, "23")
        self.assertEqual(result.status, "answered")
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(
            result.citations[0].evidence_id,
            "rag:sparse:Q26217865::c000",
        )
        self.assertEqual(result.citations[0].chunk_id, "Q26217865::c000")
        self.assertEqual(result.retrieval_metadata["method"], "sparse")
        self.assertEqual(result.evidence_metadata[0]["text"], "nations: 23")
        self.assertEqual(
            retriever.calls,
            [("Judo question", {"method": "sparse", "top_k": 10})],
        )

    def test_rag_orphan_citation_is_fail_closed(self) -> None:
        adapter = RAGAdapter(
            FakeTextRetriever(_text_result()),
            ContextPacker(),
            OrphanGenerator(),
        )
        result = adapter.run("Judo question")
        self.assertEqual(result.status, "invalid_citations")
        self.assertEqual(result.answer, "")
        self.assertEqual(result.citations, [])
        self.assertIn("orphan_citation:orphan", result.errors)


class ThreeWayHarnessTests(unittest.TestCase):
    def test_graph_adapter_preserves_retrieval_and_evidence_metadata(self) -> None:
        pipeline = FakeGraphPipeline(_graph_answer())
        result = GraphRAGAdapter(pipeline).run("Judo question", qtype="lookup")
        self.assertEqual(result.system_name, "graphrag")
        self.assertEqual(result.answer, "23")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.retrieval_metadata["status"], "supported")
        self.assertEqual(
            result.evidence_metadata[0]["evidence_id"],
            "call:fact:nations:Q26217865",
        )
        self.assertEqual(pipeline.calls, [("Judo question", "lookup")])

    def test_agentic_placeholder_is_callable_but_executes_nothing(self) -> None:
        result = AgenticGraphRAGPlaceholder().run("question", qtype="lookup")
        self.assertEqual(result.system_name, "agentic_graphrag")
        self.assertEqual(result.status, "not_implemented")
        self.assertEqual(result.answer, "")
        self.assertEqual(result.citations, [])
        self.assertEqual(result.retrieval_metadata["retrieval_calls"], 0)
        self.assertIn("agentic_planner_not_implemented", result.errors)

    def test_same_record_runs_all_three_independently(self) -> None:
        rag = RAGAdapter(
            FakeTextRetriever(_text_result()),
            ContextPacker(),
            FirstEvidenceGenerator(),
        )
        graph_pipeline = FakeGraphPipeline(_graph_answer())
        harness = ThreeWayEvaluationHarness(
            [rag, GraphRAGAdapter(graph_pipeline), AgenticGraphRAGPlaceholder()]
        )
        results = harness.run_records(
            [{"question": "Judo question", "qtype": "lookup", "answer": ["23"]}]
        )
        self.assertEqual(
            [result.system_name for result in results],
            ["rag", "graphrag", "agentic_graphrag"],
        )
        self.assertEqual(len(graph_pipeline.calls), 1)
        for result in results:
            payload = result.to_dict()
            self.assertEqual(
                set(payload),
                {
                    "system_name",
                    "question",
                    "answer",
                    "citations",
                    "latency_ms",
                    "retrieval_metadata",
                    "evidence_metadata",
                    "status",
                    "errors",
                    "warnings",
                    "timings",
                },
            )
        # Gold fields on public rows are not consumed by the harness.
        self.assertNotIn("expected", results[0].retrieval_metadata)

    def test_unknown_system_and_system_failure_are_isolated(self) -> None:
        class BrokenSystem:
            system_name = "broken"

            def run(self, question: str, qtype: str | None = None):
                raise RuntimeError("boom")

        harness = ThreeWayEvaluationHarness([BrokenSystem()])
        broken, unknown = harness.run_question(
            "question",
            system_names=["broken", "missing"],
        )
        self.assertEqual(broken.status, "error")
        self.assertEqual(broken.errors, ["system_error:RuntimeError"])
        self.assertEqual(unknown.status, "error")
        self.assertEqual(unknown.errors, ["unknown_system:missing"])

    def test_common_result_has_comparable_fields(self) -> None:
        result = HarnessResult(
            system_name="test",
            question="q",
            answer="a",
            citations=[],
            latency_ms=1.0,
            retrieval_metadata={},
            evidence_metadata=[],
            status="answered",
        )
        payload = result.to_dict()
        self.assertEqual(payload["system_name"], "test")
        self.assertIn("timings", payload)
        self.assertIn("errors", payload)
        self.assertIn("warnings", payload)


if __name__ == "__main__":
    unittest.main()
