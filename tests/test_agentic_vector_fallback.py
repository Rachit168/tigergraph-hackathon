from __future__ import annotations

import unittest

from answering.generator import DeterministicGroundedGenerator
from answering.packer import ContextPacker
from retrieval.agentic.pipeline import AgenticGraphRAGPipeline
from retrieval.graphrag.models import GraphEvidence, GraphRef, GraphRetrievalResult
from retrieval.graphrag.parser import GraphRAGQuestionParser
from retrieval.rag.models import RetrievalHit, RetrievalResult
from retrieval.structured.models import QuerySpec
from ui.viewmodels import investigation_from_export


def _evidence(
    evidence_id: str,
    evidence_type: str,
    *,
    event_id: str | None = "Q1",
    field_name: str | None = None,
    value: str | None = None,
    method: str = "gsql:test",
    text: str | None = None,
) -> GraphEvidence:
    return GraphEvidence(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        retrieval_method=method,
        tool_call_id="test",
        graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id)],
        event_id=event_id,
        field_name=field_name,
        value=value,
        text=text,
    )


def _result(
    operation: str,
    status: str,
    *,
    event_ids: list[str] | None = None,
    facts: list[GraphEvidence] | None = None,
    edges: list[GraphEvidence] | None = None,
    chunks: list[GraphEvidence] | None = None,
) -> GraphRetrievalResult:
    ids = list(event_ids or [])
    cardinality = "one" if len(ids) == 1 else "multiple" if ids else "empty"
    return GraphRetrievalResult(
        operation=operation,
        query_name=operation,
        status=status,
        cardinality=cardinality,
        tool_call_id="test",
        retrieval_method=f"gsql:{operation}",
        event_ids=ids,
        facts=list(facts or []),
        edges=list(edges or []),
        chunks=list(chunks or []),
    )


class _Parser:
    def __init__(self, spec: QuerySpec) -> None:
        self.spec = spec

    def parse(self, question: str, qtype: str | None = None) -> QuerySpec:
        del question, qtype
        return self.spec


class _GraphRetriever:
    def __init__(self, primary: GraphRetrievalResult, neighborhood: GraphRetrievalResult) -> None:
        self.primary = primary
        self.neighborhood = neighborhood
        self.calls: list[str] = []

    def retrieve(self, spec: QuerySpec, *, include_chunks: bool = False, max_extra_chunks: int = 0):
        del spec, include_chunks, max_extra_chunks
        self.calls.append("retrieve_spec")
        return self.primary

    def event_neighborhood(self, event_id: str):
        del event_id
        self.calls.append("event_neighborhood")
        return self.neighborhood

    def supporting_chunks(self, event_ids, *, max_extra: int = 0, tool_call_id: str | None = None):
        del event_ids, max_extra, tool_call_id
        self.calls.append("supporting_chunks")
        return _result("chunks_for_events", "not_found", event_ids=["Q1"])


class _VectorRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    def retrieve(self, query: str, *, top_k: int, method: str) -> RetrievalResult:
        self.calls.append((query, top_k, method))
        hit = RetrievalHit(
            chunk_id="Q1::c000",
            document_id="Q1",
            document_title="Q1",
            score=-0.1,
            rank=1,
            text="Gold: GBR",
            section="lead",
            kind="lead",
            event_id="Q1",
            retrieval_method="tigergraph_vector",
        )
        return RetrievalResult(
            query=query,
            method="tigergraph_vector",
            hits=[hit],
            elapsed_ms=3.0,
            params={"backend": "tigergraph_vector", "top_k": top_k},
        )


class AgenticVectorFallbackTests(unittest.TestCase):
    def test_fallback_is_one_bounded_action_after_chunk_gap(self) -> None:
        spec = QuerySpec(
            qtype="multi_hop",
            operation="events_at_venue_date",
            raw_question="Who won gold at venue?",
            matched_template=True,
            venue="Venue",
            date_text="1 January 2012",
            year=2012,
            requested_field="gold",
        )
        primary = _result(
            "events_at_venue_date",
            "supported",
            event_ids=["Q1"],
            facts=[_evidence("p:gold", "fact", field_name="gold_raw", value="GBR")],
        )
        neighborhood = _result(
            "event_neighborhood",
            "supported",
            event_ids=["Q1"],
            edges=[_evidence("n:venue", "edge", field_name="HELD_AT", value="Venue")],
        )
        graph = _GraphRetriever(primary, neighborhood)
        vector = _VectorRetriever()
        pipeline = AgenticGraphRAGPipeline(
            GraphRAGQuestionParser(_Parser(spec)),
            graph,
            ContextPacker(),
            DeterministicGroundedGenerator(),
            vector_retriever=vector,
        )

        result = pipeline.run(spec.raw_question, qtype="multi_hop")

        self.assertEqual(len(vector.calls), 1)
        self.assertEqual(vector.calls[0][1:], (5, "tigergraph_vector"))
        self.assertEqual(result.trace.total_tool_calls, 4)
        self.assertEqual(
            [observation.tool for observation in result.trace.tool_calls],
            ["retrieve_spec", "event_neighborhood", "supporting_chunks", "vector_search"],
        )
        self.assertEqual(result.status, "answered")
        self.assertIn("tigergraph_vector", result.trace.retrieval_methods)
        self.assertTrue(result.state.vector_search_used)
        self.assertTrue(result.state.has_chunks)

    def test_public_trace_and_ui_mark_vector_as_follow_up(self) -> None:
        view = investigation_from_export(
            {
                "schema_version": 1,
                "pipeline": "agentic_graphrag",
                "question": "Who won gold at venue?",
                "answer": "GBR",
                "status": "answered",
                "citations": [],
                "evidence": [],
                "agent_trace": {
                    "interpreted": {"qtype": "multi_hop"},
                    "tools": [
                        {"tool": "retrieve_spec"},
                        {"tool": "vector_search"},
                    ],
                    "tool_observations": [
                        {
                            "tool": "retrieve_spec",
                            "success": True,
                            "result_status": "supported",
                            "evidence_ids": ["primary"],
                            "retrieval_method": "gsql:events_at_venue_date",
                        },
                        {
                            "tool": "vector_search",
                            "success": True,
                            "result_status": "supported",
                            "evidence_ids": ["vector:chunk:Q1::c000"],
                            "retrieval_method": "tigergraph_vector",
                            "reason": "graph retrieval left a provenance gap; try one bounded vector fallback",
                        },
                    ],
                    "total_tool_calls": 2,
                    "total_steps": 2,
                    "retrieval_methods": ["gsql:events_at_venue_date", "tigergraph_vector"],
                    "stop_reason": "answered",
                    "follow_up_decisions": ["graph retrieval left a provenance gap; try one bounded vector fallback"],
                },
            }
        )

        self.assertTrue(view["agent"]["follow_up_occurred"])
        self.assertEqual(view["steps"][1]["action"], "vector_search")
        self.assertEqual(view["steps"][1]["kind"], "follow_up")
        self.assertIn("vector fallback", view["continue_explanation"])


if __name__ == "__main__":
    unittest.main()
