"""Phase 7 bounded Agentic GraphRAG tests. No live LLM required."""

from __future__ import annotations

import inspect
import json
import threading
import time
import unittest

from answering.generator import DeterministicGroundedGenerator
from answering.models import GeneratorResult
from answering.packer import ContextPacker
from evaluation.harness import AgenticGraphRAGAdapter, ThreeWayEvaluationHarness
from retrieval.agentic.budget import AgentBudget
from retrieval.agentic.pipeline import AgenticGraphRAGPipeline
from retrieval.agentic.planner import BoundedPlanner
from retrieval.agentic.tools import RETRIEVE_SPEC, ToolAction
from retrieval.graphrag.models import GraphEvidence, GraphRef, GraphRetrievalResult
from retrieval.graphrag.parser import GraphRAGQuestionParser
from retrieval.graphrag.pipeline import FixedGraphRAGPipeline
from retrieval.structured.models import QuerySpec


def _spec(qtype: str, operation: str, **fields) -> QuerySpec:
    return QuerySpec(
        qtype=qtype,
        operation=operation,
        raw_question=fields.get("raw_question", "question"),
        matched_template=True,
        **{key: value for key, value in fields.items() if key != "raw_question"},
    )


def _entity(event_id: str, title: str, call: str = "t") -> GraphEvidence:
    return GraphEvidence(
        evidence_id=f"{call}:entity:{event_id}",
        evidence_type="entity",
        retrieval_method="gsql:lookup_event",
        tool_call_id=call,
        graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id)],
        event_id=event_id,
        field_name="title",
        value=title,
        why_retrieved="structured lookup",
    )


def _fact(event_id: str | None, field: str, value: str, call: str = "t", operation: str = "lookup_event") -> GraphEvidence:
    suffix = event_id or "set"
    return GraphEvidence(
        evidence_id=f"{call}:fact:{field}:{suffix}",
        evidence_type="fact",
        retrieval_method=f"gsql:{operation}",
        tool_call_id=call,
        graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id, attribute=field)],
        event_id=event_id,
        field_name=field,
        value=value,
        why_retrieved=f"{operation} returned {field}",
    )


def _result(
    operation: str,
    status: str,
    *,
    event_ids: list[str] | None = None,
    entities: list[GraphEvidence] | None = None,
    facts: list[GraphEvidence] | None = None,
    edges: list[GraphEvidence] | None = None,
    chunks: list[GraphEvidence] | None = None,
    cardinality: str | None = None,
    reason: str | None = None,
    notes: dict | None = None,
) -> GraphRetrievalResult:
    ids = event_ids or []
    card = cardinality
    if card is None:
        if status == "supported" and operation == "count_over_threshold":
            card = "complete_set"
        elif status == "ambiguous":
            card = "ambiguous"
        elif not ids:
            card = "empty"
        elif len(ids) == 1:
            card = "one"
        else:
            card = "multiple"
    payload_notes = dict(notes or {})
    if status != "supported":
        payload_notes.setdefault("answer_suppressed", True)
    if card == "complete_set":
        payload_notes.setdefault("truncated_set", False)
    return GraphRetrievalResult(
        operation=operation,
        query_name=operation,
        status=status,
        cardinality=card,
        tool_call_id="t",
        retrieval_method=f"gsql:{operation}",
        reason=reason,
        event_ids=list(ids),
        entities=list(entities or []),
        facts=list(facts or []),
        edges=list(edges or []),
        chunks=list(chunks or []),
        notes=payload_notes,
    )


class SpyParser:
    def __init__(self, spec: QuerySpec) -> None:
        self.spec = spec
        self.calls: list[tuple[str, str | None]] = []

    def parse(self, question: str, qtype: str | None = None) -> QuerySpec:
        self.calls.append((question, qtype))
        return self.spec


class FakeRetriever:
    def __init__(self, **results: GraphRetrievalResult) -> None:
        self.results = results
        self.calls: list[tuple] = []
        self.thread_ids: dict[str, int] = {}

    def retrieve(self, spec: QuerySpec, *, include_chunks: bool = False, max_extra_chunks: int = 0):
        self.calls.append(("retrieve", spec.operation, include_chunks, max_extra_chunks))
        return self.results["retrieve_spec"]

    def supporting_chunks(self, event_ids, *, max_extra: int = 0, tool_call_id: str | None = None):
        self.calls.append(("supporting_chunks", list(event_ids), max_extra))
        self.thread_ids["chunks"] = threading.get_ident()
        time.sleep(0.03)
        return self.results.get("supporting_chunks") or _result("chunks_for_events", "not_found")

    def event_neighborhood(self, event_id: str):
        self.calls.append(("event_neighborhood", event_id))
        self.thread_ids["neighborhood"] = threading.get_ident()
        time.sleep(0.03)
        return self.results.get("event_neighborhood") or _result("event_neighborhood", "not_found")


class RepeatPlanner:
    def next_actions(self, state, budget):
        return [
            ToolAction(
                tool=RETRIEVE_SPEC,
                arguments={"operation": state.spec.operation},
                reason="repeat the same lookup",
            )
        ]


class OrphanGenerator:
    def generate(self, request):
        return GeneratorResult(answer_text="guess", citation_ids=("not-retrieved",))


def _pipeline(spec: QuerySpec, retriever: FakeRetriever, **kwargs) -> AgenticGraphRAGPipeline:
    return AgenticGraphRAGPipeline(
        GraphRAGQuestionParser(SpyParser(spec)),
        retriever,
        ContextPacker(),
        kwargs.pop("generator", DeterministicGroundedGenerator()),
        budget=kwargs.pop("budget", AgentBudget()),
        planner=kwargs.pop("planner", BoundedPlanner()),
    )


class LookupPolicyTests(unittest.TestCase):
    def test_lookup_uses_one_structured_retrieve_without_neighborhood(self) -> None:
        spec = _spec(
            "lookup",
            "lookup_nations",
            event_title="Judo at the 2016 Summer Olympics – Women's 57 kg",
            requested_field="nations",
        )
        retriever = FakeRetriever(
            retrieve_spec=_result(
                "lookup_event",
                "supported",
                event_ids=["Q26217865"],
                entities=[_entity("Q26217865", "Judo")],
                facts=[_fact("Q26217865", "nations", "23")],
            )
        )
        result = _pipeline(spec, retriever).run("How many nations?", qtype="lookup")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer.answer_text, "23")
        self.assertEqual(result.trace.stop_reason, "answered")
        self.assertEqual(result.trace.total_tool_calls, 1)
        self.assertEqual(retriever.calls, [("retrieve", "lookup_nations", False, 0)])
        self.assertFalse(any(call[0] == "event_neighborhood" for call in retriever.calls))
        self.assertEqual(result.answer.citations[0].event_id, "Q26217865")
        self.assertIn("gsql:lookup_event", result.trace.retrieval_methods)


class AggregationPolicyTests(unittest.TestCase):
    def test_complete_set_is_not_top_k_truncated(self) -> None:
        event_ids = [f"Q{i}" for i in range(16)]
        spec = _spec(
            "aggregation",
            "count_over_threshold",
            sport="cycling",
            year=2000,
            season="Summer",
            threshold=30,
        )
        retriever = FakeRetriever(
            retrieve_spec=_result(
                "count_over_threshold",
                "supported",
                event_ids=event_ids,
                entities=[_entity(event_id, f"Event {event_id}") for event_id in event_ids],
                facts=[_fact(None, "count", "16", operation="count_over_threshold")],
                notes={"truncated_set": False},
            )
        )
        result = _pipeline(spec, retriever).run("cycling count", qtype="aggregation")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer.answer_text, "16")
        self.assertEqual(result.answer.retrieval_result.event_ids, event_ids)
        self.assertEqual(result.answer.retrieval_result.cardinality, "complete_set")
        self.assertFalse(result.answer.retrieval_result.notes.get("truncated_set"))
        self.assertTrue(result.answer.notes["complete_set_preserved"])
        self.assertEqual(result.trace.total_tool_calls, 1)


class SuperlativePolicyTests(unittest.TestCase):
    def test_superlative_uses_argmax_without_neighborhood(self) -> None:
        spec = _spec(
            "superlative",
            "argmax_competitors",
            sport="athletics",
            year=2016,
            season="Summer",
            requested_field="competitors",
        )
        retriever = FakeRetriever(
            retrieve_spec=_result(
                "argmax_competitors",
                "supported",
                event_ids=["Qmax"],
                entities=[_entity("Qmax", "100 metres")],
                facts=[_fact("Qmax", "competitors", "84", operation="argmax_competitors")],
            )
        )
        result = _pipeline(spec, retriever).run("most competitors", qtype="superlative")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer.answer_text, "100 metres")
        self.assertEqual(retriever.calls, [("retrieve", "argmax_competitors", False, 0)])
        self.assertEqual(result.trace.stop_reason, "answered")
        self.assertEqual(result.trace.total_tool_calls, 1)
        self.assertFalse(any(call[0] == "event_neighborhood" for call in retriever.calls))

    def test_superlative_stops_after_argmax_even_without_competitor_facts(self) -> None:
        spec = _spec(
            "superlative",
            "argmax_competitors",
            sport="athletics",
            year=2008,
            season="Summer",
            requested_field="competitors",
        )
        retriever = FakeRetriever(
            retrieve_spec=_result(
                "argmax_competitors",
                "supported",
                event_ids=["Q1005784"],
                entities=[_entity("Q1005784", "Athletics at the 2008 Summer Olympics – Men's marathon")],
            ),
            supporting_chunks=_result(
                "chunks_for_events",
                "supported",
                event_ids=["Q1005784"],
                chunks=[
                    GraphEvidence(
                        evidence_id="c-max",
                        evidence_type="chunk",
                        retrieval_method="gsql:chunks_for_events",
                        tool_call_id="c",
                        chunk_id="c-max",
                        event_id="Q1005784",
                        text="marathon",
                    )
                ],
            ),
        )
        result = _pipeline(spec, retriever).run("highest competitors", qtype="superlative")
        self.assertEqual(result.trace.stop_reason, "answered")
        self.assertEqual(result.trace.total_tool_calls, 1)
        self.assertEqual(retriever.calls, [("retrieve", "argmax_competitors", False, 0)])
        self.assertEqual(result.state.slot("max_attribute").status, "resolved")
        self.assertEqual(result.answer.answer_text, "Athletics at the 2008 Summer Olympics – Men's marathon")


class TemporalPolicyTests(unittest.TestCase):
    def test_temporal_uses_previous_event_gold(self) -> None:
        spec = _spec(
            "temporal",
            "previous_event_gold",
            sport="athletics",
            event_name="men's pole vault",
            named_year=2016,
            season="Summer",
            requested_field="gold",
        )
        retriever = FakeRetriever(
            retrieve_spec=_result(
                "previous_event_gold",
                "supported",
                event_ids=["Q2012"],
                entities=[_entity("Q2012", "Pole vault 2012")],
                facts=[_fact("Q2012", "gold_raw", "Renaud Lavillenie", operation="previous_event_gold")],
            )
        )
        result = _pipeline(spec, retriever).run("who won before 2016", qtype="temporal")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer.answer_text, "Renaud Lavillenie")
        self.assertEqual(retriever.calls[0][0], "retrieve")
        self.assertEqual(retriever.calls[0][1], "previous_event_gold")


class MultiHopPolicyTests(unittest.TestCase):
    def test_unique_multihop_follows_up_with_typed_neighborhood_in_parallel(self) -> None:
        spec = _spec(
            "multi_hop",
            "events_at_venue_date",
            venue="London Velopark",
            date_text="3 to 4 August",
            year=2012,
            requested_field="gold",
        )
        retriever = FakeRetriever(
            retrieve_spec=_result(
                "events_at_venue_date",
                "supported",
                event_ids=["Q2297633"],
                entities=[_entity("Q2297633", "Team pursuit")],
                facts=[_fact("Q2297633", "gold_raw", "GBR", operation="events_at_venue_date")],
            ),
            event_neighborhood=_result(
                "event_neighborhood",
                "supported",
                event_ids=["Q2297633"],
                facts=[_fact("Q2297633", "gold_raw", "GBR", call="n", operation="event_neighborhood")],
                edges=[
                    GraphEvidence(
                        evidence_id="n:edge:HELD_AT:velopark",
                        evidence_type="edge",
                        retrieval_method="gsql:event_neighborhood",
                        tool_call_id="n",
                        field_name="HELD_AT",
                        value="velopark",
                        event_id="Q2297633",
                        why_retrieved="typed HELD_AT hop from Event",
                    )
                ],
            ),
            supporting_chunks=_result(
                "chunks_for_events",
                "supported",
                event_ids=["Q2297633"],
                chunks=[
                    GraphEvidence(
                        evidence_id="c1",
                        evidence_type="chunk",
                        retrieval_method="gsql:chunks_for_events",
                        tool_call_id="c",
                        chunk_id="c1",
                        source_chunk_id="c1",
                        event_id="Q2297633",
                        text="infobox",
                    )
                ],
            ),
        )
        result = _pipeline(spec, retriever).run("velopark", qtype="multi_hop")
        tools = [call[0] for call in retriever.calls]
        self.assertEqual(tools[0], "retrieve")
        self.assertIn("event_neighborhood", tools)
        self.assertIn("supporting_chunks", tools)
        self.assertEqual(result.trace.stop_reason, "answered")
        self.assertEqual(result.answer.answer_text, "GBR")
        self.assertGreaterEqual(result.trace.total_tool_calls, 3)
        self.assertTrue(result.state.has_neighborhood)
        reasons = " ".join(result.trace.strategy_changes)
        self.assertIn("relational gap", reasons)
        self.assertIn("repair", result.trace.parallel_groups)
        if "chunks" in retriever.thread_ids and "neighborhood" in retriever.thread_ids:
            self.assertNotEqual(retriever.thread_ids["chunks"], retriever.thread_ids["neighborhood"])

    def test_venue_collision_stays_ambiguous(self) -> None:
        spec = _spec(
            "multi_hop",
            "events_at_venue_date",
            venue="Laura Biathlon & Ski Complex",
            date_text="22 February 2014",
            year=2014,
            requested_field="gold",
        )
        retriever = FakeRetriever(
            retrieve_spec=_result(
                "events_at_venue_date",
                "ambiguous",
                event_ids=["Qrelay", "Qski"],
                entities=[_entity("Qrelay", "Relay"), _entity("Qski", "Ski")],
                reason="multiple_events_same_venue_date",
            ),
            supporting_chunks=_result(
                "chunks_for_events",
                "supported",
                event_ids=["Qrelay", "Qski"],
                chunks=[
                    GraphEvidence(
                        evidence_id="c-relay",
                        evidence_type="chunk",
                        retrieval_method="gsql:chunks_for_events",
                        tool_call_id="c",
                        chunk_id="c-relay",
                        event_id="Qrelay",
                        text="relay",
                    )
                ],
            ),
        )
        result = _pipeline(spec, retriever).run("laura", qtype="multi_hop")
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.answer.answer_text, "")
        self.assertEqual(result.answer.citations, [])
        self.assertEqual(result.trace.stop_reason, "ambiguous")
        self.assertGreaterEqual(len(result.answer.retrieval_result.event_ids), 2)
        self.assertFalse(any(call[0] == "event_neighborhood" for call in retriever.calls))


class MissingAndBoundsTests(unittest.TestCase):
    def test_missing_entity_is_not_found(self) -> None:
        spec = _spec("lookup", "lookup_nations", event_title="missing event", requested_field="nations")
        retriever = FakeRetriever(
            retrieve_spec=_result("lookup_event", "not_found", reason="event_title_not_found")
        )
        result = _pipeline(spec, retriever).run("missing", qtype="lookup")
        self.assertEqual(result.status, "not_found")
        self.assertEqual(result.answer.answer_text, "")
        self.assertEqual(result.trace.stop_reason, "not_found")
        self.assertEqual(result.trace.total_tool_calls, 1)

    def test_budget_caps_tool_calls(self) -> None:
        spec = _spec(
            "multi_hop",
            "events_at_venue_date",
            venue="X",
            date_text="1 January 2012",
            year=2012,
        )
        retriever = FakeRetriever(
            retrieve_spec=_result(
                "events_at_venue_date",
                "supported",
                event_ids=["Q1"],
                facts=[_fact("Q1", "gold_raw", "Ada", operation="events_at_venue_date")],
            )
        )
        result = _pipeline(spec, retriever, budget=AgentBudget(max_tool_calls=1, max_follow_ups=2)).run("q")
        self.assertEqual(result.trace.total_tool_calls, 1)
        self.assertEqual(result.trace.stop_reason, "budget_exhausted")
        self.assertFalse(any(call[0] == "event_neighborhood" for call in retriever.calls))

    def test_repeated_action_stops_with_no_progress(self) -> None:
        spec = _spec("lookup", "lookup_nations", event_title="Judo", requested_field="nations")
        retriever = FakeRetriever(
            retrieve_spec=_result(
                "lookup_event",
                "supported",
                event_ids=["Q1"],
                entities=[_entity("Q1", "Judo")],
            )
        )
        result = _pipeline(spec, retriever, planner=RepeatPlanner()).run("q")
        self.assertEqual(result.trace.stop_reason, "no_progress")
        self.assertEqual(result.trace.total_tool_calls, 1)

    def test_unsupported_parse_does_not_call_tools(self) -> None:
        spec = QuerySpec(qtype="unknown", operation="unknown", raw_question="hello", matched_template=False)
        retriever = FakeRetriever(retrieve_spec=_result("unknown", "unresolved"))
        result = _pipeline(spec, retriever).run("hello")
        self.assertEqual(result.status, "unsupported")
        self.assertEqual(retriever.calls, [])
        self.assertEqual(result.trace.stop_reason, "unsupported")


class ProvenanceAndCitationTests(unittest.TestCase):
    def test_provenance_survives_and_orphan_citations_fail_closed(self) -> None:
        spec = _spec("lookup", "lookup_nations", event_title="Judo", requested_field="nations")
        retriever = FakeRetriever(
            retrieve_spec=_result(
                "lookup_event",
                "supported",
                event_ids=["Q26217865"],
                entities=[_entity("Q26217865", "Judo")],
                facts=[_fact("Q26217865", "nations", "23")],
            )
        )
        honest = _pipeline(spec, retriever).run("q")
        self.assertEqual(honest.answer.evidence_used[0].field_name, "nations")
        self.assertEqual(honest.answer.evidence_used[0].graph_refs[0].vertex_id, "Q26217865")
        payload = honest.trace.to_dict()
        json.dumps(payload)
        self.assertEqual(payload["stop_reason"], "answered")
        self.assertEqual(payload["total_tool_calls"], 1)

        retriever2 = FakeRetriever(
            retrieve_spec=_result(
                "lookup_event",
                "supported",
                event_ids=["Q26217865"],
                facts=[_fact("Q26217865", "nations", "23")],
            )
        )
        closed = _pipeline(spec, retriever2, generator=OrphanGenerator()).run("q")
        self.assertEqual(closed.status, "invalid_citations")
        self.assertEqual(closed.answer.answer_text, "")
        self.assertEqual(closed.answer.citations, [])
        self.assertIn("orphan_citation:not-retrieved", closed.answer.warnings)


class HarnessAndBaselineTests(unittest.TestCase):
    def test_three_way_harness_accepts_real_agent_adapter(self) -> None:
        spec = _spec("lookup", "lookup_nations", event_title="Judo", requested_field="nations")
        retriever = FakeRetriever(
            retrieve_spec=_result(
                "lookup_event",
                "supported",
                event_ids=["Q1"],
                facts=[_fact("Q1", "nations", "23")],
            )
        )
        agent = AgenticGraphRAGAdapter(_pipeline(spec, retriever))
        results = ThreeWayEvaluationHarness([agent]).run_question("q", qtype="lookup")
        self.assertEqual(results[0].system_name, "agentic_graphrag")
        self.assertEqual(results[0].status, "answered")
        self.assertIn("trace", results[0].retrieval_metadata)
        self.assertEqual(results[0].retrieval_metadata["stop_reason"], "answered")

    def test_fixed_graphrag_source_stays_non_agentic(self) -> None:
        source = inspect.getsource(FixedGraphRAGPipeline)
        self.assertIn("exactly one GraphRetriever call", source)
        self.assertNotIn("event_neighborhood", source)
        self.assertNotIn("BoundedPlanner", source)
        self.assertEqual(source.count("self.retriever.retrieve("), 1)


if __name__ == "__main__":
    unittest.main()
