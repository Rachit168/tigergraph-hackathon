"""Phase 6 fixed GraphRAG pipeline, packing, generation, and citations."""

from __future__ import annotations

import inspect
import unittest

from answering.generator import DeterministicGroundedGenerator
from answering.models import GeneratorRequest, GeneratorResult
from answering.packer import ContextPacker, validate_generator_citations
from retrieval.graphrag.models import GraphEvidence, GraphRef, GraphRetrievalResult
from retrieval.graphrag.parser import GraphRAGQuestionParser
from retrieval.graphrag.pipeline import FixedGraphRAGPipeline
from retrieval.structured.models import QuerySpec
from retrieval.structured.question import QuestionParser


def _spec(operation: str = "lookup_nations", *, matched: bool = True) -> QuerySpec:
    return QuerySpec(
        qtype="lookup" if operation == "lookup_nations" else "unknown",
        operation=operation,
        raw_question="question",
        matched_template=matched,
        event_title="Judo at the 2016 Summer Olympics – Women's 57 kg"
        if operation == "lookup_nations"
        else None,
    )


def _entity(event_id: str, title: str, call_id: str = "call-1") -> GraphEvidence:
    return GraphEvidence(
        evidence_id=f"{call_id}:entity:{event_id}",
        evidence_type="entity",
        retrieval_method="gsql:lookup_event",
        tool_call_id=call_id,
        graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id)],
        event_id=event_id,
        document_id=event_id,
        field_name="title",
        value=title,
        why_retrieved="lookup matched Event",
    )


def _fact(
    event_id: str | None,
    field_name: str,
    value: str,
    *,
    call_id: str = "call-1",
    source_chunk_id: str | None = None,
) -> GraphEvidence:
    suffix = event_id or "set"
    return GraphEvidence(
        evidence_id=f"{call_id}:fact:{field_name}:{suffix}",
        evidence_type="fact",
        retrieval_method="gsql:lookup_event",
        tool_call_id=call_id,
        graph_refs=[
            GraphRef(vertex_type="Event", vertex_id=event_id, attribute=field_name)
        ],
        event_id=event_id,
        document_id=event_id,
        source_chunk_id=source_chunk_id,
        field_name=field_name,
        value=value,
        source_url="https://example.test/event" if source_chunk_id else None,
        why_retrieved=f"query returned {field_name}",
    )


def _lookup_result() -> GraphRetrievalResult:
    event_id = "Q26217865"
    return GraphRetrievalResult(
        operation="lookup_event",
        query_name="lookup_event",
        status="supported",
        cardinality="one",
        tool_call_id="call-1",
        retrieval_method="gsql:lookup_event",
        params={"title": "Judo"},
        event_ids=[event_id],
        entities=[_entity(event_id, "Judo at the 2016 Summer Olympics – Women's 57 kg")],
        facts=[_fact(event_id, "nations", "23", source_chunk_id="Q26217865::c000")],
    )


class SpyStructuredParser:
    def __init__(self, result: QuerySpec) -> None:
        self.result = result
        self.calls: list[tuple[str, str | None]] = []

    def parse(self, question: str, qtype: str | None = None) -> QuerySpec:
        self.calls.append((question, qtype))
        return self.result


class SpyGraphRetriever:
    def __init__(self, result: GraphRetrievalResult) -> None:
        self.result = result
        self.calls: list[tuple[QuerySpec, bool, int]] = []

    def retrieve(
        self,
        spec: QuerySpec,
        *,
        include_chunks: bool = False,
        max_extra_chunks: int = 0,
    ) -> GraphRetrievalResult:
        self.calls.append((spec, include_chunks, max_extra_chunks))
        return self.result

    def event_neighborhood(self, event_id: str):
        raise AssertionError(f"fixed pipeline must not call event_neighborhood({event_id})")


class CapturingGenerator:
    def __init__(self, result: GeneratorResult) -> None:
        self.result = result
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return self.result


class ParserAdapterTests(unittest.TestCase):
    def test_adapter_delegates_once_and_preserves_same_spec(self) -> None:
        spec = _spec()
        delegate = SpyStructuredParser(spec)
        adapter = GraphRAGQuestionParser(delegate)
        parsed = adapter.parse("How many nations?", qtype="lookup")
        self.assertIs(parsed.spec, spec)
        self.assertEqual(delegate.calls, [("How many nations?", "lookup")])
        self.assertEqual(parsed.status, "parsed")
        self.assertIsNone(parsed.reason)
        self.assertGreaterEqual(parsed.elapsed_ms, 0.0)

    def test_adapter_converts_validation_without_extracting(self) -> None:
        missing_title = QuerySpec(
            qtype="lookup",
            operation="lookup_nations",
            raw_question="",
            matched_template=True,
        )
        parsed = GraphRAGQuestionParser(SpyStructuredParser(missing_title)).parse("x")
        self.assertEqual(parsed.status, "unresolved")
        self.assertEqual(parsed.reason, "missing_event_title")

        unknown = _spec("unknown", matched=False)
        parsed = GraphRAGQuestionParser(SpyStructuredParser(unknown)).parse("x")
        self.assertEqual(parsed.status, "unsupported")
        self.assertEqual(parsed.reason, "question_template_unparsed")

    def test_real_parser_handles_required_families_through_adapter(self) -> None:
        parser = GraphRAGQuestionParser(
            QuestionParser(sports=["judo", "cycling", "biathlon", "athletics"])
        )
        cases = [
            (
                "How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?",
                "lookup",
                "lookup_nations",
            ),
            (
                "According to the provided corpus, how many cycling events at the 2000 Summer Olympics had more than 30 competitors?",
                "aggregation",
                "count_over_threshold",
            ),
            (
                "According to the provided corpus, which cycling event at the 2000 Summer Olympics had the highest number of competitors?",
                "superlative",
                "argmax_competitors",
            ),
            (
                "Who won the gold medal in the men's pole vault athletics event at the Summer Olympics held immediately before 2016?",
                "temporal",
                "previous_event_gold",
            ),
            (
                "Who won the gold medal in the event held at Laura Biathlon & Ski Complex on 22 February 2014?",
                "multi_hop",
                "events_at_venue_date",
            ),
        ]
        for question, qtype, operation in cases:
            with self.subTest(qtype=qtype):
                parsed = parser.parse(question, qtype=qtype)
                self.assertEqual(parsed.status, "parsed")
                self.assertEqual(parsed.spec.operation, operation)

    def test_parser_adapter_contains_no_duplicate_parser_or_query_logic(self) -> None:
        import retrieval.graphrag.parser as adapter_module

        source = inspect.getsource(adapter_module)
        for forbidden in (
            "re.compile",
            "LOOKUP_RE",
            "AGG_RE",
            "SUP_RE",
            "TEMP_RE",
            "MH_RE",
            "params_for_spec",
            "lookup_event",
            "count_over_threshold",
            "argmax_competitors",
            "previous_event_gold",
            "events_at_venue_date",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("self.parser.parse(", source)
        self.assertIn("validate_spec(spec)", source)


class ContextPackerTests(unittest.TestCase):
    def test_preserves_every_graph_provenance_field_and_null(self) -> None:
        evidence = GraphEvidence(
            evidence_id="ev-1",
            evidence_type="chunk",
            retrieval_method="gsql:chunks_for_events",
            tool_call_id="call-1",
            graph_refs=[
                GraphRef(
                    vertex_type="Chunk",
                    vertex_id="c1",
                    edge_type="CONTAINS_CHUNK",
                    attribute="text",
                )
            ],
            event_id="Q1",
            document_id="D1",
            chunk_id="c1",
            source_chunk_id="c1",
            field_name="infobox",
            value=None,
            text="supporting quote",
            retrieval_score=None,
            source_url="https://example.test",
            source_date=None,
            source_version=None,
            observation_truncated=False,
            why_retrieved="typed hop",
        )
        result = GraphRetrievalResult(
            operation="lookup_event",
            query_name="lookup_event",
            status="supported",
            cardinality="one",
            tool_call_id="call-1",
            retrieval_method="gsql:lookup_event",
            chunks=[evidence],
        )
        item = ContextPacker().pack_graph(result).items[0]
        self.assertEqual(item.evidence_id, evidence.evidence_id)
        self.assertEqual(item.graph_refs, tuple(evidence.graph_refs))
        self.assertEqual(item.event_id, "Q1")
        self.assertEqual(item.document_id, "D1")
        self.assertEqual(item.chunk_id, "c1")
        self.assertEqual(item.source_chunk_id, "c1")
        self.assertEqual(item.text, "supporting quote")
        self.assertIsNone(item.value)
        self.assertIsNone(item.retrieval_score)
        self.assertIsNone(item.source_date)
        self.assertIsNone(item.source_version)
        self.assertEqual(item.why_retrieved, "typed hop")

    def test_orders_and_deduplicates_by_evidence_id(self) -> None:
        result = _lookup_result()
        duplicate = _fact("Q26217865", "nations", "23", source_chunk_id="c1")
        result.facts.append(duplicate)
        result.chunks.append(
            GraphEvidence(
                evidence_id="call-1:chunk:c1",
                evidence_type="chunk",
                retrieval_method="gsql:chunks_for_events",
                tool_call_id="call-1",
                chunk_id="c1",
                source_chunk_id="c1",
                text="quote",
            )
        )
        context = ContextPacker().pack_graph(result)
        self.assertEqual(
            [item.evidence_type for item in context.items],
            ["entity", "fact", "chunk"],
        )
        self.assertEqual(len(context.evidence_ids), len(set(context.evidence_ids)))

    def test_complete_set_keeps_all_event_entities(self) -> None:
        event_ids = [f"Q{i}" for i in range(25)]
        result = GraphRetrievalResult(
            operation="count_over_threshold",
            query_name="count_over_threshold",
            status="supported",
            cardinality="complete_set",
            tool_call_id="count-call",
            retrieval_method="gsql:count_over_threshold",
            event_ids=event_ids,
            entities=[_entity(event_id, f"Event {event_id}", "count-call") for event_id in event_ids],
            facts=[_fact(None, "count", "25", call_id="count-call")],
            notes={"truncated_set": False},
        )
        context = ContextPacker().pack_graph(result)
        packed_ids = [
            item.event_id for item in context.items if item.evidence_type == "entity"
        ]
        self.assertTrue(context.complete_set)
        self.assertEqual(packed_ids, event_ids)
        self.assertEqual(context.notes["event_ids"], event_ids)


class CitationTests(unittest.TestCase):
    def test_valid_citation_resolves_to_actual_evidence(self) -> None:
        context = ContextPacker().pack_graph(_lookup_result())
        evidence_id = next(
            item.evidence_id for item in context.items if item.field_name == "nations"
        )
        validation = validate_generator_citations(
            context,
            GeneratorResult(answer_text="23", citation_ids=(evidence_id,)),
        )
        self.assertTrue(validation.valid)
        self.assertEqual(validation.citations[0].evidence_id, evidence_id)
        self.assertEqual(validation.citations[0].event_id, "Q26217865")
        self.assertEqual(validation.evidence_used[0].source_chunk_id, "Q26217865::c000")

    def test_orphan_citation_fails_closed(self) -> None:
        context = ContextPacker().pack_graph(_lookup_result())
        validation = validate_generator_citations(
            context,
            GeneratorResult(answer_text="23", citation_ids=("not-retrieved",)),
        )
        self.assertFalse(validation.valid)
        self.assertEqual(validation.citations, ())
        self.assertEqual(validation.evidence_used, ())
        self.assertIn("orphan_citation:not-retrieved", validation.errors)

    def test_fact_citation_exposes_its_retrieved_supporting_chunk(self) -> None:
        result = _lookup_result()
        result.facts[0].source_url = None
        result.chunks.append(
            GraphEvidence(
                evidence_id="call-1:chunk:Q26217865::c000",
                evidence_type="chunk",
                retrieval_method="gsql:chunks_for_events",
                tool_call_id="call-1",
                event_id="Q26217865",
                document_id="Q26217865",
                chunk_id="Q26217865::c000",
                source_chunk_id="Q26217865::c000",
                text="nations: 23",
                source_url="https://example.test/support",
            )
        )
        context = ContextPacker().pack_graph(result)
        fact_id = result.facts[0].evidence_id
        validation = validate_generator_citations(
            context,
            GeneratorResult(answer_text="23", citation_ids=(fact_id,)),
        )
        self.assertTrue(validation.valid)
        self.assertEqual(validation.citations[0].chunk_id, "Q26217865::c000")
        self.assertEqual(
            validation.citations[0].source_url,
            "https://example.test/support",
        )

    def test_answer_without_citation_fails_closed(self) -> None:
        context = ContextPacker().pack_graph(_lookup_result())
        validation = validate_generator_citations(
            context,
            GeneratorResult(answer_text="23"),
        )
        self.assertFalse(validation.valid)
        self.assertIn("missing_citations", validation.errors)


class GeneratorDedupTests(unittest.TestCase):
    def test_duplicate_gold_raw_values_render_once(self) -> None:
        retrieval = GraphRetrievalResult(
            operation="events_at_venue_date",
            query_name="events_at_venue_date",
            status="supported",
            cardinality="one",
            tool_call_id="merged",
            retrieval_method="agentic+gsql",
            event_ids=["Q25239316"],
            facts=[
                _fact("Q25239316", "gold_raw", "Naim Süleymanoğlu", call_id="retrieve"),
                _fact("Q25239316", "gold_raw", "Naim Süleymanoğlu", call_id="neighborhood"),
            ],
        )
        context = ContextPacker().pack_graph(retrieval)
        generated = DeterministicGroundedGenerator().generate(
            GeneratorRequest(question="who won?", parsed_request=None, context=context)
        )
        self.assertEqual(generated.answer_text, "Naim Süleymanoğlu")
        self.assertEqual(generated.citation_ids, ("retrieve:fact:gold_raw:Q25239316",))


class FixedPipelineTests(unittest.TestCase):
    def _pipeline(self, retrieval, generator=None):
        parser = GraphRAGQuestionParser(SpyStructuredParser(_spec()))
        retriever = SpyGraphRetriever(retrieval)
        pipeline = FixedGraphRAGPipeline(
            parser,
            retriever,
            ContextPacker(),
            generator or DeterministicGroundedGenerator(),
        )
        return pipeline, retriever

    def test_judo_answer_is_cited_and_retriever_called_exactly_once(self) -> None:
        pipeline, retriever = self._pipeline(_lookup_result())
        answer = pipeline.run("How many nations competed in Judo?", qtype="lookup")
        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.answer_text, "23")
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(
            answer.citations[0].evidence_id,
            answer.evidence_used[0].evidence_id,
        )
        self.assertIs(answer.retrieval_result, retriever.result)
        self.assertEqual(len(retriever.calls), 1)
        self.assertTrue(retriever.calls[0][1])
        self.assertEqual(retriever.calls[0][2], 0)
        self.assertEqual(answer.notes["graph_retriever_calls"], 1)
        self.assertEqual(answer.notes["followup_retrievals"], 0)
        self.assertGreaterEqual(answer.timings.parsing_ms, 0.0)
        self.assertGreaterEqual(answer.timings.retrieval_ms, 0.0)
        self.assertGreaterEqual(answer.timings.packing_ms, 0.0)
        self.assertGreaterEqual(answer.timings.generation_ms, 0.0)
        self.assertGreaterEqual(answer.timings.total_ms, 0.0)

    def test_laura_ambiguity_suppresses_malicious_generator(self) -> None:
        ambiguous = GraphRetrievalResult(
            operation="events_at_venue_date",
            query_name="events_at_venue_date",
            status="ambiguous",
            cardinality="ambiguous",
            reason="multiple_events_same_venue_date",
            tool_call_id="laura",
            retrieval_method="gsql:events_at_venue_date",
            event_ids=["Qrelay", "Qski"],
            entities=[
                _entity("Qrelay", "Biathlon relay", "laura"),
                _entity("Qski", "Skiing 30 km", "laura"),
            ],
            notes={"answer_suppressed": True},
        )
        malicious = CapturingGenerator(
            GeneratorResult(answer_text="Invented winner", citation_ids=("missing",))
        )
        pipeline, retriever = self._pipeline(ambiguous, malicious)
        answer = pipeline.run("Laura question", qtype="multi_hop")
        self.assertEqual(answer.status, "ambiguous")
        self.assertEqual(answer.answer_text, "")
        self.assertEqual(answer.citations, [])
        self.assertEqual(answer.evidence_used, [])
        self.assertEqual(answer.retrieval_result.event_ids, ["Qrelay", "Qski"])
        self.assertIn("answer_suppressed", answer.warnings)
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(len(malicious.requests), 1)

    def test_missing_entity_remains_not_found(self) -> None:
        missing = GraphRetrievalResult(
            operation="lookup_event",
            query_name="lookup_event",
            status="not_found",
            cardinality="empty",
            reason="event_title_not_found",
            tool_call_id="missing",
            retrieval_method="gsql:lookup_event",
            notes={"answer_suppressed": True},
        )
        pipeline, retriever = self._pipeline(missing)
        answer = pipeline.run("missing")
        self.assertEqual(answer.status, "not_found")
        self.assertEqual(answer.answer_text, "")
        self.assertEqual(answer.citations, [])
        self.assertEqual(len(retriever.calls), 1)

    def test_orphan_generator_citation_suppresses_answer(self) -> None:
        generator = CapturingGenerator(
            GeneratorResult(answer_text="23", citation_ids=("orphan",))
        )
        pipeline, _retriever = self._pipeline(_lookup_result(), generator)
        answer = pipeline.run("Judo")
        self.assertEqual(answer.status, "invalid_citations")
        self.assertEqual(answer.answer_text, "")
        self.assertEqual(answer.citations, [])
        self.assertEqual(answer.evidence_used, [])
        self.assertIn("orphan_citation:orphan", answer.warnings)

    def test_complete_set_is_not_truncated_by_pipeline(self) -> None:
        event_ids = [f"Q{i}" for i in range(16)]
        complete = GraphRetrievalResult(
            operation="count_over_threshold",
            query_name="count_over_threshold",
            status="supported",
            cardinality="complete_set",
            tool_call_id="count",
            retrieval_method="gsql:count_over_threshold",
            event_ids=event_ids,
            entities=[_entity(event_id, f"Event {event_id}", "count") for event_id in event_ids],
            facts=[_fact(None, "count", "16", call_id="count")],
            notes={"truncated_set": False},
        )
        generator = CapturingGenerator(
            GeneratorResult(
                answer_text="16",
                citation_ids=("count:fact:count:set",),
            )
        )
        pipeline, retriever = self._pipeline(complete, generator)
        answer = pipeline.run("cycling")
        packed_ids = [
            item.event_id
            for item in generator.requests[0].context.items
            if item.evidence_type == "entity"
        ]
        self.assertEqual(packed_ids, event_ids)
        self.assertEqual(answer.retrieval_result.event_ids, event_ids)
        self.assertTrue(answer.notes["complete_set_preserved"])
        self.assertEqual(len(retriever.calls), 1)

    def test_generation_failure_is_auditable(self) -> None:
        class BrokenGenerator:
            def generate(self, request):
                raise RuntimeError("provider unavailable")

        pipeline, retriever = self._pipeline(_lookup_result(), BrokenGenerator())
        answer = pipeline.run("Judo")
        self.assertEqual(answer.status, "generation_error")
        self.assertEqual(answer.answer_text, "")
        self.assertEqual(answer.notes["generator_error"], "RuntimeError")
        self.assertEqual(len(retriever.calls), 1)

    def test_unsupported_parse_still_uses_single_safe_retrieval_call(self) -> None:
        parser = GraphRAGQuestionParser(
            SpyStructuredParser(_spec("unknown", matched=False))
        )
        unresolved = GraphRetrievalResult(
            operation="unknown",
            query_name="unknown",
            status="unresolved",
            cardinality="unresolved",
            reason="question_template_unparsed",
            tool_call_id="unknown",
            retrieval_method="gsql:unknown",
            notes={"answer_suppressed": True},
        )
        retriever = SpyGraphRetriever(unresolved)
        pipeline = FixedGraphRAGPipeline(
            parser,
            retriever,
            ContextPacker(),
            DeterministicGroundedGenerator(),
        )
        answer = pipeline.run("Unsupported prose")
        self.assertEqual(answer.status, "unsupported")
        self.assertEqual(answer.answer_text, "")
        self.assertEqual(len(retriever.calls), 1)


if __name__ == "__main__":
    unittest.main()
