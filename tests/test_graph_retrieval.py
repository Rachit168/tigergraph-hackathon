"""Phase 5 GraphRAG retrieval contracts. No live TigerGraph required."""

from __future__ import annotations

import unittest

from ingestion.graph_records import build_graph_export
from ingestion.parse import parse_document
from retrieval.graph.contract import GraphStore
from retrieval.graph.params import params_for_spec
from retrieval.graph.results import GraphQueryResult
from retrieval.graphrag.normalize import (
    attach_chunks,
    cardinality_for,
    chunk_rows_from_payload,
    evidence_from_chunks,
    evidence_from_neighborhood_payload,
    normalize_graph_result,
)
from retrieval.graphrag.retriever import GraphRetriever, GraphStoreBackend
from retrieval.graphrag.validate import (
    chunk_query_params,
    packed_event_ids,
    spec_query_binding,
    validate_event_id,
    validate_spec,
)
from retrieval.structured.models import QuerySpec


def _event_doc(doc_id: str, title: str, **fields) -> dict:
    lines = ["[Infobox Olympic event]", f"  event: {title}", "  games: 2016 Summer"]
    venue = fields.get("venue", "Test Hall")
    date = fields.get("date", "12 August 2016")
    competitors = fields.get("competitors", "10")
    nations = fields.get("nations", "8")
    gold = fields.get("gold", "Ada Lovelace")
    prev = fields.get("prev", "2012")
    games = fields.get("games")
    if games:
        lines[2] = f"  games: {games}"
    if venue:
        lines.append(f"  venue: {venue}")
    if date:
        lines.append(f"  date: {date}")
    if competitors is not None:
        lines.append(f"  competitors: {competitors}")
    if nations is not None:
        lines.append(f"  nations: {nations}")
    lines.append(f"  gold: {gold}")
    if prev:
        lines.append(f"  prev: {prev}")
    lines.append("")
    return {"doc_id": doc_id, "title": title, "text": "\n".join(lines)}


def _store(*records: dict) -> GraphStore:
    documents = [parse_document(record) for record in records]
    return GraphStore.from_export(build_graph_export(documents))


class FakeBackend:
    def __init__(self, graph: GraphQueryResult | None = None, query_payloads: dict | None = None) -> None:
        self.graph = graph
        self.query_payloads = query_payloads or {}
        self.calls: list[tuple[str, dict | None]] = []

    def execute_spec(self, spec: QuerySpec, events_by_id: dict | None = None) -> GraphQueryResult:
        self.calls.append(("execute_spec", spec.to_dict()))
        if self.graph is None:
            raise AssertionError("no graph result configured")
        return self.graph

    def run_query(self, name: str, params: dict | None = None) -> dict:
        self.calls.append((name, params))
        if name not in self.query_payloads:
            raise RuntimeError(f"missing payload for {name}")
        return self.query_payloads[name]


class ValidateTests(unittest.TestCase):
    def test_lookup_params_and_missing_title(self) -> None:
        spec = QuerySpec(
            qtype="lookup",
            operation="lookup_nations",
            raw_question="x",
            matched_template=True,
            event_title="Judo at the 2016 Summer Olympics – Women's 57 kg",
        )
        name, params = spec_query_binding(spec)
        self.assertEqual(name, "lookup_event")
        self.assertEqual(params["title"], spec.event_title)
        self.assertTrue(params["title_folded"])
        bad = QuerySpec(qtype="lookup", operation="lookup_nations", raw_question="", matched_template=True)
        self.assertEqual(validate_spec(bad), "missing_event_title")

    def test_aggregation_and_venue_args(self) -> None:
        spec = QuerySpec(
            qtype="aggregation",
            operation="count_over_threshold",
            raw_question="x",
            matched_template=True,
            sport="cycling",
            year=2000,
            season="Summer",
            threshold=30,
        )
        name, params = spec_query_binding(spec)
        self.assertEqual(name, "count_over_threshold")
        self.assertEqual(params["threshold"], 30)
        self.assertEqual(params["year"], 2000)
        venue = QuerySpec(
            qtype="multi_hop",
            operation="events_at_venue_date",
            raw_question="x",
            matched_template=True,
            venue="Laura Biathlon & Ski Complex",
            date_text="22 February 2014",
            year=2014,
        )
        name, params = spec_query_binding(venue)
        self.assertEqual(name, "events_at_venue_date")
        self.assertEqual(params["year"], 2014)
        self.assertTrue(params["venue_compact"])
        self.assertEqual(validate_spec(QuerySpec(qtype="multi_hop", operation="events_at_venue_date", raw_question="", matched_template=True)), "missing_venue_date_args")

    def test_packed_event_ids_and_chunk_params(self) -> None:
        self.assertEqual(packed_event_ids([]), "")
        packed = packed_event_ids(["Q1", "Q12"])
        self.assertEqual(packed, "|Q1|Q12|")
        self.assertIn("|Q1|", packed)
        self.assertIn("|Q12|", packed)
        self.assertNotIn("|Q1|", "|Q12|")
        params = chunk_query_params(["Q1", "Q2"], max_extra=3)
        self.assertEqual(params["max_extra"], 3)
        with self.assertRaises(ValueError):
            chunk_query_params(["Q1"], max_extra=-1)
        self.assertEqual(validate_event_id(""), "missing_event_id")
        self.assertIsNone(validate_event_id("Q1"))

    def test_unparsed_question_is_unresolved(self) -> None:
        spec = QuerySpec(qtype="unknown", operation="unknown", raw_question="hello", matched_template=False)
        self.assertEqual(validate_spec(spec), "question_template_unparsed")


class NormalizeTests(unittest.TestCase):
    def test_empty_lookup_does_not_invent_facts(self) -> None:
        graph = GraphQueryResult(operation="lookup_event", status="not_found", reason="event_title_not_found")
        result = normalize_graph_result(
            graph,
            query_name="lookup_event",
            params={"title": "missing"},
            tool_call_id="t1",
        )
        self.assertEqual(result.cardinality, "empty")
        self.assertEqual(result.facts, [])
        self.assertEqual(result.entities, [])
        self.assertTrue(result.notes.get("answer_suppressed"))
        self.assertIsNone(result.facts[0].value if result.facts else None)

    def test_ambiguous_does_not_emit_gold_fact(self) -> None:
        graph = GraphQueryResult(
            operation="events_at_venue_date",
            status="ambiguous",
            reason="multiple_events_same_venue_date",
            event_ids=["Qa", "Qb"],
            titles=["A", "B"],
            gold=["Alice", "Bob"],
        )
        result = normalize_graph_result(
            graph,
            query_name="events_at_venue_date",
            params={},
            tool_call_id="t2",
        )
        self.assertEqual(result.cardinality, "ambiguous")
        self.assertEqual(len(result.entities), 2)
        self.assertEqual(result.facts, [])
        self.assertTrue(result.notes.get("answer_suppressed"))

    def test_aggregation_complete_set_is_not_truncated(self) -> None:
        ids = [f"Q{i}" for i in range(12)]
        graph = GraphQueryResult(
            operation="count_over_threshold",
            status="supported",
            event_ids=list(ids),
            count=12,
            notes={"pool_size": 12},
        )
        result = normalize_graph_result(
            graph,
            query_name="count_over_threshold",
            params={"threshold": 30},
            tool_call_id="t3",
        )
        self.assertEqual(result.cardinality, "complete_set")
        self.assertEqual(result.event_ids, ids)
        self.assertFalse(result.notes.get("truncated_set"))
        self.assertEqual(result.facts[0].field_name, "count")
        self.assertEqual(result.facts[0].value, "12")
        self.assertEqual(len(result.facts[0].graph_refs), 12)

    def test_supported_lookup_nations_fact_and_provenance(self) -> None:
        graph = GraphQueryResult(
            operation="lookup_event",
            status="supported",
            event_ids=["Qjudo"],
            titles=["Judo"],
            nations=[23],
        )
        result = normalize_graph_result(
            graph,
            query_name="lookup_event",
            params={"title": "Judo"},
            tool_call_id="call-1",
        )
        self.assertEqual(result.cardinality, "one")
        self.assertEqual(len(result.facts), 1)
        fact = result.facts[0]
        self.assertEqual(fact.evidence_type, "fact")
        self.assertEqual(fact.field_name, "nations")
        self.assertEqual(fact.value, "23")
        self.assertEqual(fact.retrieval_method, "gsql:lookup_event")
        self.assertEqual(fact.tool_call_id, "call-1")
        self.assertEqual(fact.graph_refs[0].vertex_type, "Event")
        self.assertEqual(fact.graph_refs[0].vertex_id, "Qjudo")
        self.assertEqual(fact.graph_refs[0].attribute, "nations")
        self.assertIsNone(fact.text)
        self.assertIsNone(fact.source_chunk_id)
        self.assertTrue(fact.why_retrieved)

    def test_chunk_provenance(self) -> None:
        payload = {
            "chunk_ids": ["c1", "c2"],
            "document_ids": ["d1", "d1"],
            "event_ids": ["Q1", "Q1"],
            "texts": ["infobox text", "lead text"],
            "kinds": ["infobox", "lead"],
            "urls": ["https://example.test/1", ""],
        }
        rows = chunk_rows_from_payload(payload)
        items = evidence_from_chunks(rows, tool_call_id="t4", retrieval_method="gsql:chunks_for_events")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].evidence_type, "chunk")
        self.assertEqual(items[0].source_chunk_id, "c1")
        self.assertEqual(items[0].document_id, "d1")
        self.assertEqual(items[0].text, "infobox text")
        self.assertEqual(items[0].source_url, "https://example.test/1")
        self.assertIsNone(items[1].source_url)
        self.assertTrue(any(ref.edge_type == "DESCRIBES" for ref in items[0].graph_refs))

    def test_neighborhood_edges(self) -> None:
        payload = {
            "event_ids": ["Q1"],
            "titles": ["Event"],
            "document_ids": ["D1"],
            "gold": ["Ada"],
            "nations": [4],
            "competitors": [10],
            "games_ids": ["2016-Summer"],
            "sport_ids": ["judo"],
            "venue_ids": ["test-hall"],
            "chunk_ids": ["c-infobox"],
            "chunk_document_ids": ["D1"],
            "chunk_event_ids": ["Q1"],
            "texts": ["quote"],
            "kinds": ["infobox"],
            "urls": [""],
        }
        entities, facts, edges, chunks = evidence_from_neighborhood_payload(payload, tool_call_id="n1")
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].document_id, "D1")
        self.assertGreaterEqual(len(facts), 2)
        self.assertEqual({edge.field_name for edge in edges}, {"IN_GAMES", "OF_SPORT", "HELD_AT"})
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "quote")

    def test_cardinality_helpers(self) -> None:
        self.assertEqual(cardinality_for(GraphQueryResult(operation="lookup_event", status="unresolved")), "unresolved")
        self.assertEqual(
            cardinality_for(GraphQueryResult(operation="count_over_threshold", status="supported", event_ids=[], count=0)),
            "complete_set",
        )


class RetrieverTests(unittest.TestCase):
    def test_store_lookup_and_absence(self) -> None:
        title = "Judo at the 2016 Summer Olympics – Women's 57 kg"
        store = _store(_event_doc("Qjudo", title, nations="23"))
        retriever = GraphRetriever.from_store(store)
        hit = retriever.lookup_event(title)
        self.assertEqual(hit.status, "supported")
        self.assertEqual(hit.cardinality, "one")
        self.assertEqual(hit.facts[0].value, "23")
        miss = retriever.lookup_event("Not an Olympic event title")
        self.assertEqual(miss.status, "not_found")
        self.assertEqual(miss.cardinality, "empty")
        self.assertEqual(miss.facts, [])
        self.assertTrue(miss.notes.get("answer_suppressed"))

    def test_aggregation_set_preserved_on_store(self) -> None:
        records = []
        for index in range(8):
            records.append(
                _event_doc(
                    f"Qc{index}",
                    f"Cycling at the 2000 Summer Olympics – Event {index}",
                    games="2000 Summer",
                    competitors="40",
                )
            )
        records.append(
            _event_doc(
                "QcLow",
                "Cycling at the 2000 Summer Olympics – Low",
                games="2000 Summer",
                competitors="10",
            )
        )
        retriever = GraphRetriever.from_store(_store(*records))
        result = retriever.count_over_threshold("cycling", 2000, "Summer", 30)
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.cardinality, "complete_set")
        self.assertEqual(len(result.event_ids), 8)
        self.assertEqual(result.facts[0].value, "8")
        self.assertFalse(result.notes.get("truncated_set"))

    def test_temporal_previous_gold(self) -> None:
        retriever = GraphRetriever.from_store(
            _store(
                _event_doc(
                    "Q2016",
                    "Athletics at the 2016 Summer Olympics – Men's pole vault",
                    gold="Thiago Braz",
                    prev="2012",
                ),
                {
                    "doc_id": "Q2012",
                    "title": "Athletics at the 2012 Summer Olympics – Men's pole vault",
                    "text": """[Infobox Olympic event]
  event: Men's pole vault
  games: 2012 Summer
  venue: Stadium
  date: 8 August 2012
  competitors: 20
  nations: 12
  gold: Renaud Lavillenie
  prev: 2008
""",
                },
            )
        )
        result = retriever.previous_event_gold("athletics", "men's pole vault", 2016, "Summer")
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.cardinality, "one")
        golds = [item.value for item in result.facts if item.field_name == "gold_raw"]
        self.assertEqual(golds, ["Renaud Lavillenie"])

    def test_laura_style_collision_stays_ambiguous(self) -> None:
        retriever = GraphRetriever.from_store(
            _store(
                _event_doc(
                    "Qrelay",
                    "Biathlon at the 2014 Winter Olympics – Men's relay",
                    venue="Laura Biathlon & Ski Complex",
                    date="22 February 2014",
                    games="2014 Winter",
                    gold="Erik Lesser",
                ),
                _event_doc(
                    "Qski",
                    "Cross-country skiing at the 2014 Winter Olympics – Women's 30 kilometre freestyle",
                    venue="Laura Biathlon & Ski Complex",
                    date="22 February 2014",
                    games="2014 Winter",
                    gold="Marit Bjørgen",
                ),
            )
        )
        result = retriever.events_at_venue_date("Laura Biathlon & Ski Complex", "22 February 2014", year=2014)
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.cardinality, "ambiguous")
        self.assertGreaterEqual(len(result.event_ids), 2)
        self.assertEqual(result.facts, [])
        self.assertTrue(result.notes.get("answer_suppressed"))

    def test_tighter_date_span_resolves_without_using_gold(self) -> None:
        retriever = GraphRetriever.from_store(
            _store(
                _event_doc(
                    "Qwide",
                    "Cycling at the 2012 Summer Olympics – Men's omnium",
                    venue="London Velopark",
                    date="4 to 5 August 2012",
                    games="2012 Summer",
                    gold="Lasse Norman Hansen",
                ),
                _event_doc(
                    "Qexact",
                    "Cycling at the 2012 Summer Olympics – Women's team pursuit",
                    venue="London Velopark",
                    date="3 to 4 August 2012",
                    games="2012 Summer",
                    gold="Dani King",
                ),
            )
        )
        result = retriever.events_at_venue_date(
            "London Velopark",
            "3 to 4 August at the 2012 Summer Olympics",
            year=2012,
        )
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.cardinality, "one")
        self.assertEqual(result.event_ids, ["Qexact"])
        golds = [item.value for item in result.facts if item.field_name == "gold_raw"]
        self.assertEqual(golds, ["Dani King"])
        self.assertEqual(result.notes.get("date_overlap_hits"), 2)
        self.assertEqual(result.notes.get("after_date_tighten"), 1)

    def test_include_chunks_attaches_without_changing_set(self) -> None:
        graph = GraphQueryResult(
            operation="count_over_threshold",
            status="supported",
            event_ids=["Q1", "Q2"],
            count=2,
        )
        backend = FakeBackend(
            graph=graph,
            query_payloads={
                "chunks_for_events": {
                    "chunk_ids": ["c1", "c2"],
                    "document_ids": ["d1", "d2"],
                    "event_ids": ["Q1", "Q2"],
                    "texts": ["one", "two"],
                    "kinds": ["infobox", "infobox"],
                }
            },
        )
        retriever = GraphRetriever(backend)
        spec = QuerySpec(
            qtype="aggregation",
            operation="count_over_threshold",
            raw_question="x",
            matched_template=True,
            sport="cycling",
            year=2000,
            season="Summer",
            threshold=30,
        )
        result = retriever.retrieve(spec, include_chunks=True)
        self.assertEqual(result.event_ids, ["Q1", "Q2"])
        self.assertEqual(len(result.chunks), 2)
        self.assertTrue(result.has_supporting_chunks)
        self.assertEqual(result.entities[0].source_chunk_id, "c1")

    def test_missing_chunk_query_does_not_become_an_answer(self) -> None:
        retriever = GraphRetriever.from_store(_store(_event_doc("Q1", "Judo at the 2016 Summer Olympics – Women's 57 kg")))
        chunks = retriever.supporting_chunks(["Q1"])
        self.assertEqual(chunks.status, "unresolved")
        self.assertEqual(chunks.reason, "chunk_query_unavailable")
        self.assertEqual(chunks.chunks, [])

    def test_neighborhood_empty_id(self) -> None:
        retriever = GraphRetriever(FakeBackend())
        result = retriever.event_neighborhood("")
        self.assertEqual(result.status, "unresolved")
        self.assertEqual(result.reason, "missing_event_id")
        self.assertEqual(result.facts, [])


class AttachChunksTests(unittest.TestCase):
    def test_attach_fills_document_ids(self) -> None:
        graph = GraphQueryResult(operation="lookup_event", status="supported", event_ids=["Q1"], nations=[2])
        result = normalize_graph_result(graph, query_name="lookup_event", params={}, tool_call_id="a")
        chunks = evidence_from_chunks(
            [{"chunk_id": "c1", "document_id": "D1", "event_id": "Q1", "text": "quote", "kind": "infobox"}],
            tool_call_id="a",
            retrieval_method="gsql:chunks_for_events",
        )
        attach_chunks(result, chunks)
        self.assertEqual(result.entities[0].document_id, "D1")
        self.assertEqual(result.facts[0].source_chunk_id, "c1")


class StoreBackendTests(unittest.TestCase):
    def test_store_backend_execute(self) -> None:
        store = _store(_event_doc("Qjudo", "Judo at the 2016 Summer Olympics – Women's 57 kg", nations="23"))
        backend = GraphStoreBackend(store)
        spec = QuerySpec(
            qtype="lookup",
            operation="lookup_nations",
            raw_question="x",
            matched_template=True,
            event_title="Judo at the 2016 Summer Olympics – Women's 57 kg",
        )
        graph = backend.execute_spec(spec)
        self.assertEqual(graph.status, "supported")
        name, params = params_for_spec(spec)
        self.assertEqual(name, "lookup_event")
        self.assertEqual(params["title"], spec.event_title)


if __name__ == "__main__":
    unittest.main()
