"""Live GraphRAG retrieval against the populated OlympicGraph. Skipped if unconfigured."""

from __future__ import annotations

import unittest

from config.settings import load_settings
from retrieval.graph.client import TigerGraphClient
from retrieval.graphrag.retriever import GraphRetriever, ensure_retrieval_queries
from retrieval.structured.question import QuestionParser


@unittest.skipUnless(load_settings().configured, "TigerGraph env not configured")
class LiveGraphRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TigerGraphClient()
        if not cls.client.connect():
            raise unittest.SkipTest(cls.client.error or "TigerGraph is not reachable")
        queries = cls.client.verify_queries()
        if not queries.get("ok"):
            raise unittest.SkipTest(f"solver queries missing: {queries.get('missing')}")
        report = ensure_retrieval_queries(cls.client)
        if not report.get("ok"):
            raise unittest.SkipTest(f"retrieval queries missing: {report.get('missing')}")
        cls.retriever = GraphRetriever.from_client(cls.client)
        cls.parser = QuestionParser()

    def test_lookup_does_not_invent_missing_event(self) -> None:
        result = self.retriever.lookup_event("This title is not an Olympic event 99999")
        self.assertEqual(result.status, "not_found")
        self.assertEqual(result.cardinality, "empty")
        self.assertEqual(result.facts, [])
        self.assertTrue(result.notes.get("answer_suppressed"))

    def test_laura_biathlon_collision_is_ambiguous(self) -> None:
        spec = self.parser.parse(
            "Who won the gold medal in the event held at Laura Biathlon & Ski Complex on 22 February 2014?",
            qtype="multi_hop",
        )
        result = self.retriever.retrieve(spec)
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.cardinality, "ambiguous")
        self.assertGreaterEqual(len(result.event_ids), 2)
        self.assertEqual(result.facts, [])
        self.assertTrue(result.notes.get("answer_suppressed"))
        self.assertGreaterEqual(result.notes.get("after_date_tighten", len(result.event_ids)), 2)

    def test_velopark_tighter_date_is_unique(self) -> None:
        spec = self.parser.parse(
            "Who won the gold medal in the event held at London Velopark on 3 to 4 August at the 2012 Summer Olympics?",
            qtype="multi_hop",
        )
        result = self.retriever.retrieve(spec)
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.cardinality, "one")
        self.assertEqual(len(result.event_ids), 1)
        self.assertGreaterEqual(int(result.notes.get("date_overlap_hits") or 1), 2)
        self.assertEqual(result.notes.get("after_date_tighten"), 1)
        self.assertIn("Q2297633", result.event_ids)

    def test_aggregation_complete_set(self) -> None:
        spec = self.parser.parse(
            "According to the provided corpus, how many cycling events at the 2000 Summer Olympics had more than 30 competitors?",
            qtype="aggregation",
        )
        result = self.retriever.retrieve(spec)
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.cardinality, "complete_set")
        self.assertFalse(result.notes.get("truncated_set"))
        self.assertEqual(len(result.event_ids), int(result.facts[0].value))
        self.assertGreaterEqual(len(result.event_ids), 1)

    def test_neighborhood_and_chunks_have_provenance(self) -> None:
        lookup = self.retriever.lookup_event("Judo at the 2016 Summer Olympics – Women's 57 kg", include_chunks=True)
        if lookup.status != "supported" or not lookup.event_ids:
            self.skipTest("lookup event not present on live graph")
        event_id = lookup.event_ids[0]
        self.assertTrue(lookup.has_supporting_chunks)
        self.assertEqual(lookup.chunks[0].evidence_type, "chunk")
        self.assertEqual(lookup.chunks[0].source_chunk_id, lookup.chunks[0].chunk_id)
        neighborhood = self.retriever.event_neighborhood(event_id)
        self.assertEqual(neighborhood.status, "supported")
        self.assertTrue(any(edge.field_name == "OF_SPORT" for edge in neighborhood.edges))
        self.assertTrue(neighborhood.has_supporting_chunks)


if __name__ == "__main__":
    unittest.main()
