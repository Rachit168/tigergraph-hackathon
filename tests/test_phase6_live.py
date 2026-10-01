"""Read-only live validation of the fixed Phase 6 GraphRAG pipeline."""

from __future__ import annotations

import unittest

from answering.generator import DeterministicGroundedGenerator
from answering.packer import ContextPacker
from config.settings import load_settings
from retrieval.graph.client import TigerGraphClient
from retrieval.graphrag.parser import GraphRAGQuestionParser
from retrieval.graphrag.pipeline import FixedGraphRAGPipeline
from retrieval.graphrag.retriever import GraphRetriever
from retrieval.structured.question import QuestionParser


@unittest.skipUnless(load_settings().configured, "TigerGraph env not configured")
class LiveFixedGraphRAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        client = TigerGraphClient()
        if not client.connect():
            raise unittest.SkipTest(client.error or "TigerGraph is not reachable")
        solver_queries = client.verify_queries()
        retrieval_queries = client.verify_retrieval_queries()
        if not solver_queries.get("ok") or not retrieval_queries.get("ok"):
            raise unittest.SkipTest("required installed queries are unavailable")
        cls.pipeline = FixedGraphRAGPipeline(
            GraphRAGQuestionParser(
                QuestionParser(
                    sports=[
                        "athletics",
                        "biathlon",
                        "cross-country skiing",
                        "cycling",
                        "judo",
                    ]
                )
            ),
            GraphRetriever.from_client(client),
            ContextPacker(),
            DeterministicGroundedGenerator(),
        )

    def test_judo_lookup_returns_cited_answer(self) -> None:
        answer = self.pipeline.run(
            "How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?",
            qtype="lookup",
        )
        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.answer_text, "23")
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(answer.citations[0].event_id, "Q26217865")
        self.assertTrue(answer.citations[0].chunk_id)
        self.assertEqual(answer.notes["graph_retriever_calls"], 1)
        self.assertEqual(answer.notes["followup_retrievals"], 0)

    def test_laura_collision_remains_suppressed(self) -> None:
        answer = self.pipeline.run(
            "Who won the gold medal in the event held at Laura Biathlon & Ski Complex on 22 February 2014?",
            qtype="multi_hop",
        )
        self.assertEqual(answer.status, "ambiguous")
        self.assertEqual(answer.answer_text, "")
        self.assertEqual(answer.citations, [])
        self.assertGreaterEqual(len(answer.retrieval_result.event_ids), 2)

    def test_cycling_aggregation_remains_complete(self) -> None:
        answer = self.pipeline.run(
            "According to the provided corpus, how many cycling events at the 2000 Summer Olympics had more than 30 competitors?",
            qtype="aggregation",
        )
        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.retrieval_result.cardinality, "complete_set")
        self.assertEqual(
            int(answer.answer_text),
            len(answer.retrieval_result.event_ids),
        )
        self.assertFalse(answer.retrieval_result.notes.get("truncated_set"))
        self.assertTrue(answer.notes["complete_set_preserved"])

    def test_missing_entity_is_not_an_answer(self) -> None:
        answer = self.pipeline.run(
            "How many nations competed in This event does not exist at the 2099 Summer Olympics?",
            qtype="lookup",
        )
        self.assertEqual(answer.status, "not_found")
        self.assertEqual(answer.answer_text, "")
        self.assertEqual(answer.citations, [])


if __name__ == "__main__":
    unittest.main()
