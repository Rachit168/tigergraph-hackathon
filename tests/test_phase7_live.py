"""Read-only live validation of Phase 7 Agentic GraphRAG. Never mutates the graph."""

from __future__ import annotations

import unittest

from answering.generator import DeterministicGroundedGenerator
from answering.packer import ContextPacker
from config.settings import load_settings
from retrieval.agentic.pipeline import AgenticGraphRAGPipeline
from retrieval.graph.client import TigerGraphClient
from retrieval.graphrag.parser import GraphRAGQuestionParser
from retrieval.graphrag.retriever import GraphRetriever
from retrieval.structured.question import QuestionParser


@unittest.skipUnless(load_settings().configured, "TigerGraph env not configured")
class LiveAgenticGraphRAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        client = TigerGraphClient()
        if not client.connect():
            raise unittest.SkipTest(client.error or "TigerGraph is not reachable")
        if not client.verify_queries().get("ok"):
            raise unittest.SkipTest("solver queries missing")
        if not client.verify_retrieval_queries().get("ok"):
            raise unittest.SkipTest("retrieval queries missing")
        cls.pipeline = AgenticGraphRAGPipeline(
            GraphRAGQuestionParser(
                QuestionParser(sports=["athletics", "biathlon", "cycling", "judo"])
            ),
            GraphRetriever.from_client(client),
            ContextPacker(),
            DeterministicGroundedGenerator(),
        )

    def test_judo_lookup_is_one_structured_call(self) -> None:
        result = self.pipeline.run(
            "How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?",
            qtype="lookup",
        )
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer.answer_text, "23")
        self.assertEqual(result.trace.total_tool_calls, 1)
        self.assertEqual(result.trace.tool_calls[0].tool, "retrieve_spec")
        self.assertNotIn("event_neighborhood", [item.tool for item in result.trace.tool_calls])

    def test_laura_collision_remains_ambiguous(self) -> None:
        result = self.pipeline.run(
            "Who won the gold medal in the event held at Laura Biathlon & Ski Complex on 22 February 2014?",
            qtype="multi_hop",
        )
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.answer.answer_text, "")
        self.assertGreaterEqual(len(result.answer.retrieval_result.event_ids), 2)
        self.assertEqual(result.trace.stop_reason, "ambiguous")

    def test_cycling_complete_set(self) -> None:
        result = self.pipeline.run(
            "According to the provided corpus, how many cycling events at the 2000 Summer Olympics had more than 30 competitors?",
            qtype="aggregation",
        )
        self.assertEqual(result.status, "answered")
        self.assertEqual(
            int(result.answer.answer_text),
            len(result.answer.retrieval_result.event_ids),
        )
        self.assertEqual(result.answer.retrieval_result.cardinality, "complete_set")
        self.assertFalse(result.answer.retrieval_result.notes.get("truncated_set"))

    def test_temporal_previous_gold_uses_one_structured_call(self) -> None:
        result = self.pipeline.run(
            "Who won the gold medal in the men's pole vault athletics event at the Summer Olympics held immediately before 2016?",
            qtype="temporal",
        )
        tools = [item.tool for item in result.trace.tool_calls]
        self.assertEqual(result.status, "answered")
        self.assertIn("Renaud Lavillenie", result.answer.answer_text)
        self.assertEqual(tools, ["retrieve_spec"])
        self.assertEqual(result.trace.stop_reason, "answered")
        self.assertEqual(result.answer.retrieval_result.operation, "previous_event_gold")

    def test_missing_entity_is_not_found(self) -> None:
        result = self.pipeline.run(
            "How many nations competed in This event does not exist at the 2099 Summer Olympics?",
            qtype="lookup",
        )
        self.assertEqual(result.status, "not_found")
        self.assertEqual(result.answer.answer_text, "")


if __name__ == "__main__":
    unittest.main()
