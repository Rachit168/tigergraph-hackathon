"""Deterministic graph export and provenance tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ingestion.chunker import chunk_corpus
from ingestion.graph_export import export_graph
from ingestion.graph_ids import chunk_id
from ingestion.graph_records import build_graph_export
from ingestion.parse import parse_document

CANOE_TITLE = "Canoeing at the 2012 Summer Olympics – Men's K-2 1000 metres"
CANOE_TEXT = """[Infobox Olympic event]
  event: Men's canoe sprint K-2 1,000 metres
  games: 2012 Summer
  venue: Eton Dorney
  date: 6 to 8 August
  competitors: 24
  nations: 12
  gold: Rudolf DombiRoland Kökény
  goldNOC: HUN
  prev: 2008
  next: 2016

The men's canoe sprint K-2 1,000 metres competition at the 2012 Olympic Games in London took place between 6 and 8 August at Eton Dorney.
"""

BODY_ONLY_TITLE = "Cycling at the 2000 Summer Olympics – Women's individual pursuit"
BODY_ONLY = """These are the official results of the Women's Individual Pursuit at the 2000 Summer Olympics in Sydney, Australia.
"""


def _docs() -> list:
    canoe = parse_document({"doc_id": "Q303623", "title": CANOE_TITLE, "url": "https://example.test/canoe", "text": CANOE_TEXT})
    body = parse_document({"doc_id": "Q3046361", "title": BODY_ONLY_TITLE, "text": BODY_ONLY})
    return [canoe, body]


class GraphExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = _docs()
        self.graph = build_graph_export(self.documents)

    def test_title_only_page_is_not_an_event(self) -> None:
        event_ids = {event.id for event in self.graph.events}
        document_ids = {document.id for document in self.graph.documents}
        self.assertIn("Q3046361", document_ids)
        self.assertNotIn("Q3046361", event_ids)
        self.assertTrue(any(chunk.document_id == "Q3046361" for chunk in self.graph.chunks))
        self.assertTrue(all(chunk.event_id == "" for chunk in self.graph.chunks if chunk.document_id == "Q3046361"))

    def test_event_to_document_provenance(self) -> None:
        self.assertEqual(len(self.graph.events), 1)
        event = self.graph.events[0]
        self.assertEqual(event.id, "Q303623")
        self.assertEqual(event.document_id, "Q303623")
        self.assertEqual(event.gold_raw, "Rudolf DombiRoland Kökény")
        self.assertTrue(any(edge.src == "Q303623" and edge.tgt == "Q303623" for edge in self.graph.describes))

    def test_chunk_to_document_provenance(self) -> None:
        document_ids = {document.id for document in self.graph.documents}
        for chunk in self.graph.chunks:
            self.assertIn(chunk.document_id, document_ids)
            self.assertEqual(chunk.id, chunk_id(chunk.document_id, chunk.chunk_index))
        self.assertEqual(len(self.graph.contains_chunk), len(self.graph.chunks))

    def test_no_duplicate_ids(self) -> None:
        for name, rows in self.graph.iter_vertex_tables():
            ids = [row.id for row in rows]
            self.assertEqual(len(ids), len(set(ids)), name)

    def test_concatenated_gold_is_not_split(self) -> None:
        self.assertEqual(self.graph.events[0].gold_raw, "Rudolf DombiRoland Kökény")
        self.assertFalse(self.graph.notes["athlete_vertices"])
        self.assertFalse(self.graph.notes["prev_event_edges"])

    def test_pavilions_are_not_collapsed(self) -> None:
        left = parse_document(
            {
                "doc_id": "Q1",
                "title": "Boxing at the 2016 Summer Olympics – Men's flyweight",
                "text": "[Infobox Olympic event]\n  event: Men's flyweight\n  games: 2016 Summer\n  venue: Riocentro – Pavilion 6\n  date: 10 August\n  competitors: 10\n  nations: 10\n  gold: A\n",
            }
        )
        right = parse_document(
            {
                "doc_id": "Q2",
                "title": "Gymnastics at the 2016 Summer Olympics – Men's artistic individual all-around",
                "text": "[Infobox Olympic event]\n  event: Men's all-around\n  games: 2016 Summer\n  venue: Riocentro – Pavilion 4\n  date: 10 August\n  competitors: 10\n  nations: 10\n  gold: B\n",
            }
        )
        graph = build_graph_export([left, right])
        self.assertEqual(len(graph.venues), 2)

    def test_export_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            one = export_graph(self.documents, Path(first), chunks=chunk_corpus(self.documents))
            two = export_graph(self.documents, Path(second), chunks=chunk_corpus(self.documents))
            self.assertEqual(one["manifest"]["checksums"], two["manifest"]["checksums"])
            self.assertEqual(one["manifest"]["counts"]["Event"], 1)
            self.assertEqual(one["manifest"]["counts"]["Document"], 2)


if __name__ == "__main__":
    unittest.main()
