"""Tests for infobox-first chunking."""

from __future__ import annotations

import unittest

from ingestion.chunker import ChunkingConfig, chunk_corpus, chunk_document
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

Rudolf Dombi and Roland Kökény from Hungary won the gold medal.

Results
First boat qualified for the final.

Heat 1
Hungary won the heat.
"""

BODY_ONLY = """These are the official results of the Women's Individual Pursuit at the 2000 Summer Olympics in Sydney, Australia.

Medalists
The medals were awarded on Sunday.
"""

FILM = "Forrest Gump is a 1994 American comedy-drama film.\n\nPlot\nThe film follows Forrest Gump."


class ChunkerTests(unittest.TestCase):
    def test_infobox_is_first_chunk(self) -> None:
        document = parse_document({"doc_id": "Q303623", "title": CANOE_TITLE, "url": "https://ex.test", "text": CANOE_TEXT})
        chunks = chunk_document(document)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].kind, "infobox")
        self.assertIn("nations: 12", chunks[0].text)
        self.assertEqual(chunks[0].chunk_id, "Q303623::c000")
        self.assertEqual(chunks[0].document_id, "Q303623")
        self.assertEqual(chunks[0].source_url, "https://ex.test")
        self.assertEqual(chunks[0].event_id, "Q303623")
        self.assertIn(CANOE_TITLE, chunks[0].indexed_text)
        self.assertGreaterEqual(chunks[0].start_char, 0)
        self.assertGreater(chunks[0].end_char, chunks[0].start_char)

    def test_lead_and_section_chunks(self) -> None:
        document = parse_document({"doc_id": "Q303623", "title": CANOE_TITLE, "text": CANOE_TEXT})
        chunks = chunk_document(document)
        kinds = [chunk.kind for chunk in chunks]
        self.assertIn("lead", kinds)
        self.assertIn("section", kinds)
        self.assertTrue(any("result" in chunk.section.lower() or chunk.section.lower().startswith("heat") for chunk in chunks))

    def test_body_only_olympic_title_is_document_only_but_chunked(self) -> None:
        document = parse_document(
            {
                "doc_id": "Q3046361",
                "title": "Cycling at the 2000 Summer Olympics – Women's individual pursuit",
                "text": BODY_ONLY,
            }
        )
        self.assertFalse(document.is_event)
        chunks = chunk_document(document)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].event_id)
        self.assertEqual(chunks[0].document_kind, "document")

    def test_distractor_is_chunked(self) -> None:
        document = parse_document({"doc_id": "Qfilm", "title": "Forrest Gump", "text": FILM})
        chunks = chunk_document(document)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].event_id)

    def test_no_duplicate_chunk_ids(self) -> None:
        docs = [
            parse_document({"doc_id": "Q303623", "title": CANOE_TITLE, "text": CANOE_TEXT}),
            parse_document({"doc_id": "Qfilm", "title": "Forrest Gump", "text": FILM}),
        ]
        chunks = chunk_corpus(docs)
        ids = [chunk.chunk_id for chunk in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_max_chars_splits_long_body(self) -> None:
        body = "Paragraph one is short.\n\n" + ("word " * 400)
        text = CANOE_TEXT.split("The men's")[0] + "The men's lead.\n\n" + body
        document = parse_document({"doc_id": "Qlong", "title": CANOE_TITLE, "text": text})
        chunks = chunk_document(document, ChunkingConfig(max_chars=200, hard_max_chars=400))
        self.assertTrue(any(len(chunk.text) <= 400 for chunk in chunks[1:]))
        self.assertGreater(len(chunks), 2)


if __name__ == "__main__":
    unittest.main()
