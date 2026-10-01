"""Deterministic RAG candidate generation, grouping, rerank, and compaction."""

from __future__ import annotations

import unittest

from ingestion.chunker import chunk_corpus
from ingestion.parse import parse_document
from retrieval.rag.query import plan_query
from retrieval.rag.retriever import TextRetriever
from retrieval.rag.select import evidence_text
from tests.test_text_retrieval import CANOE_TEXT, CANOE_TITLE, FILM_TEXT, JUDO_TEXT, JUDO_TITLE


def _event(doc_id: str, title: str, competitors: int, *, prev: str | None = None, gold: str = "Ada") -> object:
    prev_line = f"  prev: {prev}\n" if prev else ""
    text = (
        "[Infobox Olympic event]\n"
        f"  event: {title.split('–')[-1].strip() if '–' in title else title}\n"
        f"  games: {title}\n"
        "  venue: Test Venue\n"
        f"  competitors: {competitors}\n"
        "  nations: 12\n"
        f"  gold: {gold}\n"
        f"{prev_line}"
        f"The {title} competition had {competitors} competitors.\n"
    )
    return parse_document({"doc_id": doc_id, "title": title, "text": text})


class QueryPlanTests(unittest.TestCase):
    def test_set_count_extracts_sport_and_year(self) -> None:
        plan = plan_query(
            "According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?"
        )
        self.assertEqual(plan.intent, "set_count")
        self.assertEqual(plan.years, (2018,))
        self.assertEqual(plan.sport_tokens, ("biathlon",))
        self.assertTrue(any("biathlon 2018" in variant for variant in plan.variants))

    def test_set_extreme_extracts_multiword_sport(self) -> None:
        plan = plan_query(
            "which alpine skiing event at the 1988 Winter Olympics had the highest number of competitors?"
        )
        self.assertEqual(plan.intent, "set_extreme")
        self.assertEqual(plan.sport_tokens, ("alpine", "skiing"))
        self.assertEqual(plan.years, (1988,))

    def test_lookup_is_not_treated_as_set_count(self) -> None:
        plan = plan_query("How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?")
        self.assertEqual(plan.intent, "lookup")

    def test_temporal_intent(self) -> None:
        plan = plan_query("Who won gold in the previous judo event before the 2016 Summer Olympics Women's 57 kg?")
        self.assertEqual(plan.intent, "temporal")


class CandidateSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        docs = [
            _event("Qski88a", "Alpine skiing at the 1988 Winter Olympics – Men's downhill", 50),
            _event("Qski88b", "Alpine skiing at the 1988 Winter Olympics – Men's giant slalom", 110),
            _event("Qski88c", "Alpine skiing at the 1988 Winter Olympics – Women's slalom", 40),
            _event("Qjudo16", JUDO_TITLE, 23, prev="2012", gold="Rafaela Silva"),
            _event("Qjudo12", "Judo at the 2012 Summer Olympics – Women's 57 kg", 22, prev="2008", gold="An"),
            parse_document({"doc_id": "Qfilm", "title": "Forrest Gump", "text": FILM_TEXT}),
            parse_document({"doc_id": "Qcanoe", "title": CANOE_TITLE, "text": CANOE_TEXT}),
        ]
        self.chunks = chunk_corpus(docs)
        self.retriever = TextRetriever(self.chunks)

    def test_lookup_still_prefers_named_event(self) -> None:
        result = self.retriever.retrieve(
            "How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?",
            top_k=5,
            method="sparse",
        )
        self.assertTrue(result.hits)
        self.assertEqual(result.hits[0].document_id, "Qjudo16")
        self.assertLessEqual(len(result.hits), 5)
        self.assertIn("Judo at the 2016 Summer Olympics – Women's 57 kg", result.hits[0].text)

    def test_set_extreme_covers_all_matching_infoboxes(self) -> None:
        result = self.retriever.retrieve(
            "which alpine skiing event at the 1988 Winter Olympics had the highest number of competitors?",
            top_k=10,
            method="sparse",
        )
        doc_ids = {hit.document_id for hit in result.hits}
        self.assertEqual(doc_ids, {"Qski88a", "Qski88b", "Qski88c"})
        self.assertTrue(all(hit.kind == "infobox" for hit in result.hits))
        self.assertNotIn("Qfilm", doc_ids)
        self.assertIn("Men's giant slalom", " ".join(hit.text for hit in result.hits))
        self.assertIn("competitors: 110", " ".join(hit.text for hit in result.hits))

    def test_set_count_returns_more_than_top_k_when_needed(self) -> None:
        result = self.retriever.retrieve(
            "how many alpine skiing events at the 1988 Winter Olympics had more than 45 competitors?",
            top_k=2,
            method="sparse",
        )
        self.assertGreaterEqual(len(result.hits), 3)
        self.assertEqual(result.params["intent"], "set_count")

    def test_grouping_prefers_infobox_over_body(self) -> None:
        result = self.retriever.retrieve(
            "How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?",
            top_k=3,
            method="sparse",
        )
        judo = [hit for hit in result.hits if hit.document_id == "Qjudo16"]
        self.assertTrue(judo)
        self.assertEqual(judo[0].kind, "infobox")

    def test_unrelated_film_is_not_in_set(self) -> None:
        result = self.retriever.retrieve(
            "how many canoeing events at the 2012 Summer Olympics had more than 20 competitors?",
            top_k=10,
            method="sparse",
        )
        self.assertEqual({hit.document_id for hit in result.hits}, {"Qcanoe"})

    def test_temporal_expands_previous_year_infobox(self) -> None:
        result = self.retriever.retrieve(
            "Who won gold in the previous judo event before Judo at the 2016 Summer Olympics – Women's 57 kg?",
            top_k=10,
            method="sparse",
        )
        ids = {hit.document_id for hit in result.hits}
        self.assertIn("Qjudo16", ids)
        self.assertIn("Qjudo12", ids)

    def test_evidence_text_prefixes_title(self) -> None:
        judo = next(chunk for chunk in self.chunks if chunk.document_id == "Qjudo16" and chunk.kind == "infobox")
        text = evidence_text(judo, compact=True)
        self.assertIn(JUDO_TITLE, text)
        self.assertIn("competitors: 23", text)
        self.assertNotIn("event: Women's 57 kg", text)

    def test_lookup_does_not_dump_entire_set(self) -> None:
        result = self.retriever.retrieve(
            "How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?",
            top_k=4,
            method="sparse",
        )
        self.assertLessEqual(len(result.hits), 4)

    def test_deterministic_sparse_candidates(self) -> None:
        question = "which alpine skiing event at the 1988 Winter Olympics had the highest number of competitors?"
        a = self.retriever.retrieve(question, top_k=10, method="sparse")
        b = self.retriever.retrieve(question, top_k=10, method="sparse")
        self.assertEqual([hit.chunk_id for hit in a.hits], [hit.chunk_id for hit in b.hits])
        self.assertEqual([hit.text for hit in a.hits], [hit.text for hit in b.hits])
