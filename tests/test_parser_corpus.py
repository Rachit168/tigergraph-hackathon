"""Corpus and public-question structural coverage tests."""

from __future__ import annotations

import unittest

from ingestion.diagnostics import corpus_profile, extract_question_slots, question_coverage
from ingestion.loader import load_questions, parse_corpus
from ingestion.paths import DEFAULT_CORPUS_PATH, DEFAULT_PUBLIC_QUESTIONS_PATH

CORPUS_AVAILABLE = DEFAULT_CORPUS_PATH.exists()


@unittest.skipUnless(CORPUS_AVAILABLE, "hackathon corpus is not present")
class CorpusIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = parse_corpus(DEFAULT_CORPUS_PATH)
        cls.profile = corpus_profile(cls.result)

    def test_document_count(self) -> None:
        self.assertEqual(self.profile["total_documents"], 2951)

    def test_event_count_matches_infobox_gate(self) -> None:
        self.assertEqual(
            self.profile["event_documents"],
            self.profile["documents_with_olympic_event_infobox"],
        )
        self.assertGreaterEqual(self.profile["event_documents"], 2100)
        self.assertLessEqual(self.profile["event_documents"], 2300)

    def test_body_only_olympic_titles_are_document_only(self) -> None:
        self.assertGreaterEqual(self.profile["rejected_despite_olympic_title"], 1)
        by_id = self.result.document_by_id()
        pursuit = by_id["Q3046361"]
        self.assertFalse(pursuit.is_event)
        self.assertEqual(pursuit.rejection_reason, "olympic_title_without_infobox")

    def test_known_infobox_document_is_event(self) -> None:
        event = self.result.event_by_id()["Q303623"]
        self.assertEqual(event.competitors, 24)
        self.assertEqual(event.nations, 12)
        self.assertEqual(event.gold_raw(), "Rudolf DombiRoland Kökény")

    def test_no_parser_crashes(self) -> None:
        self.assertEqual(self.profile["parse_failures"], 0)
        self.assertEqual(len(self.result.documents), 2951)


@unittest.skipUnless(
    CORPUS_AVAILABLE and DEFAULT_PUBLIC_QUESTIONS_PATH.exists(),
    "corpus or public questions missing",
)
class PublicQuestionCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = parse_corpus(DEFAULT_CORPUS_PATH)
        cls.questions = load_questions(DEFAULT_PUBLIC_QUESTIONS_PATH)
        cls.coverage = question_coverage(cls.result, cls.questions)

    def test_loads_100_questions(self) -> None:
        self.assertEqual(len(self.questions), 100)
        self.assertEqual(self.coverage["n"], 100)

    def test_lookup_titles_resolve(self) -> None:
        lookup = self.coverage["by_qtype"]["lookup"]
        self.assertEqual(lookup["n"], 19)
        self.assertEqual(lookup["unsupported"], 0)

    def test_templates_parse_for_all_families(self) -> None:
        for question in self.questions:
            slots = extract_question_slots(question["question"], question["qtype"])
            self.assertTrue(slots.get("matched_template"), msg=question["qid"])

    def test_overall_structural_support_is_high(self) -> None:
        self.assertGreaterEqual(self.coverage["supported_pct"], 80.0)

    def test_does_not_use_gold_answers(self) -> None:
        sample = self.coverage["rows"][0]
        self.assertNotIn("answer", sample)
        self.assertNotIn("gold_answer", sample)


class SlotExtractionTests(unittest.TestCase):
    def test_lookup_slot(self) -> None:
        slots = extract_question_slots(
            "How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?",
            "lookup",
        )
        self.assertEqual(
            slots["event_title"],
            "Judo at the 2016 Summer Olympics – Women's 57 kg",
        )

    def test_aggregation_slot(self) -> None:
        slots = extract_question_slots(
            "According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
            "aggregation",
        )
        self.assertEqual(slots["sport"], "biathlon")
        self.assertEqual(slots["year"], 2018)
        self.assertEqual(slots["season"], "Winter")
        self.assertEqual(slots["threshold"], 73)

    def test_temporal_slot(self) -> None:
        slots = extract_question_slots(
            "Who won the gold medal in the men's 20 kilometres walk athletics event at the Summer Olympics held immediately before 2016?",
            "temporal",
        )
        self.assertEqual(slots["named_year"], 2016)
        self.assertEqual(slots["previous_year"], 2012)
        self.assertEqual(slots["season"], "Summer")


if __name__ == "__main__":
    unittest.main()
