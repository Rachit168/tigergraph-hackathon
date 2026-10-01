"""Corpus integration tests for the offline structured solver."""

from __future__ import annotations

import unittest

from evaluation.normalize import answers_match_normalized
from ingestion.loader import load_questions, parse_corpus
from ingestion.paths import DEFAULT_CORPUS_PATH, DEFAULT_PUBLIC_QUESTIONS_PATH
from retrieval.structured.index import StructuredIndex
from retrieval.structured.solver import StructuredSolver

CORPUS_AVAILABLE = DEFAULT_CORPUS_PATH.exists()
PUBLIC_AVAILABLE = DEFAULT_PUBLIC_QUESTIONS_PATH.exists()


@unittest.skipUnless(CORPUS_AVAILABLE, "hackathon corpus is not present")
class CorpusSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = parse_corpus(DEFAULT_CORPUS_PATH)
        cls.index = StructuredIndex(cls.corpus)
        cls.solver = StructuredSolver(cls.index)

    def test_index_uses_events_only(self) -> None:
        self.assertEqual(len(self.index.events), 2187)
        pursuit = self.index.document("Q3046361")
        assert pursuit is not None
        self.assertFalse(pursuit.is_event)

    def test_cycling_2000_aggregation_ignores_body_only_titles(self) -> None:
        title_docs = [
            document
            for document in self.corpus.documents
            if document.title_parse.parsed
            and document.title_parse.sport
            and document.title_parse.sport.casefold() == "cycling"
            and document.title_parse.year == 2000
        ]
        self.assertGreater(len(title_docs), len(self.index.events_of_sport_games("cycling", 2000, "Summer")))
        result = self.solver.solve(
            "According to the provided corpus, how many cycling events at the 2000 Summer Olympics had more than 30 competitors?"
        )
        pool = self.index.events_of_sport_games("cycling", 2000, "Summer")
        expected = str(sum(1 for event in pool if event.competitors is not None and event.competitors > 30))
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.answer, [expected])
        self.assertNotIn("Q3046361", [event.event_id for event in result.events])

    def test_lookup_real_event(self) -> None:
        title = "Judo at the 2016 Summer Olympics – Women's 57 kg"
        result = self.solver.solve(f"How many nations competed in {title}?")
        event = self.index.lookup_title(title)[0]
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.answer, [str(event.nations)])


@unittest.skipUnless(
    CORPUS_AVAILABLE and PUBLIC_AVAILABLE,
    "corpus or public questions missing",
)
class PublicQuestionSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.questions = load_questions(DEFAULT_PUBLIC_QUESTIONS_PATH)
        cls.solver = StructuredSolver(StructuredIndex(parse_corpus(DEFAULT_CORPUS_PATH)))

    def test_public_questions_are_the_only_default_eval_set(self) -> None:
        self.assertTrue(DEFAULT_PUBLIC_QUESTIONS_PATH.exists())
        self.assertTrue(all(question.get("qid", "").startswith("pub-") for question in self.questions))
        self.assertGreaterEqual(len(self.questions), 1)

    def test_lookup_family_is_complete(self) -> None:
        lookups = [question for question in self.questions if question["qtype"] == "lookup"]
        self.assertEqual(len(lookups), 19)
        for question in lookups:
            result = self.solver.solve(question["question"], qtype="lookup")
            self.assertTrue(
                answers_match_normalized(result.answer, question["answer"]),
                msg=question["qid"],
            )
