"""General parser robustness: closed synonym/shape classes, not qid rules.

Examples use official public templates or synthetic shapes. They do not load
private evaluation files.
"""

from __future__ import annotations

import unittest

from ingestion.loader import load_questions, parse_corpus
from ingestion.paths import DEFAULT_CORPUS_PATH, DEFAULT_PUBLIC_QUESTIONS_PATH
from retrieval.structured.question import QuestionParser, normalize_question_surface
from evaluation.retrieval_gold import build_parser

CORPUS_AVAILABLE = DEFAULT_CORPUS_PATH.exists()
PUBLIC_AVAILABLE = DEFAULT_PUBLIC_QUESTIONS_PATH.exists()

SPORTS = [
    "biathlon",
    "athletics",
    "sailing",
    "judo",
    "cycling",
    "rowing",
    "swimming",
    "boxing",
    "alpine skiing",
    "curling",
]


def _parser() -> QuestionParser:
    return QuestionParser(SPORTS)


class CaseAndSynonymTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = _parser()

    def test_lookup_is_case_insensitive(self) -> None:
        title = "Sailing at the 2016 Summer Olympics – Women's RS:X"
        spec = self.parser.parse(f"how many nations competed in {title}?")
        self.assertTrue(spec.matched_template)
        self.assertEqual(spec.operation, "lookup_nations")
        self.assertEqual(spec.event_title, title)

    def test_nations_countries_synonym(self) -> None:
        title = "Judo at the 2016 Summer Olympics – Women's 57 kg"
        spec = self.parser.parse(f"How many countries took part in {title}?")
        self.assertTrue(spec.matched_template)
        self.assertEqual(spec.event_title, title)

    def test_nation_count_framing(self) -> None:
        title = "Boxing at the 1988 Summer Olympics – Light flyweight"
        spec = self.parser.parse(f"What was the nation count for {title}?")
        self.assertTrue(spec.matched_template)
        self.assertEqual(spec.event_title, title)

    def test_more_than_over_synonym(self) -> None:
        spec = self.parser.parse(
            "According to the provided corpus, how many cycling events at the 2000 Summer Olympics had over 30 competitors?"
        )
        self.assertEqual(spec.operation, "count_over_threshold")
        self.assertEqual(spec.sport, "cycling")
        self.assertEqual(spec.year, 2000)
        self.assertEqual(spec.threshold, 30)

    def test_most_vs_highest_number(self) -> None:
        spec = self.parser.parse(
            "According to the provided corpus, which athletics event at the 2008 Summer Olympics had the highest number of competitors?"
        )
        self.assertEqual(spec.operation, "argmax_competitors")
        self.assertEqual(spec.sport, "athletics")
        self.assertEqual(spec.year, 2008)
        self.assertEqual(spec.season, "Summer")


class FramingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = _parser()

    def test_can_you_tell_me_prefix(self) -> None:
        title = "Judo at the 2016 Summer Olympics – Women's 57 kg"
        spec = self.parser.parse(f"Can you tell me how many nations competed in {title}?")
        self.assertTrue(spec.matched_template)
        self.assertEqual(spec.event_title, title)

    def test_according_to_corpus_is_harmless(self) -> None:
        spec = self.parser.parse(
            "According to the corpus, how many cycling events at the 2000 Summer Olympics had more than 30 competitors?"
        )
        self.assertTrue(spec.matched_template)
        self.assertEqual(spec.threshold, 30)

    def test_normalize_surface_strips_punctuation_not_title_text(self) -> None:
        surface = normalize_question_surface(
            "What's the nation count for Judo at the 2016 Summer Olympics – Women's 57 kg?"
        )
        self.assertTrue(surface.lower().startswith("what is the nation count for judo"))
        self.assertNotIn("?", surface)


class TemporalVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = _parser()

    def test_just_prior_to(self) -> None:
        spec = self.parser.parse(
            "Who took gold in the men's 20 kilometres walk athletics event at the Summer Olympics just prior to 2016?"
        )
        self.assertEqual(spec.operation, "previous_event_gold")
        self.assertEqual(spec.named_year, 2016)
        self.assertEqual(spec.season, "Summer")
        self.assertEqual(spec.sport, "athletics")
        self.assertEqual(spec.event_name, "men's 20 kilometres walk")

    def test_before_games_framing(self) -> None:
        spec = self.parser.parse(
            "Before the 2016 Summer Olympics, who won gold in the men's pole vault athletics event?"
        )
        self.assertEqual(spec.operation, "previous_event_gold")
        self.assertEqual(spec.named_year, 2016)
        self.assertEqual(spec.season, "Summer")
        self.assertEqual(spec.sport, "athletics")
        self.assertEqual(spec.event_name, "men's pole vault")

    def test_immediately_after_still_next(self) -> None:
        spec = self.parser.parse(
            "Who won the gold medal in the men's pole vault athletics event at the Summer Olympics held immediately after 2012?"
        )
        self.assertEqual(spec.operation, "next_event_gold")
        self.assertEqual(spec.temporal_relation, "next")
        self.assertEqual(spec.named_year, 2012)


class VenueDateVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = _parser()

    def test_event_held_at_on(self) -> None:
        spec = self.parser.parse(
            "Who won the gold medal in the event held at Olympic Weightlifting Gymnasium on 20 September 1988?"
        )
        self.assertEqual(spec.operation, "events_at_venue_date")
        self.assertEqual(spec.venue, "Olympic Weightlifting Gymnasium")
        self.assertEqual(spec.date_text, "20 September 1988")

    def test_earned_gold_at_on(self) -> None:
        spec = self.parser.parse(
            "Who earned gold at Royal Artillery Barracks on 28 July 2012?"
        )
        self.assertEqual(spec.venue, "Royal Artillery Barracks")
        self.assertEqual(spec.date_text, "28 July 2012")

    def test_from_date_range(self) -> None:
        spec = self.parser.parse(
            "Who was the gold medalist for the event held at London Velopark from 3 to 4 August 2012?"
        )
        self.assertEqual(spec.venue, "London Velopark")
        self.assertEqual(spec.date_text, "3 to 4 August 2012")

    def test_in_month_year(self) -> None:
        spec = self.parser.parse("Who won gold at Placeholder Arena in March 1850?")
        self.assertEqual(spec.venue, "Placeholder Arena")
        self.assertEqual(spec.date_text, "March 1850")


class AggregationVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = _parser()

    def test_count_with_more_than(self) -> None:
        spec = self.parser.parse(
            "Count the biathlon events at the 2018 Winter Olympics with more than 73 competitors."
        )
        self.assertEqual(spec.operation, "count_over_threshold")
        self.assertEqual(spec.sport, "biathlon")
        self.assertEqual(spec.year, 2018)
        self.assertEqual(spec.threshold, 73)

    def test_exceeded_threshold(self) -> None:
        spec = self.parser.parse(
            "How many curling events at the 1850 Summer Olympics exceeded 10 competitors?"
        )
        self.assertTrue(spec.matched_template)
        self.assertEqual(spec.year, 1850)
        self.assertEqual(spec.threshold, 10)

    def test_largest_field_superlative(self) -> None:
        spec = self.parser.parse(
            "Which 2008 Summer Olympics athletics event had the largest competitor field?"
        )
        self.assertEqual(spec.operation, "argmax_competitors")
        self.assertEqual(spec.sport, "athletics")
        self.assertEqual(spec.year, 2008)
        self.assertEqual(spec.season, "Summer")


class AmbiguityAndAbstentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = _parser()

    def test_ambiguous_venue_month_still_parses_as_multi_hop(self) -> None:
        spec = self.parser.parse("Who won gold at Placeholder Arena in March 1850?")
        self.assertEqual(spec.operation, "events_at_venue_date")
        self.assertEqual(self.parser.matching_families(spec.raw_question), ("multi_hop",))

    def test_tied_superlative_still_parses_as_superlative(self) -> None:
        spec = self.parser.parse(
            "According to the provided corpus, which athletics event at the 2008 Summer Olympics had the highest number of competitors?"
        )
        self.assertEqual(spec.operation, "argmax_competitors")
        self.assertEqual(self.parser.matching_families(spec.raw_question), ("superlative",))

    def test_missing_event_lookup_still_parses(self) -> None:
        spec = self.parser.parse(
            "What's the nation count for Curling at the 1850 Summer Olympics – Mixed team?"
        )
        self.assertTrue(spec.matched_template)
        self.assertEqual(spec.operation, "lookup_nations")

    def test_unrelated_question_fails_closed(self) -> None:
        spec = self.parser.parse("Who was the tallest athlete at the 2012 Summer Olympics?")
        self.assertFalse(spec.matched_template)
        self.assertEqual(spec.operation, "unknown")

    def test_lookup_does_not_collapse_into_venue_date(self) -> None:
        title = "Sailing at the 2016 Summer Olympics – Women's RS:X"
        question = f"How many nations competed in {title}?"
        self.assertEqual(self.parser.matching_families(question), ("lookup",))


class ParaphraseBehaviorTests(unittest.TestCase):
    """Closed-class paraphrases of public templates. No qid/production branches."""

    CASES = (
        (
            "What was the nation count for Sailing at the 2016 Summer Olympics – Women's RS:X?",
            "lookup_nations",
            {"event_title": "Sailing at the 2016 Summer Olympics – Women's RS:X"},
        ),
        (
            "How many countries took part in Judo at the 2016 Summer Olympics – Women's 57 kg?",
            "lookup_nations",
            {"event_title": "Judo at the 2016 Summer Olympics – Women's 57 kg"},
        ),
        (
            "How many nations competed in Boxing at the 1988 Summer Olympics – Light flyweight?",
            "lookup_nations",
            {"event_title": "Boxing at the 1988 Summer Olympics – Light flyweight"},
        ),
        (
            "how many nations competed in Sailing at the 2016 Summer Olympics – Women's RS:X?",
            "lookup_nations",
            {"event_title": "Sailing at the 2016 Summer Olympics – Women's RS:X"},
        ),
        (
            "Who took gold in the men's 20 kilometres walk athletics event at the Summer Olympics just prior to 2016?",
            "previous_event_gold",
            {"named_year": 2016, "sport": "athletics"},
        ),
        (
            "Before the 2016 Summer Olympics, who won gold in the men's pole vault athletics event?",
            "previous_event_gold",
            {"named_year": 2016, "sport": "athletics"},
        ),
        (
            "Who earned gold at Royal Artillery Barracks on 28 July 2012?",
            "events_at_venue_date",
            {"venue": "Royal Artillery Barracks"},
        ),
        (
            "Which athlete won gold at Olympic Weightlifting Gymnasium on 20 September 1988?",
            "events_at_venue_date",
            {"venue": "Olympic Weightlifting Gymnasium"},
        ),
        (
            "Who was the gold medalist for the event held at London Velopark from 3 to 4 August 2012?",
            "events_at_venue_date",
            {"venue": "London Velopark"},
        ),
        (
            "According to the provided corpus, which athletics event at the 2008 Summer Olympics had the highest number of competitors?",
            "argmax_competitors",
            {"sport": "athletics", "year": 2008},
        ),
        (
            "Which 2008 Summer Olympics athletics event had the largest competitor field?",
            "argmax_competitors",
            {"sport": "athletics", "year": 2008},
        ),
        (
            "How many cycling events at the 2000 Summer Olympics had more than 30 competitors?",
            "count_over_threshold",
            {"sport": "cycling", "year": 2000, "threshold": 30},
        ),
        (
            "Count the biathlon events at the 2018 Winter Olympics with more than 73 competitors.",
            "count_over_threshold",
            {"sport": "biathlon", "threshold": 73},
        ),
        (
            "Who won gold at Placeholder Arena in March 1850?",
            "events_at_venue_date",
            {"venue": "Placeholder Arena"},
        ),
        (
            "What's the nation count for Curling at the 1850 Summer Olympics – Mixed team?",
            "lookup_nations",
            {
                "event_title": "Curling at the 1850 Summer Olympics – Mixed team",
            },
        ),
        (
            "How many curling events at the 1850 Summer Olympics exceeded 10 competitors?",
            "count_over_threshold",
            {"year": 1850, "threshold": 10},
        ),
        (
            "Who won gold at Olympic Weightlifting Gymnasium on New Year's Day 1988?",
            "events_at_venue_date",
            {"venue": "Olympic Weightlifting Gymnasium", "date_text": "New Year's Day 1988"},
        ),
        (
            "Who won the gold medal in the men's 20 kilometres walk athletics event at the Summer Olympics held immediately before 2016?",
            "previous_event_gold",
            {"named_year": 2016},
        ),
        (
            "According to the provided corpus, how many cycling events at the 2000 Summer Olympics had over 30 competitors?",
            "count_over_threshold",
            {"threshold": 30},
        ),
    )

    def setUp(self) -> None:
        self.parser = _parser()

    def test_each_paraphrase_maps_to_one_existing_operation(self) -> None:
        for question, operation, slots in self.CASES:
            with self.subTest(question=question):
                spec = self.parser.parse(question)
                self.assertTrue(spec.matched_template, question)
                self.assertEqual(spec.operation, operation)
                families = self.parser.matching_families(question)
                self.assertEqual(len(families), 1, families)
                for key, expected in slots.items():
                    self.assertEqual(getattr(spec, key), expected, key)


@unittest.skipUnless(CORPUS_AVAILABLE and PUBLIC_AVAILABLE, "corpus or public questions missing")
class PublicParseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        corpus = parse_corpus(DEFAULT_CORPUS_PATH)
        cls.index, cls.parser = build_parser(corpus)

    def test_every_public_question_still_parses(self) -> None:
        for record in load_questions(DEFAULT_PUBLIC_QUESTIONS_PATH):
            with self.subTest(qid=record["qid"]):
                spec = self.parser.parse(record["question"], qtype=record.get("qtype"))
                self.assertTrue(spec.matched_template, record["question"])


if __name__ == "__main__":
    unittest.main()
