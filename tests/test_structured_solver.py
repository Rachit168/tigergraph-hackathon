"""Unit tests for the offline structured solver."""

from __future__ import annotations

import unittest

from evaluation.normalize import answers_match_exact, answers_match_normalized, normalize_answer
from ingestion.parse import parse_document
from retrieval.structured.index import StructuredIndex
from retrieval.structured.match import event_names_match, venue_score
from retrieval.structured.question import QuestionParser
from retrieval.structured.solver import StructuredSolver


def _event_doc(
    doc_id: str,
    title: str,
    *,
    competitors: str | None = "10",
    nations: str | None = "8",
    gold: str = "Ada Lovelace",
    venue: str = "Test Hall",
    date: str = "12 August 2016",
    prev: str | None = "2012",
    extra: str = "",
) -> dict:
    lines = ["[Infobox Olympic event]", f"  event: {title}", "  games: 2016 Summer"]
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
    lines.append(extra)
    return {"doc_id": doc_id, "title": title, "text": "\n".join(lines)}


def _index_from(*records: dict) -> StructuredIndex:
    documents = [parse_document(record) for record in records]
    return StructuredIndex.from_documents(documents)


class QuestionParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = QuestionParser(["athletics", "judo", "cross-country skiing"])

    def test_lookup(self) -> None:
        spec = self.parser.parse("How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?")
        self.assertEqual(spec.qtype, "lookup")
        self.assertEqual(spec.event_title, "Judo at the 2016 Summer Olympics – Women's 57 kg")
        self.assertEqual(spec.requested_field, "nations")

    def test_aggregation(self) -> None:
        spec = self.parser.parse(
            "According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?"
        )
        self.assertEqual(spec.operation, "count_over_threshold")
        self.assertEqual(spec.sport, "biathlon")
        self.assertEqual(spec.year, 2018)
        self.assertEqual(spec.threshold, 73)

    def test_temporal_splits_sport(self) -> None:
        spec = self.parser.parse(
            "Who won the gold medal in the men's 20 kilometres walk athletics event at the Summer Olympics held immediately before 2016?"
        )
        self.assertEqual(spec.sport, "athletics")
        self.assertEqual(spec.event_name, "men's 20 kilometres walk")
        self.assertEqual(spec.named_year, 2016)
        self.assertEqual(spec.temporal_relation, "previous")

    def test_multi_hop_year(self) -> None:
        spec = self.parser.parse(
            "Who won the gold medal in the event held at Riocentro – Pavilion 4 on 11–19 August at the 2016 Summer Olympics?"
        )
        self.assertEqual(spec.venue, "Riocentro – Pavilion 4")
        self.assertEqual(spec.year, 2016)


class NormalizationTests(unittest.TestCase):
    def test_numeric_list(self) -> None:
        self.assertTrue(answers_match_exact(["12"], ["12"]))
        self.assertTrue(answers_match_normalized("12", ["12"]))
        self.assertEqual(normalize_answer(["12"]), ("num", (12,)))

    def test_title_dash_and_case(self) -> None:
        gold = ["Athletics at the 2008 Summer Olympics – Men's marathon"]
        pred = ["athletics at the 2008 summer olympics - men's marathon"]
        self.assertFalse(answers_match_exact(pred, gold))
        self.assertTrue(answers_match_normalized(pred, gold))

    def test_none_is_not_a_match(self) -> None:
        self.assertFalse(answers_match_normalized(None, ["12"]))


class LookupSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        title = "Judo at the 2016 Summer Olympics – Women's 57 kg"
        self.solver = StructuredSolver(
            _index_from(
                _event_doc("Qjudo", title, nations="23", gold="Rafaela Silva"),
                _event_doc(
                    "Qsibling",
                    "Judo at the 2016 Summer Olympics – Women's 52 kg",
                    nations="19",
                ),
            )
        )
        self.title = title

    def test_exact_event_lookup(self) -> None:
        result = self.solver.solve(f"How many nations competed in {self.title}?")
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.answer, ["23"])
        self.assertEqual(result.events[0].event_id, "Qjudo")
        self.assertTrue(result.evidence)
        self.assertEqual(result.evidence[0].document_id, "Qjudo")
        self.assertEqual(result.evidence[0].field_name, "nations")

    def test_event_not_found(self) -> None:
        result = self.solver.solve("How many nations competed in Judo at the 1992 Summer Olympics – Women's 57 kg?")
        self.assertEqual(result.status, "not_found")
        self.assertIsNone(result.answer)

    def test_ambiguous_lookup(self) -> None:
        title = "Sailing at the 2016 Summer Olympics – Women's RS:X"
        solver = StructuredSolver(
            _index_from(
                _event_doc("Qa", title, nations="26"),
                _event_doc("Qb", title, nations="10"),
            )
        )
        result = solver.solve(f"How many nations competed in {title}?")
        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.answer)

    def test_missing_nations(self) -> None:
        title = "Fencing at the 2012 Summer Olympics – Men's foil"
        solver = StructuredSolver(_index_from(_event_doc("Qmiss", title, nations=None)))
        result = solver.solve(f"How many nations competed in {title}?")
        self.assertEqual(result.status, "unresolved")
        self.assertEqual(result.reason, "missing_field:nations")


class AggregationAndSuperlativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = StructuredSolver(
            _index_from(
                _event_doc(
                    "Q1",
                    "Cycling at the 2000 Summer Olympics – Men's sprint",
                    competitors="40",
                    gold="A",
                ),
                _event_doc(
                    "Q2",
                    "Cycling at the 2000 Summer Olympics – Men's keirin",
                    competitors="20",
                    gold="B",
                ),
                {
                    "doc_id": "Q3046361",
                    "title": "Cycling at the 2000 Summer Olympics – Women's individual pursuit",
                    "text": "These are the official results of the Women's Individual Pursuit.\n",
                },
            )
        )

    def test_aggregation_counts_only_events(self) -> None:
        result = self.solver.solve(
            "According to the provided corpus, how many cycling events at the 2000 Summer Olympics had more than 30 competitors?"
        )
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.answer, ["1"])
        self.assertEqual([event.event_id for event in result.events], ["Q1"])

    def test_title_only_document_is_not_an_event(self) -> None:
        document = self.solver.index.document("Q3046361")
        assert document is not None
        self.assertFalse(document.is_event)
        self.assertNotIn("Q3046361", self.solver.index.event_by_id)

    def test_superlative(self) -> None:
        result = self.solver.solve(
            "According to the provided corpus, which cycling event at the 2000 Summer Olympics had the highest number of competitors?"
        )
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.answer, ["Cycling at the 2000 Summer Olympics – Men's sprint"])


class TemporalSolverTests(unittest.TestCase):
    def test_previous_edition_same_name(self) -> None:
        solver = StructuredSolver(
            _index_from(
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
        result = solver.solve(
            "Who won the gold medal in the men's pole vault athletics event at the Summer Olympics held immediately before 2016?"
        )
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.answer, ["Renaud Lavillenie"])
        self.assertEqual(result.events[0].event_id, "Q2012")

    def test_program_change_is_unresolved(self) -> None:
        solver = StructuredSolver(
            _index_from(
                _event_doc(
                    "Q1996",
                    "Wrestling at the 1996 Summer Olympics – Men's freestyle 82 kg",
                    gold="NamedYearGold",
                    prev="1992",
                ),
                {
                    "doc_id": "Q1992",
                    "title": "Wrestling at the 1992 Summer Olympics – Men's freestyle 84 kg",
                    "text": """[Infobox Olympic event]
  event: Men's freestyle 84 kg
  games: 1992 Summer
  venue: Hall
  date: 1 August 1992
  competitors: 20
  nations: 12
  gold: DifferentClass
""",
                },
            )
        )
        result = solver.solve(
            "Who won the gold medal in the men's freestyle 82 kg wrestling event at the Summer Olympics held immediately before 1996?"
        )
        self.assertEqual(result.status, "unresolved")
        self.assertEqual(result.reason, "previous_event_name_missing")
        self.assertIsNone(result.answer)

    def test_does_not_match_opposite_gender(self) -> None:
        self.assertFalse(event_names_match("men's pole vault", "Women's pole vault"))
        self.assertTrue(event_names_match("men's pole vault", "Men's pole vault"))
        self.assertFalse(event_names_match("women's épée", "Women's team épée"))
        self.assertFalse(event_names_match("men's 80 kg", "Men's +80 kg"))


class MultiHopSolverTests(unittest.TestCase):
    def test_venue_and_date_unique(self) -> None:
        solver = StructuredSolver(
            _index_from(
                _event_doc(
                    "Qday12",
                    "Judo at the 2016 Summer Olympics – Women's 57 kg",
                    venue="Carioca Arena 3",
                    date="6 August 2016",
                    gold="Rafaela Silva",
                ),
                _event_doc(
                    "Qday7",
                    "Judo at the 2016 Summer Olympics – Women's 52 kg",
                    venue="Carioca Arena 3",
                    date="7 August 2016",
                    gold="Other",
                ),
            )
        )
        result = solver.solve(
            "Who won the gold medal in the event held at Carioca Arena 3 on 6 August 2016?"
        )
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.answer, ["Rafaela Silva"])

    def test_same_venue_same_day_is_ambiguous(self) -> None:
        solver = StructuredSolver(
            _index_from(
                _event_doc(
                    "Qa",
                    "Swimming at the 2004 Summer Olympics – Men's 400 metre freestyle",
                    venue="Olympic Aquatic Centre",
                    date="August 14, 2004",
                    gold="A",
                ),
                _event_doc(
                    "Qb",
                    "Swimming at the 2004 Summer Olympics – Women's 100 metre butterfly",
                    venue="Olympic Aquatic Centre",
                    date="August 14, 2004",
                    gold="B",
                ),
            )
        )
        result = solver.solve(
            "Who won the gold medal in the event held at Olympic Aquatic Centre on August 14, 2004 (heats & final)?"
        )
        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.answer)

    def test_venue_score_keeps_pavilions_apart(self) -> None:
        self.assertGreater(
            venue_score("Riocentro – Pavilion 4", "Riocentro – Pavilion 4"),
            venue_score("Riocentro – Pavilion 4", "Riocentro – Pavilion 6"),
        )


class ProvenanceTests(unittest.TestCase):
    def test_lookup_evidence_uses_infobox_field(self) -> None:
        title = "Sailing at the 2016 Summer Olympics – Women's RS:X"
        solver = StructuredSolver(_index_from(_event_doc("Qsail", title, nations="26")))
        result = solver.solve(f"How many nations competed in {title}?")
        self.assertEqual(result.evidence[0].raw_text, "26")
        self.assertEqual(result.evidence[0].retrieval_method, "lookup_event")
        assert result.evidence[0].provenance is not None
        self.assertEqual(result.evidence[0].provenance.field_name, "nations")


if __name__ == "__main__":
    unittest.main()
