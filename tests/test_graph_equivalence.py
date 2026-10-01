"""Python solver vs graph-contract equivalence."""

from __future__ import annotations

import unittest

from ingestion.graph_records import build_graph_export
from ingestion.loader import load_questions, parse_corpus
from ingestion.parse import parse_document
from ingestion.paths import DEFAULT_CORPUS_PATH, DEFAULT_PUBLIC_QUESTIONS_PATH
from retrieval.graph.contract import GraphStore, compare_solver_to_graph
from retrieval.structured.index import StructuredIndex
from retrieval.structured.solver import StructuredSolver

CORPUS_AVAILABLE = DEFAULT_CORPUS_PATH.exists() and DEFAULT_PUBLIC_QUESTIONS_PATH.exists()


def _event_doc(doc_id: str, title: str, **fields) -> dict:
    lines = ["[Infobox Olympic event]", f"  event: {title}", "  games: 2016 Summer"]
    venue = fields.get("venue", "Test Hall")
    date = fields.get("date", "12 August 2016")
    competitors = fields.get("competitors", "10")
    nations = fields.get("nations", "8")
    gold = fields.get("gold", "Ada Lovelace")
    prev = fields.get("prev", "2012")
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
    return {"doc_id": doc_id, "title": title, "text": "\n".join(lines)}


def _pair(*records: dict):
    documents = [parse_document(record) for record in records]
    solver = StructuredSolver(StructuredIndex.from_documents(documents))
    store = GraphStore.from_export(build_graph_export(documents))
    return solver, store


def _assert_equivalent(test: unittest.TestCase, solver: StructuredSolver, store: GraphStore, question: str, qtype: str | None = None) -> None:
    spec = solver.parser.parse(question, qtype=qtype)
    python_result = solver.solve_spec(spec)
    graph_result = store.execute(spec)
    compared = compare_solver_to_graph(python_result, graph_result)
    test.assertTrue(compared["ok"], msg=compared)


class GraphEquivalenceTests(unittest.TestCase):
    def test_lookup(self) -> None:
        title = "Judo at the 2016 Summer Olympics – Women's 57 kg"
        solver, store = _pair(_event_doc("Qjudo", title, nations="23", gold="Rafaela Silva"))
        _assert_equivalent(self, solver, store, f"How many nations competed in {title}?")

    def test_aggregation_ignores_title_only_documents(self) -> None:
        solver, store = _pair(
            _event_doc("Q1", "Cycling at the 2000 Summer Olympics – Men's sprint", competitors="40"),
            _event_doc("Q2", "Cycling at the 2000 Summer Olympics – Men's keirin", competitors="20"),
            {
                "doc_id": "Q3046361",
                "title": "Cycling at the 2000 Summer Olympics – Women's individual pursuit",
                "text": "These are the official results of the Women's Individual Pursuit.\n",
            },
        )
        question = "According to the provided corpus, how many cycling events at the 2000 Summer Olympics had more than 30 competitors?"
        _assert_equivalent(self, solver, store, question, qtype="aggregation")
        spec = solver.parser.parse(question, qtype="aggregation")
        result = store.execute(spec)
        self.assertNotIn("Q3046361", result.event_ids)
        self.assertEqual(result.count, 1)

    def test_superlative(self) -> None:
        solver, store = _pair(
            _event_doc("Q1", "Cycling at the 2000 Summer Olympics – Men's sprint", competitors="40"),
            _event_doc("Q2", "Cycling at the 2000 Summer Olympics – Men's keirin", competitors="20"),
        )
        _assert_equivalent(
            self,
            solver,
            store,
            "According to the provided corpus, which cycling event at the 2000 Summer Olympics had the highest number of competitors?",
            qtype="superlative",
        )

    def test_temporal_and_unresolved_program_change(self) -> None:
        solver, store = _pair(
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
        _assert_equivalent(
            self,
            solver,
            store,
            "Who won the gold medal in the men's pole vault athletics event at the Summer Olympics held immediately before 2016?",
            qtype="temporal",
        )

    def test_venue_date_collision_does_not_guess(self) -> None:
        solver, store = _pair(
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
        question = "Who won the gold medal in the event held at Olympic Aquatic Centre on August 14, 2004 (heats & final)?"
        _assert_equivalent(self, solver, store, question, qtype="multi_hop")
        spec = solver.parser.parse(question, qtype="multi_hop")
        result = store.execute(spec)
        self.assertEqual(result.status, "ambiguous")
        self.assertGreaterEqual(len(result.event_ids), 2)


@unittest.skipUnless(CORPUS_AVAILABLE, "hackathon corpus is not present")
class CorpusGraphEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = parse_corpus(DEFAULT_CORPUS_PATH)
        cls.solver = StructuredSolver(StructuredIndex(cls.corpus))
        cls.store = GraphStore.from_export(build_graph_export(cls.corpus.documents))
        cls.questions = load_questions(DEFAULT_PUBLIC_QUESTIONS_PATH)

    def test_title_only_page_is_not_an_event_vertex(self) -> None:
        self.assertNotIn("Q3046361", self.store.by_id)
        self.assertFalse(self.corpus.document_by_id()["Q3046361"].is_event)

    def test_public_questions_match_python_solver(self) -> None:
        mismatches = []
        for question in self.questions:
            spec = self.solver.parser.parse(question["question"], qtype=question.get("qtype"))
            python_result = self.solver.solve_spec(spec)
            graph_result = self.store.execute(spec)
            compared = compare_solver_to_graph(python_result, graph_result)
            if not compared["ok"]:
                mismatches.append({"qid": question.get("qid"), **compared})
        self.assertEqual(mismatches, [])

    def test_known_venue_collision_is_ambiguous(self) -> None:
        question = next(item for item in self.questions if item.get("qid") == "pub-099")
        spec = self.solver.parser.parse(question["question"], qtype="multi_hop")
        result = self.store.execute(spec)
        self.assertEqual(result.status, "ambiguous")
        self.assertGreaterEqual(len(result.event_ids), 2)


if __name__ == "__main__":
    unittest.main()
