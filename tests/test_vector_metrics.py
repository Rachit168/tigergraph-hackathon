"""Family accounting for Phase 13C vector reporting. No live graph."""

from __future__ import annotations

import unittest

from experiments.vector_metrics import (
    GOLD_QTYPES,
    RECALL_AT_K_DEFINITION,
    family_for_row,
    summarize_vector_ablation,
)


class VectorFamilyAccountingTests(unittest.TestCase):
    def test_multi_hop_is_not_aliased_to_venue_disambiguation(self) -> None:
        self.assertEqual(family_for_row({"qtype": "multi_hop"}), "multi_hop")
        self.assertNotIn("venue_disambiguation", GOLD_QTYPES)
        rows = [
            _row("pub-a", "V0", "multi_hop", True, "answered"),
            _row("pub-a", "V1", "multi_hop", False, "abstained"),
            _row("pub-a", "V2", "multi_hop", True, "answered"),
        ]
        families = summarize_vector_ablation(rows)["by_family"]
        self.assertIn("multi_hop", families)
        self.assertNotIn("venue_disambiguation", families)
        self.assertEqual(families["multi_hop"]["V0"]["n"], 1)
        self.assertEqual(families["multi_hop"]["V1"]["n"], 1)
        self.assertEqual(families["multi_hop"]["V2"]["n"], 1)

    def test_status_buckets_are_separate_from_gold_qtypes(self) -> None:
        rows = [
            _row("pub-a", "V0", "lookup", True, "answered"),
            _row("pub-b", "V0", "temporal", False, "abstained"),
            _row("pub-c", "V0", "multi_hop", False, "ambiguous"),
        ]
        families = summarize_vector_ablation(rows)["by_family"]
        self.assertEqual(families["lookup"]["V0"]["n"], 1)
        self.assertEqual(families["temporal"]["V0"]["n"], 1)
        self.assertEqual(families["multi_hop"]["V0"]["n"], 1)
        self.assertEqual(families["abstention"]["V0"]["n"], 1)
        self.assertEqual(families["ambiguity"]["V0"]["n"], 1)
        # Status buckets are orthogonal: an abstained temporal row is in both families.

    def test_recall_at_k_is_documented_as_any_gold_hit(self) -> None:
        self.assertIn("at least one gold document", RECALL_AT_K_DEFINITION)
        self.assertIn("not full-set recall", RECALL_AT_K_DEFINITION)


class TemporalGateHelperTests(unittest.TestCase):
    def test_parse_named_year_and_previous_gold(self) -> None:
        from experiments.temporal_vector_gate import parse_named_year, previous_olympiad_gold

        self.assertEqual(parse_named_year("held immediately before 2016?"), 2016)
        self.assertEqual(parse_named_year("prior edition of the 2020 Summer Olympics"), 2020)
        self.assertEqual(parse_named_year("previous Olympiad before 2022"), 2022)
        earlier = _doc("Q1", 2012)
        later = _doc("Q2", 2016)
        meta = previous_olympiad_gold(["Q1", "Q2"], 2016, {"Q1": earlier, "Q2": later})
        self.assertEqual(meta["previous_olympiad_doc_id"], "Q1")
        self.assertEqual(meta["named_year_doc_id"], "Q2")
        self.assertTrue(meta["identifiable"])

    def test_recording_retriever_keeps_chunk_ids(self) -> None:
        from experiments.temporal_vector_gate import RecordingRetriever, hits_payload
        from retrieval.rag.models import RetrievalHit, RetrievalResult

        class Inner:
            def retrieve(self, query, **kwargs):
                del query, kwargs
                return RetrievalResult(
                    query="q",
                    method="bm25",
                    hits=[
                        RetrievalHit(
                            chunk_id="Q1::c000",
                            document_id="Q1",
                            document_title="t",
                            score=1.0,
                            rank=1,
                            text="x",
                            section="lead",
                            kind="lead",
                            event_id="Q1",
                            retrieval_method="bm25",
                        )
                    ],
                    params={"candidate_chunk_ids": ["Q1::c000"], "candidate_document_ids": ["Q1"]},
                )

        wrapped = RecordingRetriever(Inner())
        result = wrapped.retrieve("q")
        self.assertEqual(result.params["candidate_chunk_ids"], ["Q1::c000"])
        self.assertEqual(hits_payload(wrapped.last_result)[0]["chunk_id"], "Q1::c000")


def _doc(doc_id: str, year: int):
    from ingestion.models import ParsedDocument, ParsedEvent, TitleParse

    event = ParsedEvent(
        event_id=doc_id,
        document_id=doc_id,
        title=f"{year} event",
        sport_raw=None,
        sport=None,
        event_name_raw=None,
        infobox_event=None,
        year=year,
        season="Summer",
        games_id=None,
        infobox_games_raw=None,
        year_source="title",
        venue_raw=None,
        venue_key=None,
        date=None,
        competitors_raw=None,
        competitors=None,
        nations_raw=None,
        nations=None,
        medals=[],
        prev_year_raw=None,
        prev_year=None,
        next_year_raw=None,
        next_year=None,
    )
    return ParsedDocument(
        doc_id=doc_id,
        title=f"{year} event",
        url=None,
        wikidata_qid=doc_id,
        wikipedia_pageid=None,
        approx_tokens=1,
        text="x",
        has_olympic_infobox=True,
        title_parse=TitleParse(raw=f"{year} event", sport=None, year=year, season="Summer", event_name=None, parsed=True),
        infobox={},
        kind="event",
        event=event,
    )


def _row(qid: str, variant: str, qtype: str, correct: bool, status: str) -> dict:
    return {
        "qid": qid,
        "qtype": qtype,
        "variant": variant,
        "correctness": correct,
        "correctness_exact": correct,
        "completeness": None,
        "grounding": True,
        "citation_validity": True,
        "answer_status": status,
        "tokens": 10,
        "latency_ms": 1.0,
        "retrieval_ms": 1.0,
        "generation_ms": 1.0,
        "recall_at_5": True,
        "recall_at_10": True,
        "candidate_recall": 1.0,
        "packed_recall": 1.0,
        "failure_category": None if correct else status,
    }


if __name__ == "__main__":
    unittest.main()
