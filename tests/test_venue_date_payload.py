"""Live-path venue/date tightening uses GSQL date payloads, not a local catalog."""

from __future__ import annotations

import unittest

from retrieval.graph.contract import interpret_installed_result
from retrieval.graphrag.retriever import GraphRetriever
from retrieval.structured.models import QuerySpec


def _venue_spec(venue: str, date_text: str, year: int | None = None) -> QuerySpec:
    return QuerySpec(
        qtype="multi_hop",
        operation="events_at_venue_date",
        raw_question="",
        matched_template=True,
        venue=venue,
        date_text=date_text,
        year=year,
        requested_field="gold",
    )


class PayloadClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.execute_calls = 0
        self.query_names: list[str] = []

    def execute_spec(self, spec: QuerySpec, events_by_id: dict | None = None):
        self.execute_calls += 1
        return interpret_installed_result("events_at_venue_date", spec, self.payload, events_by_id)

    def run_query(self, name: str, params: dict | None = None) -> dict:
        self.query_names.append(name)
        raise AssertionError(f"unexpected extra query {name}")


class VenueDatePayloadTightenTests(unittest.TestCase):
    def test_tighter_date_span_is_unique_without_local_catalog(self) -> None:
        spec = _venue_spec("London Velopark", "3 to 4 August at the 2012 Summer Olympics", 2012)
        payload = {
            "event_ids": ["Qwide", "Qexact"],
            "titles": ["Wide cycling event", "Exact cycling event"],
            "gold": ["Lasse Norman Hansen", "Dani King"],
            "date_raw": ["4 to 5 August", "3 to 4 August"],
            "date_years": ["|2012|", "|2012|"],
            "date_months": ["|8|", "|8|"],
            "date_days": ["|4||5|", "|3||4|"],
            "venue_hits": 12,
            "date_overlap_hits": 2,
        }
        result = interpret_installed_result("events_at_venue_date", spec, payload)
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.event_ids, ["Qexact"])
        self.assertEqual(result.gold, ["Dani King"])
        self.assertEqual(result.notes["date_overlap_hits"], 2)
        self.assertEqual(result.notes["after_date_tighten"], 1)

    def test_gold_values_do_not_select_the_candidate(self) -> None:
        spec = _venue_spec("Test Hall", "11 August 2016", 2016)
        payload = {
            "event_ids": ["Qwide", "Qexact"],
            "titles": ["Wide", "Exact"],
            "gold": ["Wanted Gold", "Other Gold"],
            "date_raw": ["6 to 21 August", "11 August"],
            "date_years": ["|2016|", "|2016|"],
            "date_months": ["|8|", "|8|"],
            "date_days": ["|6||7||8||9||10||11||12||13||14||15||16||17||18||19||20||21|", "|11|"],
            "venue_hits": 4,
            "date_overlap_hits": 2,
        }
        result = interpret_installed_result("events_at_venue_date", spec, payload)
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.event_ids, ["Qexact"])
        self.assertEqual(result.gold, ["Other Gold"])

    def test_identical_dates_remain_ambiguous(self) -> None:
        spec = _venue_spec("Laura Biathlon & Ski Complex", "22 February 2014", 2014)
        payload = {
            "event_ids": ["Qrelay", "Qski"],
            "titles": ["Men's relay", "Women's 30 kilometre freestyle"],
            "gold": ["Erik Lesser", "Marit Bjørgen"],
            "date_raw": ["22 February 2014", "22 February 2014"],
            "date_years": ["|2014|", "|2014|"],
            "date_months": ["|2|", "|2|"],
            "date_days": ["|22|", "|22|"],
            "venue_hits": 2,
            "date_overlap_hits": 2,
        }
        result = interpret_installed_result("events_at_venue_date", spec, payload)
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.reason, "multiple_events_same_venue_date")
        self.assertEqual(result.event_ids, ["Qrelay", "Qski"])
        self.assertEqual(result.notes["after_date_tighten"], 2)

    def test_missing_date_lists_keep_overlap_ambiguous(self) -> None:
        spec = _venue_spec("London Velopark", "3 to 4 August 2012", 2012)
        payload = {
            "event_ids": ["Qwide", "Qexact"],
            "titles": ["Wide", "Exact"],
            "gold": ["A", "B"],
            "venue_hits": 2,
            "date_overlap_hits": 2,
        }
        result = interpret_installed_result("events_at_venue_date", spec, payload)
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.event_ids, ["Qwide", "Qexact"])
        self.assertNotIn("after_date_tighten", result.notes)

    def test_misaligned_date_lists_keep_overlap_ambiguous(self) -> None:
        spec = _venue_spec("London Velopark", "3 to 4 August 2012", 2012)
        payload = {
            "event_ids": ["Qwide", "Qexact"],
            "titles": ["Wide", "Exact"],
            "gold": ["A", "B"],
            "date_raw": ["3 to 4 August"],
            "date_years": ["|2012|"],
            "date_months": ["|8|"],
            "date_days": ["|3||4|"],
            "venue_hits": 2,
            "date_overlap_hits": 2,
        }
        result = interpret_installed_result("events_at_venue_date", spec, payload)
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.event_ids, ["Qwide", "Qexact"])

    def test_lookup_payload_is_unchanged(self) -> None:
        spec = QuerySpec(
            qtype="lookup",
            operation="lookup_nations",
            raw_question="",
            matched_template=True,
            event_title="Judo at the 2016 Summer Olympics – Women's 57 kg",
        )
        payload = {
            "event_ids": ["Qjudo"],
            "titles": ["Judo at the 2016 Summer Olympics – Women's 57 kg"],
            "has_nations": [True],
            "nations": [23],
        }
        result = interpret_installed_result("lookup_event", spec, payload)
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.event_ids, ["Qjudo"])
        self.assertEqual(result.nations, [23])

    def test_aggregation_payload_is_unchanged(self) -> None:
        spec = QuerySpec(
            qtype="aggregation",
            operation="count_over_threshold",
            raw_question="",
            matched_template=True,
            sport="cycling",
            year=2000,
            season="Summer",
            threshold=30,
        )
        payload = {
            "event_ids": ["Q1", "Q2"],
            "count": 2,
            "pool_size": 10,
            "usable_with_competitors": 8,
            "missing_competitors": 2,
        }
        result = interpret_installed_result("count_over_threshold", spec, payload)
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.count, 2)
        self.assertEqual(result.event_ids, ["Q1", "Q2"])

    def test_superlative_payload_is_unchanged(self) -> None:
        spec = QuerySpec(
            qtype="superlative",
            operation="argmax_competitors",
            raw_question="",
            matched_template=True,
            sport="cycling",
            year=2000,
            season="Summer",
        )
        payload = {
            "event_ids": ["Qmax"],
            "titles": ["Cycling sprint"],
            "pool_size": 5,
            "usable_with_competitors": 5,
            "max_competitors": 84,
        }
        result = interpret_installed_result("argmax_competitors", spec, payload)
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.event_ids, ["Qmax"])
        self.assertEqual(result.max_competitors, 84)

    def test_temporal_payload_is_unchanged(self) -> None:
        spec = QuerySpec(
            qtype="temporal",
            operation="previous_event_gold",
            raw_question="",
            matched_template=True,
            sport="athletics",
            event_name="men's pole vault",
            year=2016,
            season="Summer",
            temporal_relation="previous",
        )
        payload = {
            "event_ids": ["Q2012"],
            "titles": ["Men's pole vault 2012"],
            "gold": ["Renaud Lavillenie"],
            "named_ids": ["Q2016"],
            "named_year_hits": 1,
            "target_year": 2012,
            "related_hits": 1,
        }
        result = interpret_installed_result("previous_event_gold", spec, payload)
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.event_ids, ["Q2012"])
        self.assertEqual(result.gold, ["Renaud Lavillenie"])

    def test_graphrag_still_one_execute_spec(self) -> None:
        payload = {
            "event_ids": ["Qwide", "Qexact"],
            "titles": ["Wide", "Exact"],
            "gold": ["A", "B"],
            "date_raw": ["4 to 5 August", "3 to 4 August"],
            "date_years": ["|2012|", "|2012|"],
            "date_months": ["|8|", "|8|"],
            "date_days": ["|4||5|", "|3||4|"],
            "venue_hits": 2,
            "date_overlap_hits": 2,
        }
        backend = PayloadClient(payload)
        retriever = GraphRetriever(backend)
        spec = _venue_spec("London Velopark", "3 to 4 August at the 2012 Summer Olympics", 2012)
        result = retriever.retrieve(spec)
        self.assertEqual(backend.execute_calls, 1)
        self.assertEqual(backend.query_names, [])
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.event_ids, ["Qexact"])
        self.assertEqual(result.cardinality, "one")


if __name__ == "__main__":
    unittest.main()
