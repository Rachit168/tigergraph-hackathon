"""Unit tests for the deterministic Olympic event parser."""

from __future__ import annotations

import unittest

from ingestion.parse import (
    extract_infobox,
    looks_like_olympic_event_title,
    parse_date_span,
    parse_document,
    parse_int_value,
    parse_title,
    parse_year_value,
    previous_olympiad_year,
)

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
  silver: Fernando PimentaEmanuel Silva
  silverNOC: POR
  bronze: Martin HollsteinAndreas Ihle
  bronzeNOC: GER
  prev: 2008
  next: 2016

The men's canoe sprint K-2 1,000 metres competition at the 2012 Olympic Games in London took place between 6 and 8 August at Eton Dorney.
"""

SAILING_CONFLICT_TEXT = """[Infobox Olympic event]
  event: Women's 470
  games: 1984 Summer
  venue: Busan
  dates: 20–27 September
  competitors: 42
  nations: 21
  gold: Allison Jolly Lynne Jewell
  goldNOC: USA
  prev: 1984
  next: 1992
"""

BODY_ONLY_CYCLING = """These are the official results of the Women's Individual Pursuit at the 2000 Summer Olympics in Sydney, Australia. The races were held on Sunday, 17 September.

Medalists
"""


class TitleParsingTests(unittest.TestCase):
    def test_parses_sport_year_season_event(self) -> None:
        parsed = parse_title(CANOE_TITLE)
        self.assertTrue(parsed.parsed)
        self.assertEqual(parsed.sport, "Canoeing")
        self.assertEqual(parsed.year, 2012)
        self.assertEqual(parsed.season, "Summer")
        self.assertEqual(parsed.event_name, "Men's K-2 1000 metres")

    def test_accepts_hyphen_dash(self) -> None:
        parsed = parse_title("Judo at the 2016 Summer Olympics - Women's 57 kg")
        self.assertTrue(parsed.parsed)
        self.assertEqual(parsed.event_name, "Women's 57 kg")

    def test_games_overview_title_has_no_event_name(self) -> None:
        parsed = parse_title("Athletics at the 2008 Summer Olympics")
        self.assertTrue(parsed.parsed)
        self.assertIsNone(parsed.event_name)
        self.assertFalse(looks_like_olympic_event_title(parsed.raw))

    def test_distractor_title_rejected(self) -> None:
        parsed = parse_title("Forrest Gump")
        self.assertFalse(parsed.parsed)
        self.assertFalse(looks_like_olympic_event_title("Forrest Gump"))

    def test_winter_season(self) -> None:
        parsed = parse_title("Biathlon at the 2018 Winter Olympics – Men's sprint")
        self.assertEqual(parsed.season, "Winter")
        self.assertEqual(parsed.year, 2018)


class InfoboxParsingTests(unittest.TestCase):
    def test_extracts_fields(self) -> None:
        has_box, fields, header = extract_infobox(CANOE_TEXT)
        self.assertTrue(has_box)
        self.assertIn("Infobox Olympic event", header or "")
        self.assertEqual(fields["competitors"], "24")
        self.assertEqual(fields["gold"], "Rudolf DombiRoland Kökény")
        self.assertEqual(fields["prev"], "2008")

    def test_stops_before_body(self) -> None:
        _, fields, _ = extract_infobox(CANOE_TEXT)
        self.assertNotIn("The men's canoe", " ".join(fields.values()))

    def test_missing_infobox(self) -> None:
        has_box, fields, header = extract_infobox(BODY_ONLY_CYCLING)
        self.assertFalse(has_box)
        self.assertEqual(fields, {})
        self.assertIsNone(header)


class DocumentParseTests(unittest.TestCase):
    def test_event_from_infobox(self) -> None:
        document = parse_document(
            {
                "doc_id": "Q303623",
                "title": CANOE_TITLE,
                "url": "https://example.test/canoe",
                "text": CANOE_TEXT,
                "approx_tokens": 704,
            }
        )
        self.assertTrue(document.is_event)
        assert document.event is not None
        event = document.event
        self.assertEqual(event.sport, "canoeing")
        self.assertEqual(event.year, 2012)
        self.assertEqual(event.season, "Summer")
        self.assertEqual(event.competitors, 24)
        self.assertEqual(event.nations, 12)
        self.assertEqual(event.gold_raw(), "Rudolf DombiRoland Kökény")
        self.assertEqual(event.medals[0].noc, "HUN")
        self.assertEqual(event.prev_year, 2008)
        self.assertEqual(event.next_year, 2016)
        self.assertEqual(event.venue_raw, "Eton Dorney")
        self.assertEqual(event.date.source_field, "date")
        self.assertIn(6, event.date.days)
        self.assertIn(8, event.date.days)
        self.assertEqual(event.date.months, (8,))

    def test_concatenated_gold_is_not_split(self) -> None:
        document = parse_document({"doc_id": "Q1", "title": CANOE_TITLE, "text": CANOE_TEXT})
        assert document.event is not None
        self.assertEqual(document.event.gold_raw(), "Rudolf DombiRoland Kökény")
        gold_medals = [medal for medal in document.event.medals if medal.place == "gold"]
        self.assertEqual(len(gold_medals), 1)
        self.assertNotIn("Rudolf Dombi", [medal.name_raw for medal in gold_medals])

    def test_provenance_keeps_original_field_names(self) -> None:
        document = parse_document({"doc_id": "Q303623", "title": CANOE_TITLE, "text": CANOE_TEXT})
        assert document.event is not None
        provenance = document.event.provenance
        self.assertEqual(provenance["competitors"].field_name, "competitors")
        self.assertEqual(provenance["competitors"].raw_text, "24")
        self.assertEqual(provenance["competitors"].document_id, "Q303623")
        self.assertEqual(provenance["competitors"].page_title, CANOE_TITLE)
        self.assertEqual(provenance["gold"].raw_text, "Rudolf DombiRoland Kökény")
        self.assertEqual(provenance["sport"].field_name, "title")

    def test_distractor_is_document_only(self) -> None:
        document = parse_document(
            {
                "doc_id": "Qfilm",
                "title": "Forrest Gump",
                "text": "Forrest Gump is a 1994 American comedy-drama film.",
            }
        )
        self.assertFalse(document.is_event)
        self.assertEqual(document.kind, "document")
        self.assertEqual(document.rejection_reason, "no_olympic_event_infobox")
        self.assertIsNone(document.event)

    def test_olympic_title_without_infobox_is_rejected(self) -> None:
        document = parse_document(
            {
                "doc_id": "Q3046361",
                "title": "Cycling at the 2000 Summer Olympics – Women's individual pursuit",
                "text": BODY_ONLY_CYCLING,
            }
        )
        self.assertFalse(document.is_event)
        self.assertEqual(document.rejection_reason, "olympic_title_without_infobox")
        self.assertIn("rejected_event_title", document.flags)

    def test_does_not_read_competitors_from_body(self) -> None:
        document = parse_document(
            {
                "doc_id": "Qkeirin",
                "title": "Cycling at the 2000 Summer Olympics – Men's keirin",
                "text": "The men's keirin in cycling at the 2000 Summer Olympics was contested by 20 cyclists.\n",
            }
        )
        self.assertIsNone(document.event)

    def test_title_preferred_when_games_conflict(self) -> None:
        document = parse_document(
            {
                "doc_id": "Q7400295",
                "title": "Sailing at the 1988 Summer Olympics – Women's 470",
                "text": SAILING_CONFLICT_TEXT,
            }
        )
        assert document.event is not None
        self.assertEqual(document.event.year, 1988)
        self.assertEqual(document.event.season, "Summer")
        self.assertEqual(document.event.infobox_games_raw, "1984 Summer")
        self.assertEqual(document.event.year_source, "title_preferred")
        self.assertIn("title_infobox_games_conflict", document.event.flags)

    def test_missing_optional_fields_are_unknown(self) -> None:
        text = """[Infobox Olympic event]
  event: Mystery event
  games: 2012 Summer
  gold: Somebody
"""
        document = parse_document(
            {
                "doc_id": "Qmissing",
                "title": "Fencing at the 2012 Summer Olympics – Men's foil",
                "text": text,
            }
        )
        assert document.event is not None
        self.assertIsNone(document.event.competitors)
        self.assertIsNone(document.event.nations)
        self.assertIsNone(document.event.venue_raw)
        self.assertIsNone(document.event.date)
        self.assertIn("missing_competitors", document.event.flags)
        self.assertIn("missing_nations", document.event.flags)

    def test_malformed_competitors(self) -> None:
        text = """[Infobox Olympic event]
  event: Mystery event
  games: 2012 Summer
  competitors: unknown
  gold: Somebody
"""
        document = parse_document(
            {
                "doc_id": "Qbad",
                "title": "Fencing at the 2012 Summer Olympics – Men's foil",
                "text": text,
            }
        )
        assert document.event is not None
        self.assertEqual(document.event.competitors_raw, "unknown")
        self.assertIsNone(document.event.competitors)
        self.assertIn("malformed_competitors", document.event.flags)

    def test_dates_field_and_range(self) -> None:
        document = parse_document(
            {
                "doc_id": "Qsail",
                "title": "Sailing at the 1988 Summer Olympics – Women's 470",
                "text": SAILING_CONFLICT_TEXT,
            }
        )
        assert document.event is not None
        assert document.event.date is not None
        self.assertEqual(document.event.date.source_field, "dates")
        self.assertIn(20, document.event.date.days)
        self.assertIn(27, document.event.date.days)

    def test_venue_normalization_is_light(self) -> None:
        document = parse_document({"doc_id": "Q303623", "title": CANOE_TITLE, "text": CANOE_TEXT})
        assert document.event is not None
        self.assertEqual(document.event.venue_key, "eton dorney")
        pavilion = parse_document(
            {
                "doc_id": "Qp4",
                "title": CANOE_TITLE,
                "text": CANOE_TEXT.replace("Eton Dorney", "Riocentro – Pavilion 4"),
            }
        )
        pavilion6 = parse_document(
            {
                "doc_id": "Qp6",
                "title": CANOE_TITLE,
                "text": CANOE_TEXT.replace("Eton Dorney", "Riocentro – Pavilion 6"),
            }
        )
        self.assertNotEqual(pavilion.event.venue_key, pavilion6.event.venue_key)

    def test_infobox_without_title_parse_still_creates_event(self) -> None:
        text = """[Infobox Olympic event]
  event: Special
  games: 2010 Winter
  competitors: 8
  nations: 8
  gold: Ada
"""
        document = parse_document({"doc_id": "Qodd", "title": "1994 Winter Olympics", "text": text})
        self.assertTrue(document.is_event)
        assert document.event is not None
        self.assertEqual(document.event.year, 2010)
        self.assertEqual(document.event.season, "Winter")
        self.assertEqual(document.event.year_source, "infobox")
        self.assertIn("infobox_without_title_parse", document.event.flags)


class NumericAndDateHelpersTests(unittest.TestCase):
    def test_parse_int_and_year(self) -> None:
        self.assertEqual(parse_int_value("24"), 24)
        self.assertEqual(parse_int_value(" 12 extras"), 12)
        self.assertIsNone(parse_int_value("none"))
        self.assertEqual(parse_year_value("2008"), 2008)
        self.assertIsNone(parse_year_value("next time"))

    def test_date_span_full_date(self) -> None:
        span = parse_date_span("13 February 2010", "date")
        assert span is not None
        self.assertEqual(span.years, (2010,))
        self.assertEqual(span.months, (2,))
        self.assertEqual(span.days, (13,))

    def test_date_span_new_years_day_alias(self) -> None:
        span = parse_date_span("New Year's Day 2004", "question")
        assert span is not None
        self.assertEqual(span.years, (2004,))
        self.assertEqual(span.months, (1,))
        self.assertEqual(span.days, (1,))

    def test_previous_olympiad_calendar(self) -> None:
        self.assertEqual(previous_olympiad_year(2016, "Summer"), 2012)
        self.assertEqual(previous_olympiad_year(2020, "Summer"), 2016)
        self.assertEqual(previous_olympiad_year(1994, "Winter"), 1992)


if __name__ == "__main__":
    unittest.main()
