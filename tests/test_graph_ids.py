"""Identity helpers for the TigerGraph schema."""

from __future__ import annotations

import unittest

from ingestion.graph_ids import (
    chunk_id,
    compact_venue,
    event_name_core,
    games_id,
    gsql_venue_score,
    packed_ints,
    packed_overlap,
    query_venue_tokens,
    sport_id,
    venue_id,
    venue_token_blob,
)
from retrieval.structured.match import compact_alnum, venue_score


class GraphIdTests(unittest.TestCase):
    def test_chunk_id_is_stable(self) -> None:
        self.assertEqual(chunk_id("Q303623", 0), "Q303623::c000")
        self.assertEqual(chunk_id("Q303623", 12), "Q303623::c012")

    def test_games_and_sport_ids(self) -> None:
        self.assertEqual(games_id(2016, "Summer"), "2016_Summer")
        self.assertIsNone(games_id(None, "Summer"))
        self.assertEqual(sport_id("Cross-country skiing"), "cross-country skiing")

    def test_venue_keys_are_conservative(self) -> None:
        self.assertNotEqual(venue_id("Riocentro – Pavilion 4"), venue_id("Riocentro – Pavilion 6"))
        self.assertEqual(venue_id("Eton Dorney"), venue_id("eton  dorney"))

    def test_packed_overlap_matches_set_intersection(self) -> None:
        self.assertTrue(packed_overlap("|11|12|13|", "|13|19|"))
        self.assertFalse(packed_overlap("|6|7|8|", "|11|19|"))
        self.assertTrue(packed_overlap("", "|22|"))
        self.assertEqual(packed_ints((8, 8, 7)), "|8|7|")

    def test_gsql_venue_score_matches_phase2_on_examples(self) -> None:
        query = "Riocentro – Pavilion 4"
        venue = "Riocentro – Pavilion 4"
        self.assertEqual(
            gsql_venue_score(compact_venue(query), query_venue_tokens(query), compact_venue(venue), venue_token_blob(venue)),
            venue_score(query, venue),
        )
        self.assertEqual(
            gsql_venue_score(
                compact_alnum("Laura Biathlon"),
                query_venue_tokens("Laura Biathlon"),
                compact_alnum("Laura Biathlon & Ski Complex"),
                venue_token_blob("Laura Biathlon & Ski Complex"),
            ),
            venue_score("Laura Biathlon", "Laura Biathlon & Ski Complex"),
        )

    def test_event_name_core_strips_gender(self) -> None:
        self.assertIn("20 kilometre walk", event_name_core("men's 20 kilometres walk"))


if __name__ == "__main__":
    unittest.main()
