"""Live TigerGraph checks. Skipped when Savanna/CE is not configured or reachable."""

from __future__ import annotations

import unittest

from config.settings import load_settings
from ingestion.graph_records import INSTALLED_QUERIES
from retrieval.graph.client import TigerGraphClient


@unittest.skipUnless(load_settings().configured, "TigerGraph env not configured")
class LiveTigerGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TigerGraphClient()
        if not cls.client.connect():
            raise unittest.SkipTest(cls.client.error or "TigerGraph is not reachable")

    def test_schema_and_queries(self) -> None:
        schema = self.client.verify_schema()
        queries = self.client.verify_queries()
        self.assertIsInstance(schema, dict)
        installed = queries.get("installed") or []
        if queries.get("ok"):
            for name in INSTALLED_QUERIES:
                self.assertIn(name, installed)

    def test_vertex_counts_are_nonnegative(self) -> None:
        counts = self.client.vertex_counts()
        for name in ("Document", "Chunk", "Event", "Games", "Sport", "Venue"):
            self.assertGreaterEqual(int(counts.get(name, 0)), 0)


if __name__ == "__main__":
    unittest.main()
