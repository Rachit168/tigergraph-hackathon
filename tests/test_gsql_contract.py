"""GSQL files and loading configuration."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from config.settings import TigerGraphSettings, load_settings
from ingestion.graph_records import INSTALLED_QUERIES
from ingestion.paths import GSQL_DIR, REPO_ROOT
from retrieval.graph.client import repo_gsql_queries


class GsqlContractTests(unittest.TestCase):
    def test_schema_has_required_types(self) -> None:
        text = (GSQL_DIR / "00_schema.gsql").read_text(encoding="utf-8")
        for name in ("Document", "Chunk", "Event", "Games", "Sport", "Venue"):
            self.assertIn(f"CREATE VERTEX {name}", text)
        for name in ("CONTAINS_CHUNK", "DESCRIBES", "IN_GAMES", "OF_SPORT", "HELD_AT"):
            self.assertIn(name, text)
        self.assertNotIn("CREATE VERTEX Athlete", text)
        self.assertNotRegex(text, r"(?i)CREATE\s+(DIRECTED\s+)?EDGE\s+PREV_EVENT")
        self.assertNotRegex(text, r"(?i)VECTOR\s*<")

    def test_installed_query_names(self) -> None:
        names = repo_gsql_queries()
        self.assertEqual(list(INSTALLED_QUERIES), names)
        query_text = (GSQL_DIR / "02_queries.gsql").read_text(encoding="utf-8")
        self.assertIn("s.has_competitors AND s.competitors > threshold", query_text)
        self.assertIn("FROM seed:s", query_text)
        self.assertIn("Event.*", query_text)
        self.assertNotRegex(query_text, r"(?i)FROM\s+@@")
        self.assertNotIn("Document.*", query_text.split("count_over_threshold", 1)[1].split("CREATE OR REPLACE QUERY argmax", 1)[0])

    def test_retrieval_gsql_is_additive(self) -> None:
        from retrieval.graphrag.models import RETRIEVAL_QUERIES

        names = repo_gsql_queries()
        self.assertEqual(list(INSTALLED_QUERIES), names)
        text = (GSQL_DIR / "04_retrieval.gsql").read_text(encoding="utf-8")
        for name in RETRIEVAL_QUERIES:
            self.assertIn(f"QUERY {name}", text)
        self.assertIn("DESCRIBES", text)
        self.assertIn("CONTAINS_CHUNK", text)
        self.assertIn("IN_GAMES", text)
        self.assertIn("OF_SPORT", text)
        self.assertIn("HELD_AT", text)
        self.assertIn("LIKE", text)
        self.assertNotRegex(text, r"(?i)FROM\s+@@")
        self.assertNotIn("DROP GRAPH", text)
        self.assertNotIn("CREATE GRAPH", text)

    def test_bind_graph_job_attaches_existing_types(self) -> None:
        text = (GSQL_DIR / "03_bind_graph.gsql").read_text(encoding="utf-8")
        executable = "\n".join(
            line for line in text.splitlines() if line.strip() and not line.strip().startswith("//")
        )
        self.assertIn("CREATE GLOBAL SCHEMA_CHANGE JOB", executable)
        self.assertIn(
            "ADD VERTEX Document, Chunk, Event, Games, Sport, Venue TO GRAPH OlympicGraph",
            executable,
        )
        self.assertIn(
            "ADD EDGE CONTAINS_CHUNK, DESCRIBES, IN_GAMES, OF_SPORT, HELD_AT TO GRAPH OlympicGraph",
            executable,
        )
        self.assertIn("RUN GLOBAL SCHEMA_CHANGE JOB bind_olympicgraph_types", executable)
        self.assertNotIn("DROP GRAPH", executable)
        self.assertNotIn("CREATE GRAPH", executable)
        self.assertNotRegex(executable, r"ADD VERTEX\s+\w+\s*\(")

    def test_loading_job_is_upsert_oriented(self) -> None:
        text = (GSQL_DIR / "01_loading_job.gsql").read_text(encoding="utf-8")
        self.assertIn("stable primary IDs", text)
        self.assertIn("load_olympic_vertices", text)
        self.assertIn("load_olympic_edges", text)

    def test_env_example_has_no_secrets(self) -> None:
        example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("TG_HOST", example)
        self.assertIn("TG_GRAPHNAME=OlympicGraph", example)
        blank = TigerGraphSettings(
            host="https://example.test",
            graphname="OlympicGraph",
            username="tigergraph",
            password="test-password",
            api_token="token",
            jwt_token="",
            secret="",
            restpp_port="443",
            gs_port="14240",
            ssl_port="443",
            tg_cloud=True,
            cert_path="",
        )
        redacted = blank.redacted()
        self.assertNotIn("password", redacted)
        self.assertNotIn("api_token", redacted)
        self.assertNotIn("secret", str(redacted))
        self.assertNotIn("token-value-should-not-appear", str(redacted))
        self.assertEqual(redacted["auth_method"], "api_token")

    def test_settings_validation(self) -> None:
        blank = TigerGraphSettings(
            host="",
            graphname="",
            username="",
            password="",
            api_token="",
            jwt_token="",
            secret="",
            restpp_port="9000",
            gs_port="14240",
            ssl_port="443",
            tg_cloud=False,
            cert_path="",
        )
        errors = blank.validate()
        self.assertTrue(errors)
        self.assertFalse(blank.configured)

    def test_gitignore_keeps_env_out_of_git(self) -> None:
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".env", text)
        self.assertIn("data/tigergraph/export/", text)


class SettingsIsolationTests(unittest.TestCase):
    def test_does_not_print_process_secrets(self) -> None:
        previous = {key: os.environ.get(key) for key in ("TG_PASSWORD", "TG_API_TOKEN")}
        os.environ["TG_PASSWORD"] = "test-password-value"
        os.environ["TG_API_TOKEN"] = "token-secret-value"
        try:
            settings = load_settings(Path("__missing__.env"))
            dumped = str(settings.redacted())
            self.assertNotIn("test-password-value", dumped)
            self.assertNotIn("token-secret-value", dumped)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
