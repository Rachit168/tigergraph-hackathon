"""Public verifier CLI: importable, read-only, no hidden data."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import scripts.verify_tigergraph as verify


class VerifyTigergraphCliTests(unittest.TestCase):
    def test_module_imports_without_missing_renderer(self) -> None:
        import inspect

        self.assertTrue(hasattr(verify, "main"))
        self.assertTrue(hasattr(verify, "run_verify"))
        source = inspect.getsource(verify)
        self.assertNotIn("render_tigergraph_report", source)
        self.assertNotIn("run_ingest", source)

    def test_help_reaches_argparse_without_credentials(self) -> None:
        with patch("sys.argv", ["verify_tigergraph", "--help"]):
            with self.assertRaises(SystemExit) as caught:
                verify.main()
        self.assertEqual(caught.exception.code, 0)

    def test_live_path_uses_run_verify_not_ingest(self) -> None:
        report = {
            "connection": {"ok": True, "error": "", "environment": "test"},
            "schema": {"ok": True, "attempted": True},
            "queries": {"ok": True, "attempted": True, "installed": ["lookup_event"]},
            "retrieval_queries": {"ok": True, "attempted": True},
            "read_only": True,
            "blockers": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "verify.json"
            with (
                patch.object(verify, "run_verify", return_value=report) as run_verify,
                patch("retrieval.graph.verify.run_ingest") as run_ingest,
                patch("sys.argv", ["verify_tigergraph", "--verify-out", str(out)]),
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                code = verify.main()
                run_verify.assert_called_once_with()
                run_ingest.assert_not_called()
            written = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertTrue(written["read_only"])
        self.assertTrue(written["schema"]["ok"])
        self.assertIn("schema_ok", stdout.getvalue())
        self.assertNotIn("sk-", stdout.getvalue())
        self.assertNotIn("eval_hidden", str(written))
        self.assertNotIn("holdout", str(written).casefold())

    def test_skip_live_does_not_connect_or_ingest(self) -> None:
        exported = {
            "export": {"counts": {"Document": 1, "Event": 1}},
            "graph": object(),
            "documents": [],
            "chunks": [],
        }
        local = {"equivalence": {"local_mismatches": 0}}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "verify.json"
            with (
                patch("retrieval.graph.verify.run_export", return_value=exported),
                patch("retrieval.graph.verify._local_verification", return_value=local),
                patch("retrieval.graph.verify.run_ingest") as run_ingest,
                patch.object(verify, "run_verify") as run_verify,
                patch("sys.argv", ["verify_tigergraph", "--skip-live", "--verify-out", str(out)]),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                code = verify.main()
                run_ingest.assert_not_called()
                run_verify.assert_not_called()
            written = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(written["connection"]["error"], "skip-live")
        self.assertTrue(written["read_only"])
        self.assertEqual(written["export"]["counts"]["Document"], 1)

    def test_unconfigured_live_verify_is_fail_closed_json(self) -> None:
        report = {
            "connection": {"ok": False, "error": "TigerGraph is not configured", "environment": "unconfigured"},
            "schema": {"ok": False, "attempted": False},
            "queries": {"ok": False, "attempted": False},
            "retrieval_queries": {"ok": False, "attempted": False},
            "read_only": True,
            "blockers": ["TigerGraph is not configured"],
            "settings": {"host": "(set)", "api_key_set": False},
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "verify.json"
            with (
                patch.object(verify, "run_verify", return_value=report),
                patch("sys.argv", ["verify_tigergraph", "--verify-out", str(out)]),
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                code = verify.main()
        self.assertEqual(code, 2)
        self.assertNotIn("password", stdout.getvalue().casefold())
        self.assertIn("TigerGraph is not configured", stdout.getvalue())


class RunVerifyReadOnlyTests(unittest.TestCase):
    def test_run_verify_does_not_mutate_or_install(self) -> None:
        from retrieval.graph.verify import run_verify

        client = Mock()
        client.error = ""
        client.environment = "test"
        client.connect.return_value = True
        client.verify_schema.return_value = {"ok": True, "missing_vertices": [], "missing_edges": []}
        client.verify_queries.return_value = {"ok": True, "installed": ["lookup_event"], "missing": []}
        client.verify_retrieval_queries.return_value = {
            "ok": True,
            "installed": ["lookup_event"],
            "missing": [],
        }
        settings = Mock()
        settings.redacted.return_value = {"host": "(set)"}
        with (
            patch("retrieval.graph.verify.load_settings", return_value=settings),
            patch("retrieval.graph.verify.TigerGraphClient", return_value=client),
        ):
            report = run_verify()
        self.assertTrue(report["read_only"])
        self.assertTrue(report["connection"]["ok"])
        self.assertTrue(report["schema"]["ok"])
        self.assertTrue(report["queries"]["ok"])
        self.assertTrue(report["retrieval_queries"]["ok"])
        client.ensure_schema.assert_not_called()
        client.upsert_export.assert_not_called()
        client.install_queries.assert_not_called()
        self.assertFalse(report["load"]["attempted"])


    def test_verify_diagnostics_are_redacted_in_report_and_health(self) -> None:
        from retrieval.graph.client import GraphOperationError
        from retrieval.graph.verify import run_verify
        from ui.runtime import health_payload

        message = (
            'https://private.example.invalid/api?token=URL_VALUE '
            'api_key=KEY_VALUE token=TOKEN_VALUE Authorization: Basic AUTH_VALUE; '
            'password=PASSWORD_VALUE secret="SECRET VALUE"; '
            'path="D:\\private folder\\config.txt" /srv/private/config.txt'
        )
        graph_error = GraphOperationError({
            "exception_class": "BackendError", "operation": "schema",
            "method": "verify_schema", "message": message, "response": message,
        })
        cases = (
            ("connection_error", None),
            ("connect", RuntimeError(message)),
            ("verify_schema", graph_error),
            ("verify_schema", ValueError(message)),
            ("verify_queries", RuntimeError(message)),
            ("verify_retrieval_queries", RuntimeError(message)),
        )
        for stage, error in cases:
            with self.subTest(stage=stage, error_class=type(error).__name__):
                client = Mock()
                client.error = message if stage == "connection_error" else ""
                client.environment = "test"
                client.connect.return_value = stage != "connection_error"
                client.verify_schema.return_value = {"ok": True}
                client.verify_queries.return_value = {"ok": True}
                client.verify_retrieval_queries.return_value = {"ok": True}
                if error:
                    getattr(client, stage).side_effect = error
                settings = Mock(graphname="OlympicGraph", configured=False)
                settings.redacted.return_value = {"host": "(set)"}
                with (
                    patch("retrieval.graph.verify.load_settings", return_value=settings),
                    patch("retrieval.graph.verify.TigerGraphClient", return_value=client),
                ):
                    report = run_verify()
                self.assertTrue(report["read_only"])
                self.assertFalse(report["vector"]["ok"])
                self.assertEqual(report["vector"]["status"], "UNAVAILABLE")
                self.assertTrue(report["blockers"])
                if isinstance(error, GraphOperationError):
                    self.assertIn("BackendError", report["blockers"][0])
                    self.assertIn("operation=schema", report["blockers"][0])
                elif error:
                    self.assertIn(type(error).__name__, report["blockers"][0])
                with (
                    patch("ui.runtime.get_runtime", return_value=Mock(capabilities=lambda: {})),
                    patch("ui.runtime.run_verify", return_value=report),
                    patch("ui.runtime.load_settings", return_value=settings),
                ):
                    health = health_payload()
                for payload in (report, health):
                    blob = json.dumps(payload)
                    for value in ("private.example.invalid", "URL_VALUE", "KEY_VALUE", "TOKEN_VALUE",
                                  "AUTH_VALUE", "PASSWORD_VALUE", "SECRET VALUE", "private folder",
                                  "/srv/private/config.txt"):
                        self.assertNotIn(value, blob)
                for name in ("ensure_schema", "install_queries", "upsert_export", "probe_write"):
                    getattr(client, name).assert_not_called()

    def test_diagnostic_redaction_handles_quoted_credentials_and_local_paths(self) -> None:
        from retrieval.graph.verify import _redact_diagnostic

        cases = (
            ('"api_key": "KEY VALUE"', "KEY VALUE"),
            ("TG_API_TOKEN=TOKEN_VALUE", "TOKEN_VALUE"),
            ("Authorization: Bearer AUTH_VALUE;", "AUTH_VALUE"),
            ("Authorization: [REDACTED] AUTH_VALUE;", "AUTH_VALUE"),
            ('"Authorization": "Basic AUTH VALUE"', "AUTH VALUE"),
            ("password='PASSWORD VALUE'", "PASSWORD VALUE"),
            ("secret=SECRET_VALUE", "SECRET_VALUE"),
            (r"D:\private\config.txt", r"D:\private\config.txt"),
            (r'"D:\private folder\config.txt"', "private folder"),
            (r"\\server\private\config.txt", "server"),
            ("/srv/private/config.txt", "/srv/private/config.txt"),
            ("https://private.example.invalid/api", "private.example.invalid"),
        )
        for message, sensitive in cases:
            with self.subTest(message=message):
                self.assertNotIn(sensitive, _redact_diagnostic(message))
        self.assertEqual(_redact_diagnostic("query missing; index not ready"), "query missing; index not ready")


if __name__ == "__main__":
    unittest.main()
