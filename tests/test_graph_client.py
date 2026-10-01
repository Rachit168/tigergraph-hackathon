"""Client helpers that do not require a live TigerGraph instance."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from ingestion.graph_records import EDGE_TYPES, VERTEX_TYPES
from ingestion.paths import GSQL_DIR
from retrieval.graph.client import (
    GraphOperationError,
    PROBE_KIND,
    PROBE_VERTEX_ID,
    PROBE_VERTEX_TYPE,
    STALE_PROBE_IDS,
    TigerGraphClient,
    _is_absent_vertex_error,
    _probe_document_row,
    _probe_delete_predicate,
    _redact_secrets,
    _vertex_payload,
    exception_details,
    flatten_query_result,
    format_operation_error,
)
from retrieval.graph.params import lookup_params, params_for_spec, venue_date_params
from retrieval.structured.models import QuerySpec


class ClientHelperTests(unittest.TestCase):
    def test_flatten_query_result(self) -> None:
        raw = [{"event_ids": ["Q1"]}, {"count": 1, "pool_size": 4}]
        payload = flatten_query_result(raw)
        self.assertEqual(payload["event_ids"], ["Q1"])
        self.assertEqual(payload["count"], 1)

    def test_installed_query_names_from_rest_endpoints(self) -> None:
        from retrieval.graph.client import _query_names

        payload = {
            "GET /query/OlympicGraph/lookup_event": {},
            "GET /query/OlympicGraph/count_over_threshold": {},
        }
        names = set(_query_names(payload))
        self.assertEqual(names, {"lookup_event", "count_over_threshold"})
        self.assertEqual(_query_names(["lookup_event", "argmax_competitors"]), ["lookup_event", "argmax_competitors"])
        self.assertEqual(
            _query_names(["chunks_for_events", "event_neighborhood"]),
            ["chunks_for_events", "event_neighborhood"],
        )

    def test_lookup_params_fold_title(self) -> None:
        params = lookup_params("Judo at the 2016 Summer Olympics – Women's 57 kg")
        self.assertEqual(params["title"], "Judo at the 2016 Summer Olympics – Women's 57 kg")
        self.assertTrue(params["title_folded"])

    def test_venue_params_pad_tokens(self) -> None:
        spec = QuerySpec(
            qtype="multi_hop",
            operation="events_at_venue_date",
            raw_question="x",
            matched_template=True,
            venue="Laura Biathlon & Ski Complex",
            date_text="22 February 2014",
            year=2014,
        )
        params = venue_date_params(spec)
        self.assertEqual(params["year"], 2014)
        self.assertIn("|2014|", params["q_years"])
        self.assertIn("|2|", params["q_months"])
        self.assertIn("|22|", params["q_days"])
        self.assertEqual(params["tok8"], "")

    def test_params_for_spec_dispatch(self) -> None:
        spec = QuerySpec(
            qtype="aggregation",
            operation="count_over_threshold",
            raw_question="x",
            matched_template=True,
            sport="biathlon",
            year=2018,
            season="Winter",
            threshold=73,
        )
        name, params = params_for_spec(spec)
        self.assertEqual(name, "count_over_threshold")
        self.assertEqual(params["threshold"], 73)
        self.assertEqual(params["sport"], "biathlon")


class EnsureSchemaTests(unittest.TestCase):
    def _client(self) -> TigerGraphClient:
        client = TigerGraphClient.__new__(TigerGraphClient)
        client.available = True
        client.conn = object()
        client.error = ""
        return client

    def test_existing_complete_schema_skips_create(self) -> None:
        client = self._client()
        calls: list[str] = []
        client.verify_schema = lambda: {"ok": True, "missing_vertices": [], "missing_edges": []}
        client.run_gsql_file = lambda path: calls.append(str(path)) or "created"
        client.gsql = lambda command: calls.append(command) or "ok"
        result = client.ensure_schema(reset=False)
        self.assertEqual(result, "existing schema verified")
        self.assertEqual(calls, [])

    def test_ensure_schema_false_does_not_bind(self) -> None:
        client = self._client()
        client.bind_global_types = lambda: (_ for _ in ()).throw(AssertionError("bind_global_types should not run"))
        client.verify_schema = lambda: {"ok": True, "missing_vertices": [], "missing_edges": []}
        client.run_gsql_file = lambda path: (_ for _ in ()).throw(AssertionError(f"unexpected gsql {path}"))
        result = client.ensure_schema(reset=False)
        self.assertEqual(result, "existing schema verified")

    def test_incomplete_schema_does_not_recreate(self) -> None:
        client = self._client()
        calls: list[str] = []
        client.verify_schema = lambda: {"ok": False, "missing_vertices": ["Chunk"], "missing_edges": ["HELD_AT"]}
        client.run_gsql_file = lambda path: calls.append(str(path)) or "created"
        client.gsql = lambda command: calls.append(command) or "ok"
        with self.assertRaises(RuntimeError) as caught:
            client.ensure_schema(reset=False)
        self.assertEqual(calls, [])
        self.assertIn("Chunk", str(caught.exception))
        self.assertIn("HELD_AT", str(caught.exception))
        self.assertIn("Refusing to run 00_schema.gsql", str(caught.exception))

    def test_reset_true_still_drops_and_creates(self) -> None:
        client = self._client()
        calls: list[str] = []
        client.verify_schema = lambda: (_ for _ in ()).throw(AssertionError("verify_schema should not run on reset"))
        client.gsql = lambda command: calls.append("reset") or "dropped"
        client.run_gsql_file = lambda path: calls.append(Path(path).name) or "created"
        result = client.ensure_schema(reset=True)
        self.assertEqual(result, "created")
        self.assertEqual(calls, ["reset", "00_schema.gsql"])
        self.assertTrue((GSQL_DIR / "00_schema.gsql").exists())


class VerifySchemaTests(unittest.TestCase):
    def _client(self, conn) -> TigerGraphClient:
        client = TigerGraphClient.__new__(TigerGraphClient)
        client.available = True
        client.conn = conn
        client.error = ""
        return client

    def test_all_required_types_present(self) -> None:
        conn = _SchemaConn(list(VERTEX_TYPES), list(EDGE_TYPES))
        result = self._client(conn).verify_schema()
        self.assertEqual(result["missing_vertices"], [])
        self.assertEqual(result["missing_edges"], [])
        self.assertTrue(result["ok"])
        self.assertTrue(conn.forced)

    def test_missing_vertex_is_reported(self) -> None:
        vertices = [name for name in VERTEX_TYPES if name != "Chunk"]
        conn = _SchemaConn(vertices, list(EDGE_TYPES))
        result = self._client(conn).verify_schema()
        self.assertFalse(result["ok"])
        self.assertEqual(result["missing_vertices"], ["Chunk"])
        self.assertEqual(result["missing_edges"], [])

    def test_missing_edge_is_reported(self) -> None:
        edges = [name for name in EDGE_TYPES if name != "HELD_AT"]
        conn = _SchemaConn(list(VERTEX_TYPES), edges)
        result = self._client(conn).verify_schema()
        self.assertFalse(result["ok"])
        self.assertEqual(result["missing_vertices"], [])
        self.assertEqual(result["missing_edges"], ["HELD_AT"])

    def test_fallback_when_type_apis_unavailable(self) -> None:
        conn = _SchemaOnlyConn(
            {
                "VertexTypes": [{"Name": name} for name in VERTEX_TYPES],
                "EdgeTypes": [{"Name": name} for name in EDGE_TYPES],
            }
        )
        result = self._client(conn).verify_schema()
        self.assertTrue(result["ok"])
        self.assertEqual(result["missing_vertices"], [])
        self.assertEqual(result["missing_edges"], [])

    def test_fallback_when_type_apis_error(self) -> None:
        conn = _FailingTypesConn(
            {
                "VertexTypes": [{"Name": name} for name in VERTEX_TYPES],
                "EdgeTypes": [{"Name": name} for name in EDGE_TYPES],
            }
        )
        result = self._client(conn).verify_schema()
        self.assertTrue(result["ok"])

    def test_empty_schema_reports_all_missing(self) -> None:
        conn = _SchemaOnlyConn({})
        result = self._client(conn).verify_schema()
        self.assertFalse(result["ok"])
        self.assertEqual(result["missing_vertices"], list(VERTEX_TYPES))
        self.assertEqual(result["missing_edges"], list(EDGE_TYPES))

    def test_nested_wrapper_payload(self) -> None:
        conn = _SchemaConn(
            {"VertexTypes": list(VERTEX_TYPES)},
            {"EdgeTypes": [{"Name": name} for name in EDGE_TYPES]},
        )
        result = self._client(conn).verify_schema()
        self.assertTrue(result["ok"])

    def test_savanna_empty_graph_schema_uses_global_ls(self) -> None:
        conn = _SavannaEmptyGraphConn(_SAVANNA_GLOBAL_LS)
        result = self._client(conn).verify_schema()
        self.assertEqual(result["missing_vertices"], [])
        self.assertEqual(result["missing_edges"], [])
        self.assertTrue(result["ok"])

    def test_savanna_empty_graph_reports_missing_vertex(self) -> None:
        catalog = _SAVANNA_GLOBAL_LS.replace("- VERTEX Chunk(", "- VERTEX OtherChunk(")
        conn = _SavannaEmptyGraphConn(catalog)
        result = self._client(conn).verify_schema()
        self.assertFalse(result["ok"])
        self.assertEqual(result["missing_vertices"], ["Chunk"])


class UpsertDiagnosticsTests(unittest.TestCase):
    def _client(self, conn) -> TigerGraphClient:
        client = TigerGraphClient.__new__(TigerGraphClient)
        client.available = True
        client.conn = conn
        client.error = ""
        client.last_error = None
        client.settings = type("Settings", (), {"graphname": "OlympicGraph"})()
        return client

    def test_upsert_uses_id_attr_tuples(self) -> None:
        conn = _BoundUpsertConn()
        client = self._client(conn)
        count = client._call_upsert_vertices(
            "Document",
            [{"id": "Q1", "title": "Judo", "is_event": True}],
        )
        self.assertEqual(count, 1)
        vertex_type, vertices = conn.calls[0]
        self.assertEqual(vertex_type, "Document")
        self.assertEqual(vertices, [("Q1", {"title": "Judo", "is_event": True})])

    def test_unbound_graph_refuses_upsert_without_schema_change(self) -> None:
        conn = _SavannaEmptyGraphConn(_SAVANNA_GLOBAL_LS)
        client = self._client(conn)
        with self.assertRaises(GraphOperationError) as caught:
            client.upsert_export()
        text = str(caught.exception)
        self.assertIn("operation=require_bound_graph", text)
        self.assertIn("REST-30200", text)
        self.assertIn("missing_vertices", text)
        self.assertIn("Refusing to ALTER GRAPH", text)
        self.assertFalse(conn.upsert_called)

    def test_upsert_rest_error_keeps_operation_details(self) -> None:
        conn = _BoundUpsertConn(error=_FakeTigerGraphError())
        client = self._client(conn)
        with self.assertRaises(GraphOperationError) as caught:
            client._call_upsert_vertices("Document", [{"id": "Q1", "title": "x"}])
        details = caught.exception.details
        self.assertEqual(details["operation"], "upsert_vertices:Document")
        self.assertEqual(details["method"], "upsertVertices")
        self.assertEqual(details["exception_class"], "TigerGraphException")
        self.assertEqual(details["code"], "REST-30200")
        self.assertIn("Document", details["message"])
        self.assertEqual(client.last_error["code"], "REST-30200")

    def test_partial_vertex_batch_is_failure(self) -> None:
        conn = _BoundUpsertConn()
        conn.upsertVertices = lambda *args, **kwargs: 0
        client = self._client(conn)
        with self.assertRaises(GraphOperationError) as caught:
            client._upsert_vertices("Document", [{"id": "Q1", "title": "x"}])
        self.assertIn("partial batch", str(caught.exception))
        self.assertEqual(caught.exception.details["code"], "partial_upsert")

    def test_partial_edge_batch_is_failure(self) -> None:
        conn = _BoundUpsertConn()
        conn.upsertEdges = lambda *args, **kwargs: 0
        client = self._client(conn)
        with self.assertRaises(GraphOperationError) as caught:
            client._upsert_edges(
                "Document",
                "DESCRIBES",
                "Event",
                [{"src": "Q1", "tgt": "Q1"}],
                ["src", "tgt"],
            )
        self.assertIn("partial batch", str(caught.exception))
        self.assertEqual(caught.exception.details["code"], "partial_upsert")

    def test_numeric_string_attributes_stay_strings(self) -> None:
        payload = _vertex_payload(
            {
                "id": "Q1",
                "competitors_raw": "42",
                "nations_raw": "8",
                "prev_year_raw": "2012",
                "date_years": "2016",
                "competitors": "42",
                "has_competitors": "true",
                "year": "2016",
            }
        )
        self.assertEqual(payload["competitors_raw"], "42")
        self.assertEqual(payload["nations_raw"], "8")
        self.assertEqual(payload["prev_year_raw"], "2012")
        self.assertEqual(payload["date_years"], "2016")
        self.assertEqual(payload["competitors"], 42)
        self.assertIs(payload["has_competitors"], True)
        self.assertEqual(payload["year"], 2016)

    def test_error_formatter_redacts_secrets(self) -> None:
        class Boom(Exception):
            def __init__(self) -> None:
                super().__init__("auth failed")
                self.message = "password=test-password token=test-token Authorization: Bearer test-token https://example.test/graph"
                self.code = "401"

        details = exception_details(Boom(), operation="connect", method="echo")
        text = format_operation_error(details)
        self.assertNotIn("test-password", text)
        self.assertNotIn("abcd", text)
        self.assertNotIn("https://", text)
        self.assertIn("[REDACTED]", text)
        self.assertIn("[REDACTED_URL]", text)


class BindGlobalTypesTests(unittest.TestCase):
    def _client(self, conn) -> TigerGraphClient:
        client = TigerGraphClient.__new__(TigerGraphClient)
        client.available = True
        client.conn = conn
        client.error = ""
        client.last_error = None
        client.settings = type("Settings", (), {"graphname": "OlympicGraph"})()
        return client

    def test_bind_succeeds_when_catalog_exists_and_rest_is_empty(self) -> None:
        conn = _BindableSavannaConn(_SAVANNA_GLOBAL_LS)
        result = self._client(conn).bind_global_types()
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "bound")
        self.assertTrue(result["attempted"])
        self.assertEqual(result["file"], "03_bind_graph.gsql")
        self.assertEqual(result["vertices"], list(VERTEX_TYPES))
        self.assertEqual(result["edges"], list(EDGE_TYPES))
        self.assertEqual(result["missing_vertices"], [])
        self.assertEqual(result["missing_edges"], [])
        self.assertTrue(conn.bound)
        self.assertFalse(conn.reset_called)
        bind_commands = [command for command in conn.gsql_commands if "ADD VERTEX" in command]
        self.assertEqual(len(bind_commands), 1)
        self.assertIn("TO GRAPH OlympicGraph", bind_commands[0])
        self.assertNotRegex(bind_commands[0], r"(?im)^\s*DROP GRAPH")
        self.assertNotIn("00_schema.gsql", "\n".join(line for line in bind_commands[0].splitlines() if not line.strip().startswith("//")))
        self.assertNotIn("99_reset.gsql", bind_commands[0])

    def test_bind_refuses_when_global_vertex_missing(self) -> None:
        catalog = _SAVANNA_GLOBAL_LS.replace("- VERTEX Chunk(", "- VERTEX OtherChunk(")
        conn = _BindableSavannaConn(catalog)
        with self.assertRaises(GraphOperationError) as caught:
            self._client(conn).bind_global_types()
        self.assertIn("missing_vertices=['Chunk']", str(caught.exception))
        self.assertFalse(conn.bound)
        self.assertFalse(any("CREATE GLOBAL SCHEMA_CHANGE" in command for command in conn.gsql_commands))

    def test_bind_refuses_when_global_edge_missing(self) -> None:
        catalog = _SAVANNA_GLOBAL_LS.replace("DIRECTED EDGE CONTAINS_CHUNK(", "DIRECTED EDGE OTHER_CHUNK(")
        conn = _BindableSavannaConn(catalog)
        with self.assertRaises(GraphOperationError) as caught:
            self._client(conn).bind_global_types()
        self.assertIn("missing_edges=['CONTAINS_CHUNK']", str(caught.exception))
        self.assertFalse(conn.bound)
        self.assertFalse(any("CREATE GLOBAL SCHEMA_CHANGE" in command for command in conn.gsql_commands))

    def test_bind_is_noop_when_rest_already_complete(self) -> None:
        conn = _BoundUpsertConn()
        result = self._client(conn).bind_global_types()
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "already_bound")
        self.assertFalse(result["attempted"])
        self.assertFalse(getattr(conn, "gsql_commands", []))

    def test_bind_never_calls_reset(self) -> None:
        conn = _BindableSavannaConn(_SAVANNA_GLOBAL_LS)
        self._client(conn).bind_global_types()
        joined = "\n".join(conn.gsql_commands)
        self.assertNotRegex(joined, r"(?im)^\s*DROP GRAPH")
        self.assertNotIn("99_reset.gsql", joined)
        self.assertFalse(conn.reset_called)


class BindCliTests(unittest.TestCase):
    def test_bind_flag_does_not_invoke_ingest_or_reset(self) -> None:
        import io
        import tempfile
        from unittest.mock import patch

        import scripts.ingest_tigergraph as ingest

        bind_report = {
            "connection": {"ok": True},
            "bind": {"ok": True, "status": "bound", "attempted": True, "file": "03_bind_graph.gsql"},
            "schema": {"ok": True},
            "load": {"ok": False},
            "blockers": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "verify.json"
            with (
                patch.object(ingest, "run_bind", return_value=bind_report) as run_bind,
                patch.object(ingest, "run_ingest") as run_ingest,
                patch("sys.argv", ["ingest_tigergraph", "--bind", "--verify-out", str(out)]),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                code = ingest.main()
                run_bind.assert_called_once_with()
                run_ingest.assert_not_called()
            written = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertTrue(written["bind"]["ok"])
        self.assertFalse(written["load"]["ok"])

    def test_bind_and_reset_are_rejected(self) -> None:
        from unittest.mock import patch

        import scripts.ingest_tigergraph as ingest

        with patch("sys.argv", ["ingest_tigergraph", "--bind", "--reset"]):
            with self.assertRaises(SystemExit):
                ingest.main()


class ProbeWriteTests(unittest.TestCase):
    def _client(self, conn) -> TigerGraphClient:
        client = TigerGraphClient.__new__(TigerGraphClient)
        client.available = True
        client.conn = conn
        client.error = ""
        client.last_error = None
        client.settings = type("Settings", (), {"graphname": "OlympicGraph"})()
        return client

    def test_probe_id_is_deterministic_and_not_a_corpus_qid(self) -> None:
        self.assertEqual(PROBE_VERTEX_TYPE, "Document")
        self.assertEqual(PROBE_VERTEX_ID, "TGProbeDocument20260915")
        self.assertEqual(PROBE_KIND, "probe")
        self.assertTrue(PROBE_VERTEX_ID.isalnum())
        self.assertFalse(PROBE_VERTEX_ID.startswith("Q"))
        self.assertIn("__tg_write_probe_document__", STALE_PROBE_IDS)
        self.assertIn(PROBE_VERTEX_ID, STALE_PROBE_IDS)
        row = _probe_document_row()
        self.assertEqual(row["id"], PROBE_VERTEX_ID)
        self.assertEqual(row["kind"], "probe")
        self.assertFalse(row["is_event"])

    def test_require_bound_graph_runs_before_write(self) -> None:
        conn = _SavannaEmptyGraphConn(_SAVANNA_GLOBAL_LS)
        result = self._client(conn).probe_write()
        self.assertFalse(result["write_ok"])
        self.assertFalse(result["read_ok"])
        self.assertFalse(conn.upsert_called)
        self.assertIn("require_bound_graph", result["error"])

    def test_writes_exactly_one_synthetic_vertex_then_cleans_up(self) -> None:
        conn = _ProbeConn()
        result = self._client(conn).probe_write()
        self.assertTrue(result["write_ok"])
        self.assertTrue(result["read_ok"])
        self.assertTrue(result["delete_attempted"])
        self.assertTrue(result["delete_ok"])
        self.assertTrue(result["cleaned_up"])
        self.assertEqual(result["probe_id"], PROBE_VERTEX_ID)
        self.assertEqual(result["delete_methods"], ["gsql_delete_from"])
        self.assertEqual(conn.upsert_calls, 1)
        self.assertEqual(len(conn.upserted), 1)
        vertex_type, vertices = conn.upserted[0]
        self.assertEqual(vertex_type, "Document")
        self.assertEqual(len(vertices), 1)
        self.assertEqual(vertices[0][0], PROBE_VERTEX_ID)
        self.assertNotIn(PROBE_VERTEX_ID, conn.store)
        self.assertEqual(conn.by_id_calls, 0)
        self.assertEqual(conn.export_reads, 0)
        predicate = conn.gsql_commands[0]
        self.assertIn("DELETE FROM Document:s WHERE", predicate)
        self.assertIn('s.kind == "probe"', predicate)
        self.assertIn(f's.id == "{PROBE_VERTEX_ID}"', predicate)
        self.assertIn('s.id == "__tg_write_probe_document__"', predicate)

    def test_skips_cleanup_when_requested(self) -> None:
        conn = _ProbeConn()
        result = self._client(conn).probe_write(cleanup=False)
        self.assertTrue(result["write_ok"])
        self.assertTrue(result["read_ok"])
        self.assertFalse(result["delete_attempted"])
        self.assertIn(PROBE_VERTEX_ID, conn.store)

    def test_write_failure_is_returned_without_raising(self) -> None:
        conn = _ProbeConn(write_error=RuntimeError("REST failed"))
        result = self._client(conn).probe_write()
        self.assertFalse(result["write_ok"])
        self.assertFalse(result["read_ok"])
        self.assertFalse(result["delete_attempted"])
        self.assertIn("upsertVertices", result["error"])
        self.assertEqual(conn.store, {})

    def test_path_delete_601_is_not_treated_as_success_until_vertex_is_gone(self) -> None:
        conn = _ProbeConn(fail_gsql=True, fail_filter=True, fail_by_id=True)
        result = self._client(conn).probe_write()
        self.assertTrue(result["write_ok"])
        self.assertTrue(result["read_ok"])
        self.assertFalse(result["delete_ok"])
        self.assertFalse(result["cleaned_up"])
        self.assertIn(PROBE_VERTEX_ID, conn.store)
        self.assertIn("601", result["error"])

    def test_cleans_up_via_filter_when_path_delete_returns_601(self) -> None:
        conn = _ProbeConn(fail_gsql=True, fail_by_id=True)
        result = self._client(conn).probe_write()
        self.assertTrue(result["write_ok"])
        self.assertTrue(result["read_ok"])
        self.assertTrue(result["delete_ok"])
        self.assertTrue(result["cleaned_up"])
        self.assertIn("delVertices_filter", result["delete_methods"])
        self.assertNotIn("delVerticesById", result["delete_methods"])
        self.assertNotIn(PROBE_VERTEX_ID, conn.store)
        self.assertTrue(conn.filter_used_permanent_false)

    def test_get_601_after_delete_counts_as_cleaned_up(self) -> None:
        conn = _ProbeConn(absent_raises_601=True)
        result = self._client(conn).probe_write()
        self.assertTrue(result["delete_ok"])
        self.assertTrue(result["cleaned_up"])
        self.assertNotIn(PROBE_VERTEX_ID, conn.store)

    def test_stale_probe_ids_are_deleted_without_touching_corpus_rows(self) -> None:
        conn = _ProbeConn()
        conn.store["__tg_write_probe_document__"] = {"kind": "probe", "title": "stale"}
        conn.store["Q123"] = {"kind": "event", "title": "real event"}
        conn.store["Q456"] = {"kind": "document", "title": "real document"}
        result = self._client(conn).probe_write()
        self.assertTrue(result["cleaned_up"])
        self.assertEqual(result["stale_remaining"], [])
        self.assertNotIn(PROBE_VERTEX_ID, conn.store)
        self.assertNotIn("__tg_write_probe_document__", conn.store)
        self.assertIn("Q123", conn.store)
        self.assertIn("Q456", conn.store)

    def test_absent_vertex_error_detects_restpp_601(self) -> None:
        self.assertTrue(_is_absent_vertex_error(_FakeInvalidVertexId("TGProbeDocument20260915")))
        self.assertFalse(_is_absent_vertex_error(_FakeTigerGraphError()))
        predicate = _probe_delete_predicate(["TGProbeDocument20260915", "__tg_write_probe_document__"])
        self.assertTrue(predicate.startswith('s.kind == "probe"'))
        self.assertIn('s.id == "TGProbeDocument20260915"', predicate)
        self.assertIn('s.id == "__tg_write_probe_document__"', predicate)


class ProbeCliTests(unittest.TestCase):
    def test_probe_write_flag_does_not_invoke_ingest(self) -> None:
        import io
        import tempfile
        from unittest.mock import patch

        import scripts.ingest_tigergraph as ingest

        probe_report = {
            "connection": {"ok": True},
            "probe": {
                "ok": True,
                "operation": "probe_write",
                "vertex_type": "Document",
                "probe_id": PROBE_VERTEX_ID,
                "write_ok": True,
                "read_ok": True,
                "delete_attempted": True,
                "delete_ok": True,
                "cleaned_up": True,
            },
            "load": {"ok": False},
            "blockers": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "verify.json"
            with (
                patch.object(ingest, "run_probe_write", return_value=probe_report) as run_probe,
                patch.object(ingest, "run_ingest") as run_ingest,
                patch.object(ingest, "run_bind") as run_bind,
                patch("sys.argv", ["ingest_tigergraph", "--probe-write", "--verify-out", str(out)]),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                code = ingest.main()
                run_probe.assert_called_once_with()
                run_ingest.assert_not_called()
                run_bind.assert_not_called()
            written = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertTrue(written["probe"]["ok"])
        self.assertFalse(written["load"]["ok"])

    def test_probe_write_and_reset_are_rejected(self) -> None:
        from unittest.mock import patch

        import scripts.ingest_tigergraph as ingest

        with patch("sys.argv", ["ingest_tigergraph", "--probe-write", "--reset"]):
            with self.assertRaises(SystemExit):
                ingest.main()


_SAVANNA_GLOBAL_LS = """---- Global vertices, edges, and all graphs
Vertex Types:
  - VERTEX Document(PRIMARY_ID id STRING, title STRING) WITH STATS="OUTDEGREE_BY_EDGETYPE"
  - VERTEX Chunk(PRIMARY_ID id STRING, document_id STRING) WITH STATS="OUTDEGREE_BY_EDGETYPE"
  - VERTEX Event(PRIMARY_ID id STRING, document_id STRING) WITH STATS="OUTDEGREE_BY_EDGETYPE"
  - VERTEX Games(PRIMARY_ID id STRING, year INT) WITH STATS="OUTDEGREE_BY_EDGETYPE"
  - VERTEX Sport(PRIMARY_ID id STRING, name STRING) WITH STATS="OUTDEGREE_BY_EDGETYPE"
  - VERTEX Venue(PRIMARY_ID id STRING, name STRING) WITH STATS="OUTDEGREE_BY_EDGETYPE"
Edge Types:
  - DIRECTED EDGE CONTAINS_CHUNK(FROM Document, TO Chunk, document_id STRING)
  - DIRECTED EDGE DESCRIBES(FROM Document, TO Event, document_id STRING)
  - DIRECTED EDGE IN_GAMES(FROM Event, TO Games, document_id STRING)
  - DIRECTED EDGE OF_SPORT(FROM Event, TO Sport, document_id STRING)
  - DIRECTED EDGE HELD_AT(FROM Event, TO Venue, document_id STRING)

Graphs:
  - Graph OlympicGraph()
Jobs:
"""


class _SchemaConn:
    def __init__(self, vertices, edges) -> None:
        self.vertices = vertices
        self.edges = edges
        self.forced = False

    def getVertexTypes(self, force: bool = False):
        self.forced = self.forced or force
        return self.vertices

    def getEdgeTypes(self, force: bool = False):
        self.forced = self.forced or force
        return self.edges


class _SchemaOnlyConn:
    def __init__(self, schema: dict) -> None:
        self._schema = schema

    def getSchema(self) -> dict:
        return self._schema


class _FailingTypesConn(_SchemaOnlyConn):
    def getVertexTypes(self, force: bool = False):
        raise RuntimeError("unavailable")

    def getEdgeTypes(self, force: bool = False):
        raise RuntimeError("unavailable")


class _SavannaEmptyGraphConn:
    """Live Savanna shape: graph-scoped REST schema is empty; types exist globally."""

    def __init__(self, catalog: str) -> None:
        self.catalog = catalog
        self.upsert_called = False
        self.gsql_commands: list[str] = []

    def getVertexTypes(self, force: bool = False) -> list:
        return []

    def getEdgeTypes(self, force: bool = False) -> list:
        return []

    def getSchema(self, udts: bool = True, force: bool = False) -> dict:
        return {"GraphName": "OlympicGraph", "VertexTypes": [], "EdgeTypes": [], "UDTs": []}

    def gsql(self, command: str) -> str:
        self.gsql_commands.append(command)
        return self.catalog

    def upsertVertices(self, vertex_type: str, vertices: list) -> int:
        self.upsert_called = True
        raise AssertionError("upsertVertices must not run against an unbound graph")


class _FakeTigerGraphError(Exception):
    def __init__(self) -> None:
        super().__init__("The input vertex type: 'Document' is not a valid vertex type.")
        self.message = "The input vertex type: 'Document' is not a valid vertex type."
        self.code = "REST-30200"


_FakeTigerGraphError.__name__ = "TigerGraphException"


class _BoundUpsertConn:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple] = []
        self.error = error
        self.upsert_called = False

    def getVertexTypes(self, force: bool = False) -> list[str]:
        return list(VERTEX_TYPES)

    def getEdgeTypes(self, force: bool = False) -> list[str]:
        return list(EDGE_TYPES)

    def getSchema(self, udts: bool = True, force: bool = False) -> dict:
        return {
            "GraphName": "OlympicGraph",
            "VertexTypes": [{"Name": name} for name in VERTEX_TYPES],
            "EdgeTypes": [{"Name": name} for name in EDGE_TYPES],
            "UDTs": [],
        }

    def upsertVertices(self, vertex_type: str, vertices: list) -> int:
        self.upsert_called = True
        self.calls.append((vertex_type, vertices))
        if self.error is not None:
            raise self.error
        return len(vertices)


class _BindableSavannaConn(_SavannaEmptyGraphConn):
    def __init__(self, catalog: str) -> None:
        super().__init__(catalog)
        self.bound = False
        self.reset_called = False

    def getVertexTypes(self, force: bool = False) -> list:
        return list(VERTEX_TYPES) if self.bound else []

    def getEdgeTypes(self, force: bool = False) -> list:
        return list(EDGE_TYPES) if self.bound else []

    def getSchema(self, udts: bool = True, force: bool = False) -> dict:
        if not self.bound:
            return {"GraphName": "OlympicGraph", "VertexTypes": [], "EdgeTypes": [], "UDTs": []}
        return {
            "GraphName": "OlympicGraph",
            "VertexTypes": [{"Name": name} for name in VERTEX_TYPES],
            "EdgeTypes": [{"Name": name} for name in EDGE_TYPES],
            "UDTs": [],
        }

    def gsql(self, command: str) -> str:
        self.gsql_commands.append(command)
        text = command.casefold()
        if re.search(r"(?im)^\s*drop\s+graph\b", command) or "99_reset.gsql" in text:
            self.reset_called = True
        if "add vertex" in text and "to graph olympicgraph" in text:
            self.bound = True
            return "Graph OlympicGraph updated"
        return self.catalog


class _FakeInvalidVertexId(Exception):
    def __init__(self, vertex_id: str) -> None:
        self.message = (
            f"The input vertex id '{vertex_id}' is not a valid vertex id for vertex type = Document."
        )
        self.code = "601"
        super().__init__(self.message)


_FakeInvalidVertexId.__name__ = "TigerGraphException"


class _ProbeConn(_BoundUpsertConn):
    def __init__(
        self,
        write_error: Exception | None = None,
        *,
        fail_gsql: bool = False,
        fail_filter: bool = False,
        fail_by_id: bool = False,
        absent_raises_601: bool = True,
    ) -> None:
        super().__init__()
        self.write_error = write_error
        self.fail_gsql = fail_gsql
        self.fail_filter = fail_filter
        self.fail_by_id = fail_by_id
        self.absent_raises_601 = absent_raises_601
        self.store: dict[str, dict] = {}
        self.upsert_calls = 0
        self.upserted: list[tuple] = []
        self.export_reads = 0
        self.gsql_commands: list[str] = []
        self.filter_deletes: list[str] = []
        self.filter_used_permanent_false = False
        self.by_id_calls = 0
        self.delete_used_permanent_false = False

    def upsertVertices(self, vertex_type: str, vertices: list) -> int:
        self.upsert_calls += 1
        self.upserted.append((vertex_type, vertices))
        if self.write_error is not None:
            raise self.write_error
        for vertex_id, attrs in vertices:
            self.store[str(vertex_id)] = dict(attrs)
        return len(vertices)

    def getVerticesById(self, vertex_type: str, vertexIds):
        probe_id = str(vertexIds)
        if probe_id not in self.store:
            if self.absent_raises_601:
                raise _FakeInvalidVertexId(probe_id)
            return []
        return [{"v_id": probe_id, "v_type": vertex_type, "attributes": self.store[probe_id]}]

    def gsql(self, command: str) -> str:
        self.gsql_commands.append(command)
        if "DELETE FROM" in command or "DELETE(s)" in command or "DELETE (s)" in command:
            if self.fail_gsql:
                raise _FakeInvalidVertexId("gsql")
            self._apply_probe_delete(command)
            return "Successfully deleted vertices from graph 'OlympicGraph'"
        return ""

    def delVertices(self, vertex_type: str, where: str = "", permanent: bool = False, **kwargs) -> int:
        self.filter_deletes.append(where)
        self.filter_used_permanent_false = permanent is False
        if self.fail_filter:
            raise _FakeInvalidVertexId("filter")
        if not where:
            raise AssertionError("unfiltered Document delete is not allowed")
        deleted = 0
        for vertex_id, attrs in list(self.store.items()):
            if _probe_where_matches(where, vertex_id, attrs):
                del self.store[vertex_id]
                deleted += 1
        return deleted

    def delVerticesById(self, vertex_type: str, vertexIds, permanent: bool = False) -> int:
        self.by_id_calls += 1
        self.delete_used_permanent_false = permanent is False
        probe_id = str(vertexIds)
        if self.fail_by_id:
            raise _FakeInvalidVertexId(probe_id)
        if probe_id in self.store:
            del self.store[probe_id]
            return 1
        return 0

    def _apply_probe_delete(self, command: str) -> None:
        for vertex_id, attrs in list(self.store.items()):
            kind = str(attrs.get("kind", ""))
            if kind == "probe" or f's.id == "{vertex_id}"' in command:
                if 's.kind == "probe"' in command or f's.id == "{vertex_id}"' in command:
                    del self.store[vertex_id]


def _probe_where_matches(where: str, vertex_id: str, attrs: dict) -> bool:
    kind = str(attrs.get("kind", ""))
    if where in {f'kind="{kind}"', f'kind=="{kind}"'}:
        return True
    if where in {f'id="{vertex_id}"', f'id=="{vertex_id}"'}:
        return True
    return False


if __name__ == "__main__":
    unittest.main()
