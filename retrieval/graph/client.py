"""Small pyTigerGraph client. Credentials stay in settings; nothing secret is printed."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from config.settings import TigerGraphSettings, load_settings
from ingestion.graph_export import DOCUMENT_FIELDS, EDGE_FIELD_MAP
from ingestion.graph_records import EDGE_TYPES, INSTALLED_QUERIES, VERTEX_TYPES
from ingestion.graph_tsv import read_jsonl, read_tsv
from ingestion.paths import DEFAULT_GRAPH_EXPORT_DIR, GSQL_DIR
from retrieval.graph.contract import interpret_installed_result
from retrieval.graph.params import params_for_spec
from retrieval.graph.results import GraphQueryResult
from retrieval.structured.models import QuerySpec

UPSERT_BATCH = 400
CHUNK_UPSERT_BATCH = 80
PROBE_VERTEX_TYPE = "Document"
PROBE_VERTEX_ID = "TGProbeDocument20260915"
PROBE_KIND = "probe"
# Previous failed probes may have left these synthetic Document IDs behind.
STALE_PROBE_IDS: tuple[str, ...] = (
    "__tg_write_probe_document__",
    "TGProbeDocument20260915",
)
_LOG = logging.getLogger("retrieval.graph.client")
_SECRET_RE = re.compile(
    r"(?i)(password|passwd|secret|token|api[_-]?token|jwt|authorization|bearer|gsqlsecret)(['\"]?\s*[:=]\s*)([^\s,;]+)"
)


class TigerGraphClient:
    def __init__(self, settings: TigerGraphSettings | None = None) -> None:
        self.settings = settings or load_settings()
        self.conn = None
        self.available = False
        self.error = ""
        self.environment = "unconfigured"
        self.last_error: dict[str, Any] | None = None

    def connect(self) -> bool:
        errors = self.settings.validate()
        if errors:
            self.available = False
            self.error = "; ".join(errors)
            self.environment = "unconfigured"
            return False
        try:
            from pyTigerGraph import TigerGraphConnection
        except ImportError as exc:
            self.available = False
            self.error = f"pyTigerGraph is not installed: {exc.__class__.__name__}"
            self.environment = "missing_sdk"
            return False
        kwargs: dict[str, Any] = {
            "host": self.settings.host,
            "graphname": self.settings.graphname,
            "tgCloud": self.settings.tg_cloud,
            "restppPort": self.settings.restpp_port,
            "gsPort": self.settings.gs_port,
            "sslPort": self.settings.ssl_port,
        }
        if self.settings.username:
            kwargs["username"] = self.settings.username
        if self.settings.password:
            kwargs["password"] = self.settings.password
        if self.settings.api_token:
            kwargs["apiToken"] = self.settings.api_token
        if self.settings.jwt_token:
            kwargs["jwtToken"] = self.settings.jwt_token
        if self.settings.secret:
            kwargs["gsqlSecret"] = self.settings.secret
        if self.settings.cert_path:
            kwargs["certPath"] = self.settings.cert_path
        try:
            self.conn = TigerGraphConnection(**kwargs)
            if self.settings.secret and not self.settings.api_token:
                try:
                    self.conn.getToken(self.settings.secret)
                except Exception:
                    pass
            echo = None
            if hasattr(self.conn, "echo"):
                echo = self.conn.echo()
            elif hasattr(self.conn, "ping"):
                echo = self.conn.ping()
            else:
                echo = self.conn.gsql("ls")
            self.available = True
            self.environment = _classify_environment(self.settings)
            self.error = ""
            _ = echo
            return True
        except Exception as exc:
            self.available = False
            self.conn = None
            self.error = _safe_error(exc)
            self.environment = "unreachable"
            return False

    def gsql(self, command: str) -> str:
        self._require()
        return str(self.conn.gsql(command))

    def run_gsql_file(self, path: str | Path) -> str:
        text = Path(path).read_text(encoding="utf-8")
        return self.gsql(text)

    def schema(self) -> dict[str, Any]:
        self._require()
        if hasattr(self.conn, "getSchema"):
            return self.conn.getSchema()
        raw = self.gsql(f"USE GRAPH {self.settings.graphname}\nls")
        return {"raw": raw}

    def vertex_counts(self) -> dict[str, int]:
        self._require()
        counts: dict[str, int] = {}
        getter = getattr(self.conn, "getVertexCount", None)
        if getter is None:
            raise RuntimeError("getVertexCount is not available on this pyTigerGraph version")
        try:
            all_counts = getter("*")
            if isinstance(all_counts, dict):
                return {str(key): int(value) for key, value in all_counts.items()}
        except Exception:
            pass
        for vertex_type in VERTEX_TYPES:
            counts[vertex_type] = int(getter(vertex_type))
        return counts

    def edge_counts(self) -> dict[str, int]:
        self._require()
        counts: dict[str, int] = {}
        getter = getattr(self.conn, "getEdgeCount", None)
        if getter is None:
            raise RuntimeError("getEdgeCount is not available on this pyTigerGraph version")
        try:
            all_counts = getter("*")
            if isinstance(all_counts, dict):
                return {str(key): int(value) for key, value in all_counts.items()}
        except Exception:
            pass
        for edge_type in EDGE_TYPES:
            counts[edge_type] = int(getter(edge_type))
        return counts

    def installed_queries(self) -> list[str]:
        self._require()
        getter = getattr(self.conn, "getInstalledQueries", None)
        if getter is None:
            return []
        try:
            try:
                payload = getter(fmt="list")
            except TypeError:
                payload = getter()
            return sorted(set(_query_names(payload)))
        except Exception:
            return []

    def verify_schema(self) -> dict[str, Any]:
        self._require()
        vertex_names = self._graph_type_names("vertex")
        edge_names = self._graph_type_names("edge")
        missing_vertices = [name for name in VERTEX_TYPES if name not in vertex_names]
        missing_edges = [name for name in EDGE_TYPES if name not in edge_names]
        return {
            "ok": not missing_vertices and not missing_edges,
            "missing_vertices": missing_vertices,
            "missing_edges": missing_edges,
        }

    def _graph_type_names(self, kind: str) -> set[str]:
        method_name = "getVertexTypes" if kind == "vertex" else "getEdgeTypes"
        getter = getattr(self.conn, method_name, None)
        if getter is not None:
            try:
                try:
                    payload = getter(force=True)
                except TypeError:
                    payload = getter()
                names = _type_names(payload)
                if names:
                    return names
            except Exception:
                pass
        try:
            names = _type_names_from_schema(self.schema(), kind)
            if names:
                return names
        except Exception:
            pass
        return self._gsql_catalog_type_names(kind)

    def _gsql_catalog_type_names(self, kind: str) -> set[str]:
        catalogs: list[str] = []
        for command in ("ls", "USE GLOBAL\nls"):
            try:
                catalogs.append(str(self.gsql(command)))
            except Exception:
                continue
        names: set[str] = set()
        for catalog in catalogs:
            names |= _type_names_from_gsql(catalog, kind)
        return names

    def verify_queries(self) -> dict[str, Any]:
        installed = self.installed_queries()
        missing = [name for name in INSTALLED_QUERIES if name not in installed]
        return {"ok": not missing, "installed": installed, "missing": missing}

    def verify_retrieval_queries(self) -> dict[str, Any]:
        from retrieval.graphrag.models import RETRIEVAL_QUERIES

        installed = self.installed_queries()
        missing = [name for name in RETRIEVAL_QUERIES if name not in installed]
        return {"ok": not missing, "installed": installed, "missing": missing}

    def run_query(self, name: str, params: dict | None = None, timeout: int = 60) -> Any:
        self._require()
        runner = getattr(self.conn, "runInstalledQuery", None)
        if runner is None:
            raise RuntimeError("runInstalledQuery is not available on this pyTigerGraph version")
        try:
            return runner(name, params or {}, timeout=timeout, usePost=True)
        except TypeError:
            return runner(name, params or {})

    def execute_spec(self, spec: QuerySpec, events_by_id: dict | None = None) -> GraphQueryResult:
        started = time.perf_counter()
        name, params = params_for_spec(spec)
        raw = self.run_query(name, params)
        payload = flatten_query_result(raw)
        result = interpret_installed_result(name, spec, payload, events_by_id)
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        result.source = "tigergraph"
        return result

    def ensure_schema(self, reset: bool = False) -> str:
        self._require()
        if reset:
            reset_sql = (GSQL_DIR / "99_reset.gsql").read_text(encoding="utf-8")
            try:
                self.gsql(reset_sql)
            except Exception as exc:
                return _safe_error(exc)
            return self.run_gsql_file(GSQL_DIR / "00_schema.gsql")
        check = self.verify_schema()
        if check.get("ok"):
            return "existing schema verified"
        missing_vertices = check.get("missing_vertices") or []
        missing_edges = check.get("missing_edges") or []
        raise RuntimeError(
            "schema is missing or incomplete: "
            f"missing_vertices={missing_vertices}; missing_edges={missing_edges}. "
            "Refusing to run 00_schema.gsql; pass --reset only to drop and recreate."
        )

    def install_queries(self) -> str:
        created = str(self.run_gsql_file(GSQL_DIR / "02_queries.gsql"))
        lowered = created.casefold()
        if "saved as draft" in lowered or "type check error" in lowered or "syntax error" in lowered:
            raise GraphOperationError(
                {
                    "operation": "install_queries",
                    "method": "gsql",
                    "command": "02_queries.gsql",
                    "exception_class": "RuntimeError",
                    "message": "GSQL queries were saved as drafts with compile errors: " + created[:2000],
                    "code": "gsql_draft",
                    "status": None,
                    "response": created[:2000],
                }
            )
        extra = str(
            self.gsql(
                f"USE GRAPH {self.settings.graphname}\n"
                "INSTALL QUERY lookup_event, count_over_threshold, argmax_competitors, "
                "previous_event_gold, events_at_venue_date"
            )
        )
        return f"{created}\n{extra}"

    def install_retrieval_queries(self) -> str:
        created = str(self.run_gsql_file(GSQL_DIR / "04_retrieval.gsql"))
        lowered = created.casefold()
        if "saved as draft" in lowered or "type check error" in lowered or "syntax error" in lowered:
            raise GraphOperationError(
                {
                    "operation": "install_retrieval_queries",
                    "method": "gsql",
                    "command": "04_retrieval.gsql",
                    "exception_class": "RuntimeError",
                    "message": "GSQL retrieval queries were saved as drafts with compile errors: " + created[:2000],
                    "code": "gsql_draft",
                    "status": None,
                    "response": created[:2000],
                }
            )
        extra = str(
            self.gsql(
                f"USE GRAPH {self.settings.graphname}\n"
                "INSTALL QUERY chunks_for_events, event_neighborhood"
            )
        )
        return f"{created}\n{extra}"

    def wait_for_export_counts(
        self,
        expected: dict[str, int],
        attempts: int = 18,
        delay_s: float = 5.0,
    ) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
        counts = {"vertices": {}, "edges": {}}
        mismatches: dict[str, dict[str, int]] = {}
        for attempt in range(max(1, attempts)):
            counts = {
                "vertices": self.vertex_counts(),
                "edges": self.edge_counts(),
            }
            mismatches = _live_count_mismatches(expected, counts["vertices"], counts["edges"])
            if not mismatches:
                return counts, mismatches
            if attempt + 1 < attempts and delay_s > 0:
                time.sleep(delay_s)
        return counts, mismatches

    def upsert_export(self, export_dir: str | Path | None = None) -> dict[str, int]:
        self._require()
        self.require_bound_graph()
        directory = Path(export_dir) if export_dir is not None else DEFAULT_GRAPH_EXPORT_DIR
        loaded: dict[str, int] = {}
        loaded["Document"] = self._upsert_vertices("Document", read_tsv(directory / "vertices_document.tsv"))
        loaded["Event"] = self._upsert_vertices("Event", read_tsv(directory / "vertices_event.tsv"))
        loaded["Games"] = self._upsert_vertices("Games", read_tsv(directory / "vertices_games.tsv"))
        loaded["Sport"] = self._upsert_vertices("Sport", read_tsv(directory / "vertices_sport.tsv"))
        loaded["Venue"] = self._upsert_vertices("Venue", read_tsv(directory / "vertices_venue.tsv"))
        loaded["Chunk"] = self._upsert_vertices("Chunk", read_jsonl(directory / "vertices_chunk.jsonl"))
        loaded["CONTAINS_CHUNK"] = self._upsert_edges(
            "Document", "CONTAINS_CHUNK", "Chunk", read_tsv(directory / "edges_contains_chunk.tsv"), EDGE_FIELD_MAP["CONTAINS_CHUNK"]
        )
        loaded["DESCRIBES"] = self._upsert_edges(
            "Document", "DESCRIBES", "Event", read_tsv(directory / "edges_describes.tsv"), EDGE_FIELD_MAP["DESCRIBES"]
        )
        loaded["IN_GAMES"] = self._upsert_edges(
            "Event", "IN_GAMES", "Games", read_tsv(directory / "edges_in_games.tsv"), EDGE_FIELD_MAP["IN_GAMES"]
        )
        loaded["OF_SPORT"] = self._upsert_edges(
            "Event", "OF_SPORT", "Sport", read_tsv(directory / "edges_of_sport.tsv"), EDGE_FIELD_MAP["OF_SPORT"]
        )
        loaded["HELD_AT"] = self._upsert_edges(
            "Event", "HELD_AT", "Venue", read_tsv(directory / "edges_held_at.tsv"), EDGE_FIELD_MAP["HELD_AT"]
        )
        return loaded

    def rest_type_names(self) -> tuple[set[str], set[str]]:
        vertices: set[str] = set()
        edges: set[str] = set()
        getter_v = getattr(self.conn, "getVertexTypes", None)
        getter_e = getattr(self.conn, "getEdgeTypes", None)
        if getter_v is not None:
            try:
                try:
                    vertices = _type_names(getter_v(force=True))
                except TypeError:
                    vertices = _type_names(getter_v())
            except Exception:
                vertices = set()
        if getter_e is not None:
            try:
                try:
                    edges = _type_names(getter_e(force=True))
                except TypeError:
                    edges = _type_names(getter_e())
            except Exception:
                edges = set()
        return vertices, edges

    def require_bound_graph(self) -> None:
        rest_vertices, rest_edges = self.rest_type_names()
        missing_vertices = [name for name in VERTEX_TYPES if name not in rest_vertices]
        missing_edges = [name for name in EDGE_TYPES if name not in rest_edges]
        if not missing_vertices and not missing_edges:
            return
        graphname = getattr(self.settings, "graphname", "OlympicGraph")
        raise GraphOperationError(
            {
                "operation": "require_bound_graph",
                "method": "getVertexTypes/getEdgeTypes",
                "command": "",
                "exception_class": "RuntimeError",
                "message": (
                    f"REST graph schema for {graphname} is empty or incomplete, so "
                    "upsertVertices cannot proceed (REST-30200: vertex type is not valid "
                    "on this graph). Types may exist in GLOBAL without being bound to the graph. "
                    f"rest_vertices={sorted(rest_vertices)}; rest_edges={sorted(rest_edges)}; "
                    f"missing_vertices={missing_vertices}; missing_edges={missing_edges}. "
                    "Refusing to ALTER GRAPH or run 00_schema.gsql; pass --bind to attach existing "
                    "GLOBAL types, or --reset only to drop and recreate."
                ),
                "code": "REST-30200",
                "status": None,
                "response": "",
            }
        )

    def rest_membership(self) -> dict[str, Any]:
        vertices, edges = self.rest_type_names()
        missing_vertices = [name for name in VERTEX_TYPES if name not in vertices]
        missing_edges = [name for name in EDGE_TYPES if name not in edges]
        return {
            "ok": not missing_vertices and not missing_edges,
            "vertices": [name for name in VERTEX_TYPES if name in vertices],
            "edges": [name for name in EDGE_TYPES if name in edges],
            "missing_vertices": missing_vertices,
            "missing_edges": missing_edges,
        }

    def catalog_type_names(self) -> tuple[set[str], set[str]]:
        return self._gsql_catalog_type_names("vertex"), self._gsql_catalog_type_names("edge")

    def bind_global_types(self) -> dict[str, Any]:
        self._require()
        membership = self.rest_membership()
        bind_file = GSQL_DIR / "03_bind_graph.gsql"
        if membership["ok"]:
            return {
                "ok": True,
                "status": "already_bound",
                "attempted": False,
                "file": bind_file.name,
                "output": "",
                "schema": self._schema_type_snapshot(),
                **membership,
            }
        catalog_vertices, catalog_edges = self.catalog_type_names()
        missing_vertices = [name for name in VERTEX_TYPES if name not in catalog_vertices]
        missing_edges = [name for name in EDGE_TYPES if name not in catalog_edges]
        if missing_vertices or missing_edges:
            raise GraphOperationError(
                {
                    "operation": "bind_global_types",
                    "method": "gsql ls",
                    "command": "USE GLOBAL\\nls",
                    "exception_class": "RuntimeError",
                    "message": (
                        "GLOBAL catalog is missing required types; refusing to bind. "
                        f"missing_vertices={missing_vertices}; missing_edges={missing_edges}."
                    ),
                    "code": None,
                    "status": None,
                    "response": "",
                }
            )
        try:
            output = self.run_gsql_file(bind_file)
        except Exception as exc:
            error = GraphOperationError.from_exc(
                exc,
                operation="bind_global_types",
                method="run_gsql_file",
                command=bind_file.name,
            )
            self.last_error = error.details
            raise error from exc
        membership = self.rest_membership()
        payload = {
            "ok": membership["ok"],
            "status": "bound" if membership["ok"] else "incomplete",
            "attempted": True,
            "file": bind_file.name,
            "output": _redact_secrets(str(output))[:2000],
            "schema": self._schema_type_snapshot(),
            **membership,
        }
        if not membership["ok"]:
            raise GraphOperationError(
                {
                    "operation": "bind_global_types",
                    "method": "getVertexTypes/getEdgeTypes",
                    "command": bind_file.name,
                    "exception_class": "RuntimeError",
                    "message": (
                        "bind job ran but REST membership is still incomplete. "
                        f"missing_vertices={membership['missing_vertices']}; "
                        f"missing_edges={membership['missing_edges']}."
                    ),
                    "code": None,
                    "status": None,
                    "response": payload["output"],
                }
            )
        return payload

    def probe_write(self, cleanup: bool = True) -> dict[str, Any]:
        self._require()
        result: dict[str, Any] = {
            "operation": "probe_write",
            "vertex_type": PROBE_VERTEX_TYPE,
            "probe_id": PROBE_VERTEX_ID,
            "write_ok": False,
            "read_ok": False,
            "delete_attempted": False,
            "delete_ok": False,
            "cleaned_up": False,
            "accepted": 0,
            "error": "",
            "details": None,
        }
        try:
            self.require_bound_graph()
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, GraphOperationError)
                else GraphOperationError.from_exc(exc, operation="probe_write", method="require_bound_graph")
            )
            result["error"] = str(error)
            result["details"] = getattr(error, "details", None)
            return result
        row = _probe_document_row()
        try:
            accepted = self._call_upsert_vertices(PROBE_VERTEX_TYPE, [row])
            result["accepted"] = int(accepted or 0)
            result["write_ok"] = result["accepted"] >= 1
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, GraphOperationError)
                else GraphOperationError.from_exc(exc, operation="probe_write", method="upsertVertices")
            )
            result["error"] = str(error)
            result["details"] = getattr(error, "details", None)
            return result
        try:
            found = self.conn.getVerticesById(PROBE_VERTEX_TYPE, PROBE_VERTEX_ID)
            result["read_ok"] = PROBE_VERTEX_ID in _vertex_ids(found)
            if result["read_ok"]:
                result["write_ok"] = True
            elif not result["error"]:
                result["error"] = f"probe vertex {PROBE_VERTEX_ID} was not returned by getVerticesById"
        except Exception as exc:
            error = GraphOperationError.from_exc(exc, operation="probe_write", method="getVerticesById")
            result["error"] = str(error)
            result["details"] = error.details
        if not cleanup or not result["write_ok"]:
            return result
        result["delete_attempted"] = True
        targets = _probe_target_ids()
        result["stale_probe_ids"] = [vid for vid in targets if vid != PROBE_VERTEX_ID]
        try:
            delete_report = self._delete_probe_vertices(targets)
            result["delete_methods"] = delete_report["methods"]
            result["stale_remaining"] = [
                vid for vid in targets if vid != PROBE_VERTEX_ID and self._vertex_exists(PROBE_VERTEX_TYPE, vid)
            ]
            still_present = self._vertex_exists(PROBE_VERTEX_TYPE, PROBE_VERTEX_ID)
            result["delete_ok"] = not still_present
            result["cleaned_up"] = result["delete_ok"]
            if not result["delete_ok"] and not result["error"]:
                result["error"] = f"probe vertex {PROBE_VERTEX_ID} was still present after delete"
                if delete_report["errors"]:
                    result["details"] = delete_report["errors"][-1]
                    result["error"] = (
                        f"{result['error']}; {format_operation_error(delete_report['errors'][-1])}"
                    )
        except Exception as exc:
            error = GraphOperationError.from_exc(exc, operation="probe_write", method="delete_probe_vertices")
            result["error"] = str(error)
            result["details"] = error.details
            result["delete_ok"] = False
            result["cleaned_up"] = False
        return result

    def _vertex_exists(self, vertex_type: str, vertex_id: str) -> bool:
        getter = getattr(self.conn, "getVerticesById", None)
        if getter is None:
            raise RuntimeError("getVerticesById is not available on this pyTigerGraph version")
        try:
            found = getter(vertex_type, vertex_id)
        except Exception as exc:
            if _is_absent_vertex_error(exc):
                return False
            raise
        return vertex_id in _vertex_ids(found)

    def _delete_probe_vertices(self, vertex_ids: list[str]) -> dict[str, Any]:
        """Delete synthetic probe Documents without using REST DELETE-by-path as the primary API.

        Savanna/RESTPP accepts STRING Document IDs on upsert and GET, but
        DELETE /vertices/{type}/{id} (pyTigerGraph delVerticesById) returns 601
        even for those same IDs. Document uses PRIMARY_ID_AS_ATTRIBUTE, so
        GSQL DELETE FROM and REST filter-delete on `id`/`kind` are the supported
        deletion paths. delVerticesById is only a last resort.
        """
        attempted: list[str] = []
        errors: list[dict[str, Any]] = []
        strategies = (
            ("gsql_delete_from", self._delete_probes_gsql_delete_from),
            ("gsql_interpret_delete", self._delete_probes_gsql_interpret),
            ("delVertices_filter", self._delete_probes_rest_filter),
            ("delVerticesById", self._delete_probes_by_id),
        )
        for name, fn in strategies:
            remaining = self._remaining_probe_ids(vertex_ids)
            if not remaining:
                break
            attempted.append(name)
            try:
                fn(vertex_ids)
            except Exception as exc:
                errors.append(exception_details(exc, operation="probe_write", method=name))
        return {"methods": attempted, "errors": errors}

    def _remaining_probe_ids(self, vertex_ids: list[str]) -> list[str]:
        remaining: list[str] = []
        for vertex_id in vertex_ids:
            try:
                if self._vertex_exists(PROBE_VERTEX_TYPE, vertex_id):
                    remaining.append(vertex_id)
            except Exception:
                remaining.append(vertex_id)
        return remaining

    def _delete_probes_gsql_delete_from(self, vertex_ids: list[str]) -> None:
        graph = _gsql_ident(self.settings.graphname)
        command = f"USE GRAPH {graph}\nDELETE FROM Document:s WHERE {_probe_delete_predicate(vertex_ids)}"
        self.gsql(command)

    def _delete_probes_gsql_interpret(self, vertex_ids: list[str]) -> None:
        graph = _gsql_ident(self.settings.graphname)
        predicate = _probe_delete_predicate(vertex_ids)
        command = (
            f"USE GRAPH {graph}\n"
            f"INTERPRET QUERY () FOR GRAPH {graph} {{\n"
            f"  Deleted = SELECT s FROM Document:s WHERE {predicate} POST-ACCUM DELETE(s);\n"
            f"  PRINT Deleted;\n"
            f"}}"
        )
        self.gsql(command)

    def _delete_probes_rest_filter(self, vertex_ids: list[str]) -> None:
        deleter = getattr(self.conn, "delVertices", None)
        if deleter is None:
            raise RuntimeError("delVertices is not available on this pyTigerGraph version")
        last_error: Exception | None = None
        for where in _probe_rest_filters(vertex_ids):
            try:
                deleter(PROBE_VERTEX_TYPE, where=where, permanent=False)
            except Exception as exc:
                last_error = exc
        if last_error is not None and self._remaining_probe_ids(vertex_ids):
            raise last_error

    def _delete_probes_by_id(self, vertex_ids: list[str]) -> None:
        deleter = getattr(self.conn, "delVerticesById", None)
        if deleter is None:
            raise RuntimeError("delVerticesById is not available on this pyTigerGraph version")
        last_error: Exception | None = None
        for vertex_id in vertex_ids:
            try:
                deleter(PROBE_VERTEX_TYPE, vertex_id, permanent=False)
            except Exception as exc:
                last_error = exc
        if last_error is not None and self._remaining_probe_ids(vertex_ids):
            raise last_error

    def _schema_type_snapshot(self) -> dict[str, Any]:
        try:
            schema = self.schema()
        except Exception:
            return {"GraphName": getattr(self.settings, "graphname", "OlympicGraph"), "VertexTypes": [], "EdgeTypes": []}
        if not isinstance(schema, dict):
            return {"raw": True, "VertexTypes": [], "EdgeTypes": []}
        return {
            "GraphName": schema.get("GraphName"),
            "VertexTypes": sorted(_type_names_from_schema(schema, "vertex")),
            "EdgeTypes": sorted(_type_names_from_schema(schema, "edge")),
        }

    def _upsert_vertices(self, vertex_type: str, rows: list[dict]) -> int:
        total = 0
        batch_size = CHUNK_UPSERT_BATCH if vertex_type == "Chunk" else UPSERT_BATCH
        for batch in _batches(rows, batch_size):
            payload = [_vertex_payload(row) for row in batch]
            accepted = self._call_upsert_vertices(vertex_type, payload)
            total += _require_accepted(accepted, len(payload), operation=f"upsert_vertices:{vertex_type}", method="upsertVertices")
        return total

    def _call_upsert_vertices(self, vertex_type: str, payload: list[dict]) -> int:
        tuples = [(row["id"], {key: value for key, value in row.items() if key != "id"}) for row in payload]
        try:
            return self.conn.upsertVertices(vertex_type, tuples)
        except Exception as exc:
            error = GraphOperationError.from_exc(
                exc,
                operation=f"upsert_vertices:{vertex_type}",
                method="upsertVertices",
            )
            self.last_error = error.details
            raise error from exc

    def _upsert_edges(self, src_type: str, edge_type: str, tgt_type: str, rows: list[dict], fields: list[str]) -> int:
        total = 0
        attr_fields = [name for name in fields if name not in {"src", "tgt"}]
        for batch in _batches(rows, UPSERT_BATCH):
            tuples = []
            for row in batch:
                attrs = {name: _coerce_attr(name, row.get(name, "")) for name in attr_fields}
                tuples.append((row["src"], row["tgt"], attrs))
            try:
                accepted = self.conn.upsertEdges(src_type, edge_type, tgt_type, tuples)
                total += _require_accepted(
                    accepted,
                    len(tuples),
                    operation=f"upsert_edges:{edge_type}",
                    method="upsertEdges",
                )
            except GraphOperationError:
                raise
            except Exception as exc:
                raise GraphOperationError.from_exc(
                    exc,
                    operation=f"upsert_edges:{edge_type}",
                    method="upsertEdges",
                    command=f"{src_type}-[{edge_type}]->{tgt_type}",
                ) from exc
        return total

    def _require(self) -> None:
        if not self.available or self.conn is None:
            raise RuntimeError(self.error or "TigerGraph is not connected")


_GSQL_VERTEX_RE = re.compile(r"(?im)^\s*-?\s*VERTEX\s+([A-Za-z_][\w]*)\s*\(")
_GSQL_EDGE_RE = re.compile(
    r"(?im)^\s*-?\s*(?:(?:DIRECTED|UNDIRECTED)\s+)?EDGE\s+([A-Za-z_][\w]*)\s*\("
)
_SCHEMA_TYPE_KEYS = {
    "vertex": ("VertexTypes", "vertexTypes", "vertices", "VertexType", "vertex_types"),
    "edge": ("EdgeTypes", "edgeTypes", "edges", "EdgeType", "edge_types"),
}


def _type_names(payload: Any, parent: str | None = None) -> set[str]:
    if payload is None or isinstance(payload, (bool, int, float)):
        return set()
    if isinstance(payload, str):
        if payload and (parent is None or parent in _SCHEMA_TYPE_KEYS["vertex"] + _SCHEMA_TYPE_KEYS["edge"]):
            return {payload}
        return set()
    if isinstance(payload, (list, tuple, set)):
        names: set[str] = set()
        for item in payload:
            if isinstance(item, str) and item:
                names.add(item)
            else:
                names |= _type_names(item, parent=parent)
        return names
    if isinstance(payload, dict):
        names = set()
        if parent in _SCHEMA_TYPE_KEYS["vertex"] + _SCHEMA_TYPE_KEYS["edge"]:
            for key in ("Name", "name"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    names.add(value)
        for key, value in payload.items():
            names |= _type_names(value, parent=str(key))
        return names
    name_attr = getattr(payload, "Name", None) or getattr(payload, "name", None)
    if isinstance(name_attr, str) and name_attr:
        return {name_attr}
    return set()


def _type_names_from_schema(schema: Any, kind: str) -> set[str]:
    if not isinstance(schema, dict):
        return _type_names(schema)
    names: set[str] = set()
    for key in _SCHEMA_TYPE_KEYS[kind]:
        if key in schema:
            names |= _type_names(schema[key], parent=key)
    if names:
        return names
    for nested_key in ("results", "schema", "graph", "data", "output"):
        nested = schema.get(nested_key)
        if nested is not None:
            names |= _type_names_from_schema(nested, kind)
            if names:
                return names
    return names


def _type_names_from_gsql(catalog: str, kind: str) -> set[str]:
    if not catalog:
        return set()
    pattern = _GSQL_VERTEX_RE if kind == "vertex" else _GSQL_EDGE_RE
    return {match.group(1) for match in pattern.finditer(catalog)}


def flatten_query_result(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    merged: dict[str, Any] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                merged.update(item)
    return merged


def _probe_document_row() -> dict[str, Any]:
    row = {
        "id": PROBE_VERTEX_ID,
        "title": "synthetic TigerGraph write probe",
        "url": "",
        "wikidata_qid": PROBE_VERTEX_ID,
        "wikipedia_pageid": 0,
        "approx_tokens": 0,
        "has_olympic_infobox": False,
        "kind": PROBE_KIND,
        "rejection_reason": "synthetic_write_probe",
        "is_event": False,
        "event_id": "",
    }
    return {name: row[name] for name in DOCUMENT_FIELDS}


def _probe_target_ids() -> list[str]:
    seen: list[str] = []
    for vertex_id in (PROBE_VERTEX_ID, *STALE_PROBE_IDS):
        if vertex_id not in seen:
            seen.append(vertex_id)
    return seen


def _gsql_ident(name: str) -> str:
    ident = str(name or "")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ident):
        raise ValueError("graph name is not a safe GSQL identifier")
    return ident


def _gsql_string(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _probe_delete_predicate(vertex_ids: list[str]) -> str:
    clauses = [f"s.kind == {_gsql_string(PROBE_KIND)}"]
    for vertex_id in vertex_ids:
        clauses.append(f"s.id == {_gsql_string(vertex_id)}")
    return " OR ".join(clauses)


def _probe_rest_filters(vertex_ids: list[str]) -> list[str]:
    filters = [f'kind="{PROBE_KIND}"', f'kind=="{PROBE_KIND}"']
    for vertex_id in vertex_ids:
        filters.append(f'id="{vertex_id}"')
        filters.append(f'id=="{vertex_id}"')
    return filters


def _is_absent_vertex_error(exc: Exception) -> bool:
    code = str(getattr(exc, "code", "") or "")
    if code in {"601", "REST-601"}:
        return True
    message = str(getattr(exc, "message", None) or exc).casefold()
    return "not a valid vertex id" in message


def _vertex_ids(payload: Any) -> set[str]:
    found: set[str] = set()
    if payload is None:
        return found
    if isinstance(payload, (str, int)):
        return {str(payload)} if str(payload) else found
    if isinstance(payload, dict):
        for key in ("v_id", "id"):
            value = payload.get(key)
            if isinstance(value, (str, int)) and str(value):
                found.add(str(value))
        attributes = payload.get("attributes")
        if isinstance(attributes, dict):
            attr_id = attributes.get("id")
            if isinstance(attr_id, (str, int)) and str(attr_id):
                found.add(str(attr_id))
        return found
    if isinstance(payload, (list, tuple)):
        for item in payload:
            found |= _vertex_ids(item)
    return found


def _vertex_payload(row: dict) -> dict:
    payload = {"id": row["id"]}
    for key, value in row.items():
        if key == "id":
            continue
        payload[key] = _coerce_attr(key, value)
    return payload


# Attribute types from gsql/00_schema.gsql. STRING fields must stay strings even when
# the export value is digits (e.g. Event.competitors_raw); Savanna rejects JSON numbers
# for STRING attributes (REST-30200).
_INT_ATTRS = {
    "approx_tokens",
    "chunk_index",
    "competitors",
    "end_char",
    "nations",
    "next_year",
    "prev_year",
    "start_char",
    "token_count",
    "wikipedia_pageid",
    "year",
}
_BOOL_ATTRS = {
    "competitors_overflow",
    "has_competitors",
    "has_gold",
    "has_nations",
    "has_olympic_infobox",
    "is_event",
}


def _coerce_attr(name: str, value: Any) -> Any:
    if name in _BOOL_ATTRS:
        return _as_bool(value)
    if name in _INT_ATTRS:
        return _as_int(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = "" if value is None else str(value).strip().casefold()
    return text in {"true", "1", "yes"}


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = "" if value is None else str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return 0


def _batches(rows: list, size: int):
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


def _require_accepted(accepted: Any, expected: int, *, operation: str, method: str) -> int:
    if accepted is None:
        count = 0
    else:
        count = int(accepted)
    if count != expected:
        raise GraphOperationError(
            {
                "operation": operation,
                "method": method,
                "command": "",
                "exception_class": "RuntimeError",
                "message": (
                    f"partial batch: accepted {count} of {expected} items; "
                    "refusing to treat a shortfall as success"
                ),
                "code": "partial_upsert",
                "status": None,
                "response": "",
            }
        )
    return count


def _live_count_mismatches(
    expected: dict[str, Any],
    vertices: dict[str, Any],
    edges: dict[str, Any],
) -> dict[str, dict[str, int]]:
    mismatches: dict[str, dict[str, int]] = {}
    for name, count in expected.items():
        actual = vertices.get(name)
        if actual is None:
            actual = edges.get(name)
        if actual is None or int(actual) != int(count):
            mismatches[name] = {"expected": int(count), "actual": 0 if actual is None else int(actual)}
    return mismatches


def _query_names(payload: Any) -> list[str]:
    names: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            names.extend(_query_names(key))
            names.extend(_query_names(value))
    elif isinstance(payload, list):
        for item in payload:
            names.extend(_query_names(item))
    elif isinstance(payload, str):
        path = payload.split("?", 1)[0].strip()
        if not path:
            return names
        if "/" in path:
            tail = path.rsplit("/", 1)[-1]
            if tail:
                names.append(tail)
        else:
            names.append(path)
    return names


def _classify_environment(settings: TigerGraphSettings) -> str:
    host = settings.host.casefold()
    if settings.tg_cloud or "tgcloud" in host or "tigergraph.io" in host or "i.tgcloud" in host:
        return "savanna_or_tgcloud"
    if "localhost" in host or "127.0.0.1" in host:
        return "local_ce"
    return "remote"


class GraphOperationError(RuntimeError):
    def __init__(self, details: dict[str, Any]) -> None:
        self.details = details
        super().__init__(format_operation_error(details))

    @classmethod
    def from_exc(
        cls,
        exc: Exception,
        *,
        operation: str,
        method: str,
        command: str = "",
    ) -> "GraphOperationError":
        return cls(exception_details(exc, operation=operation, method=method, command=command))


def _redact_secrets(text: str) -> str:
    if not text:
        return ""
    redacted = _SECRET_RE.sub(r"\1\2[REDACTED]", text)
    redacted = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1[REDACTED]", redacted)
    redacted = re.sub(r"https?://[^\s'\"\\]+", "[REDACTED_URL]", redacted)
    return redacted


def exception_details(
    exc: Exception,
    *,
    operation: str,
    method: str,
    command: str = "",
) -> dict[str, Any]:
    raw_message = getattr(exc, "message", None)
    if raw_message is None:
        raw_message = str(exc)
    response = ""
    response_obj = getattr(exc, "response", None)
    if response_obj is not None:
        status = getattr(response_obj, "status_code", None) or getattr(response_obj, "status", None)
        try:
            response = getattr(response_obj, "text", None) or str(response_obj)
        except Exception:
            response = str(response_obj)
    else:
        status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    details = {
        "operation": operation,
        "method": method,
        "command": _redact_secrets(command)[:500],
        "exception_class": exc.__class__.__name__,
        "message": _redact_secrets(str(raw_message))[:2000],
        "code": getattr(exc, "code", None),
        "status": status,
        "response": _redact_secrets(str(response))[:2000],
    }
    _LOG.warning(
        "graph operation failed operation=%s method=%s class=%s code=%s status=%s message=%s",
        details["operation"],
        details["method"],
        details["exception_class"],
        details["code"],
        details["status"],
        details["message"],
    )
    return details


def format_operation_error(details: dict[str, Any]) -> str:
    parts = [
        details.get("exception_class") or "Error",
        f"operation={details.get('operation') or 'unknown'}",
        f"method={details.get('method') or 'unknown'}",
    ]
    if details.get("command"):
        parts.append(f"command={details['command']}")
    if details.get("code") not in (None, ""):
        parts.append(f"code={details['code']}")
    if details.get("status") not in (None, ""):
        parts.append(f"status={details['status']}")
    if details.get("message"):
        parts.append(f"message={details['message']}")
    if details.get("response"):
        parts.append(f"response={details['response']}")
    return "; ".join(str(part) for part in parts if part)


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, GraphOperationError):
        return str(exc)[:300]
    return format_operation_error(
        exception_details(exc, operation="connect", method="TigerGraphConnection")
    )[:300]


def repo_gsql_queries() -> list[str]:
    text = (GSQL_DIR / "02_queries.gsql").read_text(encoding="utf-8")
    return [name for name in INSTALLED_QUERIES if f"QUERY {name}" in text]
