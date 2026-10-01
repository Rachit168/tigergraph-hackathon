"""Live TigerGraph Vector schema, upsert, and search. Fail closed."""

from __future__ import annotations

import time
from typing import Any, Iterable

from retrieval.graph.client import TigerGraphClient, _require_accepted
from retrieval.graph.vector import (
    SCHEMA_JOB,
    SEARCH_QUERY,
    VECTOR_ATTRIBUTE,
    VectorBackendError,
    VectorHit,
    index_is_ready,
    parse_vector_attributes_from_ls,
    parse_vector_search_result,
    schema_change_gsql,
    search_query_gsql,
)

UPSERT_BATCH = 5
INDEX_WAIT_S = 5.0
INDEX_ATTEMPTS = 36
UPSERT_RETRIES = 6


class TigerGraphVectorStore:
    def __init__(self, client: TigerGraphClient, *, dimension: int) -> None:
        self.client = client
        self.dimension = int(dimension)

    def list_vector_attributes(self) -> list[dict[str, Any]]:
        ls = self.client.gsql(f"USE GRAPH {self.client.settings.graphname}\nLS")
        return parse_vector_attributes_from_ls(str(ls))

    def chunk_embedding_spec(self) -> dict[str, Any] | None:
        for item in self.list_vector_attributes():
            if item.get("vertex_type") == "Chunk" and item.get("vector_name") == VECTOR_ATTRIBUTE:
                return item
        return None

    def ensure_schema(self) -> dict[str, Any]:
        existing = self.chunk_embedding_spec()
        created = False
        if existing is None:
            try:
                self.client.gsql(f"USE GLOBAL\nDROP JOB {SCHEMA_JOB}")
            except Exception:
                pass
            raw = str(self.client.gsql(schema_change_gsql(self.dimension, graph_name=self.client.settings.graphname)))
            lowered = raw.casefold()
            if "error" in lowered and "already" not in lowered and "exist" not in lowered:
                raise VectorBackendError(f"vector schema change failed: {raw[:1500]}")
            created = True
            existing = self.chunk_embedding_spec()
        if existing is None:
            raise VectorBackendError("Chunk.embedding VECTOR ATTRIBUTE is not present after schema change")
        actual_dim = int(existing.get("dimension") or 0)
        if actual_dim and actual_dim != self.dimension:
            raise VectorBackendError(
                f"Chunk.embedding dimension is {actual_dim}, embedder is {self.dimension}"
            )
        return {"created": created, "spec": existing, "ls_ok": True}

    def require_schema(self) -> dict[str, Any]:
        """Read-only check. Never creates or alters VECTOR ATTRIBUTE."""
        existing = self.chunk_embedding_spec()
        if existing is None:
            raise VectorBackendError(
                "Chunk.embedding VECTOR ATTRIBUTE is not present; "
                "refusing schema change because skip-ensure is set"
            )
        actual_dim = int(existing.get("dimension") or 0)
        if actual_dim and actual_dim != self.dimension:
            raise VectorBackendError(
                f"Chunk.embedding dimension is {actual_dim}, embedder is {self.dimension}"
            )
        return {"created": False, "spec": existing, "ls_ok": True, "ensured": False}

    def ensure_search_query(self) -> dict[str, Any]:
        installed = set(self.client.installed_queries())
        raw = str(self.client.gsql(search_query_gsql(graph_name=self.client.settings.graphname)))
        lowered = raw.casefold()
        if "saved as draft" in lowered or "syntax error" in lowered or "type check error" in lowered:
            raise VectorBackendError(f"search_chunk_embedding failed to compile: {raw[:2000]}")
        if SEARCH_QUERY not in set(self.client.installed_queries()) and SEARCH_QUERY not in installed:
            extra = str(
                self.client.gsql(
                    f"USE GRAPH {self.client.settings.graphname}\nINSTALL QUERY {SEARCH_QUERY}"
                )
            )
            raw = f"{raw}\n{extra}"
        if SEARCH_QUERY not in set(self.client.installed_queries()):
            raise VectorBackendError("search_chunk_embedding is not installed")
        return {"query": SEARCH_QUERY, "installed": True}

    def require_search_query(self) -> dict[str, Any]:
        """Read-only check. Never CREATE OR REPLACE or INSTALL QUERY."""
        installed = set(self.client.installed_queries())
        if SEARCH_QUERY not in installed:
            raise VectorBackendError(
                "search_chunk_embedding is not installed; "
                "refusing CREATE OR REPLACE / INSTALL because skip-ensure is set"
            )
        return {"query": SEARCH_QUERY, "installed": True, "ensured": False}

    def require_index(self) -> dict[str, Any]:
        """Read-only index probe. Never rebuilds or upserts."""
        status = self.index_status()
        payload = status if isinstance(status, dict) else {"raw": status}
        if not index_is_ready(payload):
            raise VectorBackendError(f"vector index is not ready: {payload}")
        return {"ready": True, "status": payload, "ensured": False}

    def index_status(self) -> dict[str, Any]:
        getter = getattr(self.client.conn, "getVectorIndexStatus", None)
        if getter is None:
            raise VectorBackendError("pyTigerGraph getVectorIndexStatus is unavailable")
        try:
            return getter(
                graphName=self.client.settings.graphname,
                vertexType="Chunk",
                vectorName=VECTOR_ATTRIBUTE,
            )
        except TypeError:
            return getter()

    def wait_until_ready(self, *, attempts: int = INDEX_ATTEMPTS, delay_s: float = INDEX_WAIT_S) -> dict[str, Any]:
        last: dict[str, Any] = {}
        for attempt in range(max(1, attempts)):
            try:
                status = self.index_status()
            except Exception as exc:
                raise VectorBackendError(f"vector index status failed: {exc}") from exc
            last = status if isinstance(status, dict) else {"raw": status}
            if index_is_ready(last):
                return {"ready": True, "attempts": attempt + 1, "status": last}
            if attempt + 1 < attempts and delay_s > 0:
                time.sleep(delay_s)
        raise VectorBackendError(f"vector index was not ready: {last}")

    def upsert_vectors(self, rows: Iterable[tuple[str, list[float]]]) -> int:
        payload = list(rows)
        total = 0
        for start in range(0, len(payload), UPSERT_BATCH):
            batch = payload[start : start + UPSERT_BATCH]
            tuples = []
            for chunk_id, vector in batch:
                if len(vector) != self.dimension:
                    raise VectorBackendError(
                        f"vector length {len(vector)} != {self.dimension} for {chunk_id}"
                    )
                tuples.append(
                    (
                        chunk_id,
                        {VECTOR_ATTRIBUTE: list(vector), "embedding_status": "ready"},
                    )
                )
            last_error: Exception | None = None
            for attempt in range(UPSERT_RETRIES):
                try:
                    accepted = self.client.conn.upsertVertices("Chunk", tuples)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(min(30.0, 2.0 ** attempt))
            if last_error is not None:
                raise VectorBackendError(
                    f"vector upsert failed at {start}/{len(payload)}: {last_error}"
                ) from last_error
            if accepted is True:
                total += len(tuples)
            else:
                total += _require_accepted(
                    accepted,
                    len(tuples),
                    operation="upsert_vectors:Chunk",
                    method="upsertVertices",
                )
            if total % 1000 == 0 or start + UPSERT_BATCH >= len(payload):
                print(f"upserted {total}/{len(payload)}", flush=True)
        return total

    def search(self, query_vector: list[float], *, top_k: int) -> list[VectorHit]:
        if len(query_vector) != self.dimension:
            raise VectorBackendError(
                f"query vector dimension {len(query_vector)} != {self.dimension}"
            )
        if top_k <= 0:
            return []
        try:
            raw = self.client.run_query(
                SEARCH_QUERY,
                {"query_vec": [float(value) for value in query_vector], "k": int(top_k)},
            )
        except Exception as exc:
            raise VectorBackendError(f"vectorSearch failed: {exc}") from exc
        return parse_vector_search_result(raw)
