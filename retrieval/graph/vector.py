"""TigerGraph Vector search helpers. Fail closed; never fall back to BM25."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

VECTOR_ATTRIBUTE = "embedding"
VECTOR_METRIC = "COSINE"
VECTOR_INDEX_TYPE = "HNSW"
SEARCH_QUERY = "search_chunk_embedding"
SCHEMA_JOB = "add_chunk_embedding_v1"

_VERTEX_HEADER_RE = re.compile(r"^\s*-\s+(\w+)\s*:\s*$")
_VEC_ATTR_RE = re.compile(r"^\s*-\s+(\w+)\((.+)\)\s*$")
_KV_RE = re.compile(r'(\w+)\s*=\s*"?([^",]+)"?')


class VectorBackendError(RuntimeError):
    """TigerGraph vector search/schema is unavailable or returned an invalid payload."""


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    score: float
    rank: int
    attributes: dict[str, Any] = field(default_factory=dict)


def parse_vector_attributes_from_ls(ls_text: str) -> list[dict[str, Any]]:
    in_section = False
    current_vertex: str | None = None
    found: list[dict[str, Any]] = []
    for line in (ls_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("Vector Embeddings"):
            in_section = True
            current_vertex = None
            continue
        if not in_section:
            continue
        if stripped and not line[:1].isspace() and not stripped.startswith("-"):
            in_section = False
            current_vertex = None
            continue
        header = _VERTEX_HEADER_RE.match(line)
        if header:
            current_vertex = header.group(1)
            continue
        match = _VEC_ATTR_RE.match(line)
        if match and current_vertex:
            params = {key: value for key, value in _KV_RE.findall(match.group(2))}
            entry: dict[str, Any] = {
                "vertex_type": current_vertex,
                "vector_name": match.group(1),
            }
            if "Dimension" in params:
                entry["dimension"] = int(params["Dimension"])
            if "IndexType" in params:
                entry["index_type"] = params["IndexType"]
            if "Metric" in params:
                entry["metric"] = params["Metric"].upper()
            if "DataType" in params:
                entry["data_type"] = params["DataType"]
            found.append(entry)
    return found


def schema_change_gsql(dimension: int, *, graph_name: str = "OlympicGraph") -> str:
    del graph_name
    return (
        "USE GLOBAL\n"
        f"CREATE GLOBAL SCHEMA_CHANGE JOB {SCHEMA_JOB} {{\n"
        f"  ALTER VERTEX Chunk ADD VECTOR ATTRIBUTE {VECTOR_ATTRIBUTE}"
        f'(DIMENSION={int(dimension)}, METRIC="{VECTOR_METRIC}", '
        f'INDEXTYPE="{VECTOR_INDEX_TYPE}", DATATYPE="FLOAT");\n'
        "}\n"
        f"RUN GLOBAL SCHEMA_CHANGE JOB {SCHEMA_JOB}\n"
        f"DROP JOB {SCHEMA_JOB}\n"
    )


def search_query_gsql(*, graph_name: str = "OlympicGraph") -> str:
    return (
        f"USE GRAPH {graph_name}\n"
        f"CREATE OR REPLACE QUERY {SEARCH_QUERY}(LIST<FLOAT> query_vec, INT k) "
        f"FOR GRAPH {graph_name} SYNTAX v3 {{\n"
        "  MapAccum<VERTEX, FLOAT> @@distances;\n"
        f"  v = vectorSearch({{Chunk.{VECTOR_ATTRIBUTE}}}, query_vec, k, {{distance_map: @@distances}});\n"
        "  PRINT v;\n"
        "  PRINT @@distances AS distances;\n"
        "}\n"
        f"INSTALL QUERY {SEARCH_QUERY}\n"
    )


def parse_vector_search_result(raw: Any) -> list[VectorHit]:
    blocks = _as_list(raw)
    vertices: list[Any] = []
    distances: dict[str, float] = {}
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if "v" in block:
            vertices.extend(_as_list(block.get("v")))
        if "distances" in block:
            distances.update(_parse_distance_map(block.get("distances")))
    parsed: list[tuple[str, float, dict[str, Any], int]] = []
    seen: set[str] = set()
    for index, item in enumerate(vertices):
        vertex_id, attrs = _vertex_id_and_attrs(item)
        if not vertex_id or vertex_id in seen:
            continue
        seen.add(vertex_id)
        distance = distances.get(vertex_id)
        score = -float(distance) if distance is not None else -float(index)
        parsed.append((vertex_id, score, attrs, index))
    parsed.sort(key=lambda row: (-row[1], row[0]))
    hits: list[VectorHit] = []
    for rank, (chunk_id, score, attrs, _index) in enumerate(parsed, start=1):
        hits.append(VectorHit(chunk_id=chunk_id, score=score, rank=rank, attributes=attrs))
    return hits


def map_vector_hits_to_chunks(hits: list[VectorHit], chunks_by_id: dict[str, Any]) -> list[Any]:
    from retrieval.rag.models import RetrievalHit

    mapped: list[RetrievalHit] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        chunk = chunks_by_id.get(hit.chunk_id)
        if chunk is None:
            continue
        mapped.append(
            RetrievalHit(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                score=float(hit.score),
                rank=len(mapped) + 1,
                text=chunk.text,
                section=chunk.section,
                kind=chunk.kind,
                event_id=chunk.event_id,
                retrieval_method="tigergraph_vector",
                source_url=chunk.source_url,
            )
        )
    return mapped


def index_is_ready(status: Any) -> bool:
    if not isinstance(status, dict):
        return False
    pending = status.get("NeedRebuildServers")
    return pending == [] or pending is None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_distance_map(raw: Any) -> dict[str, float]:
    distances: dict[str, float] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            vertex_id = _coerce_vertex_id(key)
            if vertex_id is None:
                nested_id, _attrs = _vertex_id_and_attrs(key)
                vertex_id = nested_id
            if vertex_id is None:
                continue
            try:
                distances[vertex_id] = float(value)
            except (TypeError, ValueError):
                continue
        return distances
    for item in _as_list(raw):
        if isinstance(item, dict):
            vertex_id, _attrs = _vertex_id_and_attrs(item)
            if vertex_id is None:
                vertex_id = _coerce_vertex_id(item.get("key") or item.get("from") or item.get("vertex"))
            score = item.get("distance", item.get("value", item.get("score")))
            if vertex_id is None or score is None:
                continue
            try:
                distances[vertex_id] = float(score)
            except (TypeError, ValueError):
                continue
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            vertex_id = _coerce_vertex_id(item[0])
            if vertex_id is None:
                continue
            try:
                distances[vertex_id] = float(item[1])
            except (TypeError, ValueError):
                continue
    return distances


def _vertex_id_and_attrs(item: Any) -> tuple[str | None, dict[str, Any]]:
    if isinstance(item, str):
        return item, {}
    if not isinstance(item, dict):
        return _coerce_vertex_id(item), {}
    attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
    vertex_id = (
        item.get("v_id")
        or item.get("id")
        or attrs.get("id")
        or item.get("vertex_id")
    )
    if vertex_id is None and isinstance(item.get("v_id"), dict):
        vertex_id = item["v_id"].get("id")
    return _coerce_vertex_id(vertex_id), dict(attrs)


def _coerce_vertex_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return _coerce_vertex_id(value.get("id") or value.get("v_id"))
    text = str(value).strip()
    return text or None
