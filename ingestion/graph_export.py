"""Write deterministic local TigerGraph export artifacts."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ingestion.graph_records import (
    EDGE_TYPES,
    VERTEX_TYPES,
    GraphEdge,
    GraphExport,
    build_graph_export,
    record_to_dict,
)
from ingestion.graph_tsv import write_jsonl, write_tsv
from ingestion.models import ParsedDocument, TextChunk
from ingestion.paths import DEFAULT_GRAPH_EXPORT_DIR

DOCUMENT_FIELDS = [
    "id",
    "title",
    "url",
    "wikidata_qid",
    "wikipedia_pageid",
    "approx_tokens",
    "has_olympic_infobox",
    "kind",
    "rejection_reason",
    "is_event",
    "event_id",
]
EVENT_FIELDS = [
    "id",
    "document_id",
    "title",
    "title_folded",
    "sport",
    "sport_raw",
    "event_name_raw",
    "event_name_folded",
    "event_name_normalized",
    "event_name_core",
    "event_name_gender",
    "infobox_event",
    "year",
    "season",
    "games_id",
    "infobox_games_raw",
    "year_source",
    "venue_raw",
    "venue_key",
    "venue_compact",
    "venue_tokens",
    "date_raw",
    "date_years",
    "date_months",
    "date_days",
    "date_compact",
    "competitors",
    "has_competitors",
    "competitors_raw",
    "nations",
    "has_nations",
    "nations_raw",
    "gold_raw",
    "gold_noc",
    "silver_raw",
    "bronze_raw",
    "has_gold",
    "prev_year",
    "next_year",
    "prev_year_raw",
    "next_year_raw",
    "flags",
    "url",
    "competitors_overflow",
]
GAMES_FIELDS = ["id", "year", "season"]
SPORT_FIELDS = ["id", "name"]
VENUE_FIELDS = ["id", "name", "compact", "tokens"]
CONTAINS_FIELDS = ["src", "tgt", "document_id", "source_chunk_id", "chunk_index", "section"]
DESCRIBES_FIELDS = ["src", "tgt", "document_id", "source_chunk_id"]
IN_GAMES_FIELDS = ["src", "tgt", "document_id", "source_chunk_id", "year", "season"]
OF_SPORT_FIELDS = ["src", "tgt", "document_id", "source_chunk_id", "sport_raw"]
HELD_AT_FIELDS = ["src", "tgt", "document_id", "source_chunk_id", "venue_raw"]

EDGE_FIELD_MAP = {
    "CONTAINS_CHUNK": CONTAINS_FIELDS,
    "DESCRIBES": DESCRIBES_FIELDS,
    "IN_GAMES": IN_GAMES_FIELDS,
    "OF_SPORT": OF_SPORT_FIELDS,
    "HELD_AT": HELD_AT_FIELDS,
}


def export_graph(
    documents: list[ParsedDocument],
    output_dir: str | Path | None = None,
    chunks: list[TextChunk] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    directory = Path(output_dir) if output_dir is not None else DEFAULT_GRAPH_EXPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    graph = build_graph_export(documents, chunks=chunks)
    files: dict[str, str] = {}
    checksums: dict[str, str] = {}

    files["vertices_document.tsv"] = str(directory / "vertices_document.tsv")
    write_tsv(directory / "vertices_document.tsv", map(record_to_dict, graph.documents), DOCUMENT_FIELDS)
    files["vertices_event.tsv"] = str(directory / "vertices_event.tsv")
    write_tsv(directory / "vertices_event.tsv", map(record_to_dict, graph.events), EVENT_FIELDS)
    files["vertices_games.tsv"] = str(directory / "vertices_games.tsv")
    write_tsv(directory / "vertices_games.tsv", map(record_to_dict, graph.games), GAMES_FIELDS)
    files["vertices_sport.tsv"] = str(directory / "vertices_sport.tsv")
    write_tsv(directory / "vertices_sport.tsv", map(record_to_dict, graph.sports), SPORT_FIELDS)
    files["vertices_venue.tsv"] = str(directory / "vertices_venue.tsv")
    write_tsv(directory / "vertices_venue.tsv", map(record_to_dict, graph.venues), VENUE_FIELDS)
    files["vertices_chunk.jsonl"] = str(directory / "vertices_chunk.jsonl")
    write_jsonl(directory / "vertices_chunk.jsonl", map(record_to_dict, graph.chunks))

    edge_tables = dict(graph.iter_edge_tables())
    for edge_type, fields in EDGE_FIELD_MAP.items():
        name = f"edges_{edge_type.lower()}.tsv"
        files[name] = str(directory / name)
        write_tsv(directory / name, map(_edge_dict, edge_tables[edge_type]), fields)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    for relative, absolute in files.items():
        checksums[relative] = _sha256(Path(absolute))
    manifest = {
        "format": "tsv+jsonl",
        "loading_strategy": "upsert_by_stable_id",
        "graph_name": "OlympicGraph",
        "vertex_types": list(VERTEX_TYPES),
        "edge_types": list(EDGE_TYPES),
        "counts": graph.counts(),
        "files": files,
        "checksums": checksums,
        "elapsed_ms": round(elapsed_ms, 3),
        "notes": graph.notes,
        "identity_rules": {
            "Document": "corpus doc_id",
            "Event": "Phase 1 event_id == doc_id",
            "Chunk": "{doc_id}::c{index:03d}",
            "Games": "{year}_{season}",
            "Sport": "normalize_sport_key(title sport)",
            "Venue": "normalize_venue_key(venue_raw) — NFKC, whitespace, casefold only",
        },
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    graph.elapsed_ms = elapsed_ms
    return {"directory": str(directory), "manifest": manifest, "graph": graph}


def _edge_dict(edge: GraphEdge) -> dict[str, Any]:
    return record_to_dict(edge)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
