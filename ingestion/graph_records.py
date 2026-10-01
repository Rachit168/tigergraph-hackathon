"""Flat graph records derived from Phase 1 Events and Phase 3 chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from ingestion.chunker import chunk_corpus
from ingestion.graph_ids import (
    clamp_int32,
    compact_venue,
    event_name_core,
    event_name_gender,
    games_id,
    packed_ints,
    sport_id,
    venue_id,
    venue_token_blob,
)
from ingestion.models import ParsedDocument, ParsedEvent, TextChunk
from retrieval.structured.match import compact_alnum, fold_text, normalize_event_name

GRAPH_NAME = "OlympicGraph"
VERTEX_TYPES = ("Document", "Chunk", "Event", "Games", "Sport", "Venue")
EDGE_TYPES = ("CONTAINS_CHUNK", "DESCRIBES", "IN_GAMES", "OF_SPORT", "HELD_AT")
INSTALLED_QUERIES = (
    "lookup_event",
    "count_over_threshold",
    "argmax_competitors",
    "previous_event_gold",
    "events_at_venue_date",
)


@dataclass(frozen=True)
class GraphDocument:
    id: str
    title: str
    url: str
    wikidata_qid: str
    wikipedia_pageid: int
    approx_tokens: int
    has_olympic_infobox: bool
    kind: str
    rejection_reason: str
    is_event: bool
    event_id: str


@dataclass(frozen=True)
class GraphChunk:
    id: str
    document_id: str
    document_title: str
    source_url: str
    text: str
    indexed_text: str
    start_char: int
    end_char: int
    token_count: int
    chunk_index: int
    section: str
    kind: str
    event_id: str
    document_kind: str
    embedding_status: str = "deferred"


@dataclass(frozen=True)
class GraphEvent:
    id: str
    document_id: str
    title: str
    title_folded: str
    sport: str
    sport_raw: str
    event_name_raw: str
    event_name_folded: str
    event_name_normalized: str
    event_name_core: str
    event_name_gender: str
    infobox_event: str
    year: int
    season: str
    games_id: str
    infobox_games_raw: str
    year_source: str
    venue_raw: str
    venue_key: str
    venue_compact: str
    venue_tokens: str
    date_raw: str
    date_years: str
    date_months: str
    date_days: str
    date_compact: str
    competitors: int
    has_competitors: bool
    competitors_raw: str
    nations: int
    has_nations: bool
    nations_raw: str
    gold_raw: str
    gold_noc: str
    silver_raw: str
    bronze_raw: str
    has_gold: bool
    prev_year: int
    next_year: int
    prev_year_raw: str
    next_year_raw: str
    flags: str
    url: str
    competitors_overflow: bool = False


@dataclass(frozen=True)
class GraphGames:
    id: str
    year: int
    season: str


@dataclass(frozen=True)
class GraphSport:
    id: str
    name: str


@dataclass(frozen=True)
class GraphVenue:
    id: str
    name: str
    compact: str
    tokens: str


@dataclass(frozen=True)
class GraphEdge:
    src: str
    tgt: str
    document_id: str = ""
    source_chunk_id: str = ""
    chunk_index: int = -1
    section: str = ""
    year: int = 0
    season: str = ""
    sport_raw: str = ""
    venue_raw: str = ""


@dataclass
class GraphExport:
    documents: list[GraphDocument]
    chunks: list[GraphChunk]
    events: list[GraphEvent]
    games: list[GraphGames]
    sports: list[GraphSport]
    venues: list[GraphVenue]
    contains_chunk: list[GraphEdge]
    describes: list[GraphEdge]
    in_games: list[GraphEdge]
    of_sport: list[GraphEdge]
    held_at: list[GraphEdge]
    elapsed_ms: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {
            "Document": len(self.documents),
            "Chunk": len(self.chunks),
            "Event": len(self.events),
            "Games": len(self.games),
            "Sport": len(self.sports),
            "Venue": len(self.venues),
            "CONTAINS_CHUNK": len(self.contains_chunk),
            "DESCRIBES": len(self.describes),
            "IN_GAMES": len(self.in_games),
            "OF_SPORT": len(self.of_sport),
            "HELD_AT": len(self.held_at),
        }

    def iter_vertex_tables(self) -> Iterator[tuple[str, list[Any]]]:
        yield "Document", self.documents
        yield "Chunk", self.chunks
        yield "Event", self.events
        yield "Games", self.games
        yield "Sport", self.sports
        yield "Venue", self.venues

    def iter_edge_tables(self) -> Iterator[tuple[str, list[GraphEdge]]]:
        yield "CONTAINS_CHUNK", self.contains_chunk
        yield "DESCRIBES", self.describes
        yield "IN_GAMES", self.in_games
        yield "OF_SPORT", self.of_sport
        yield "HELD_AT", self.held_at


def build_graph_export(
    documents: list[ParsedDocument],
    chunks: list[TextChunk] | None = None,
) -> GraphExport:
    chunk_rows = list(chunks) if chunks is not None else chunk_corpus(documents)
    documents_by_id = {document.doc_id: document for document in documents}
    infobox_chunk: dict[str, str] = {}
    first_chunk: dict[str, str] = {}
    for chunk in chunk_rows:
        first_chunk.setdefault(chunk.document_id, chunk.chunk_id)
        if chunk.kind == "infobox":
            infobox_chunk.setdefault(chunk.document_id, chunk.chunk_id)

    graph_documents = [_document_record(document) for document in documents]
    graph_chunks = [_chunk_record(chunk) for chunk in chunk_rows]
    graph_events = [_event_record(documents_by_id[event.document_id], event) for event in _events(documents)]

    games_map: dict[str, GraphGames] = {}
    sport_map: dict[str, GraphSport] = {}
    venue_raws: dict[str, set[str]] = {}
    for event in graph_events:
        if event.games_id:
            games_map[event.games_id] = GraphGames(id=event.games_id, year=event.year, season=event.season)
        if event.sport:
            sport_map.setdefault(event.sport, GraphSport(id=event.sport, name=event.sport_raw or event.sport))
        if event.venue_key:
            venue_raws.setdefault(event.venue_key, set()).add(event.venue_raw)

    venues = []
    for key in sorted(venue_raws):
        name = sorted(venue_raws[key])[0]
        venues.append(
            GraphVenue(
                id=key,
                name=name,
                compact=compact_venue(name),
                tokens=venue_token_blob(name),
            )
        )

    contains = [
        GraphEdge(
            src=chunk.document_id,
            tgt=chunk.id,
            document_id=chunk.document_id,
            source_chunk_id=chunk.id,
            chunk_index=chunk.chunk_index,
            section=chunk.section,
        )
        for chunk in graph_chunks
    ]
    describes = []
    in_games = []
    of_sport = []
    held_at = []
    for event in graph_events:
        source_chunk = infobox_chunk.get(event.document_id) or first_chunk.get(event.document_id, "")
        describes.append(
            GraphEdge(
                src=event.document_id,
                tgt=event.id,
                document_id=event.document_id,
                source_chunk_id=source_chunk,
            )
        )
        if event.games_id:
            in_games.append(
                GraphEdge(
                    src=event.id,
                    tgt=event.games_id,
                    document_id=event.document_id,
                    source_chunk_id=source_chunk,
                    year=event.year,
                    season=event.season,
                )
            )
        if event.sport:
            of_sport.append(
                GraphEdge(
                    src=event.id,
                    tgt=event.sport,
                    document_id=event.document_id,
                    source_chunk_id=source_chunk,
                    sport_raw=event.sport_raw,
                )
            )
        if event.venue_key:
            held_at.append(
                GraphEdge(
                    src=event.id,
                    tgt=event.venue_key,
                    document_id=event.document_id,
                    source_chunk_id=source_chunk,
                    venue_raw=event.venue_raw,
                )
            )

    overflow = sum(1 for event in graph_events if event.competitors_overflow)
    return GraphExport(
        documents=sorted(graph_documents, key=lambda row: row.id),
        chunks=sorted(graph_chunks, key=lambda row: row.id),
        events=sorted(graph_events, key=lambda row: row.id),
        games=sorted(games_map.values(), key=lambda row: row.id),
        sports=sorted(sport_map.values(), key=lambda row: row.id),
        venues=venues,
        contains_chunk=sorted(contains, key=lambda row: (row.src, row.tgt)),
        describes=sorted(describes, key=lambda row: (row.src, row.tgt)),
        in_games=sorted(in_games, key=lambda row: (row.src, row.tgt)),
        of_sport=sorted(of_sport, key=lambda row: (row.src, row.tgt)),
        held_at=sorted(held_at, key=lambda row: (row.src, row.tgt)),
        notes={
            "athlete_vertices": False,
            "nation_vertices": False,
            "prev_event_edges": False,
            "vector_attributes": False,
            "competitors_overflow": overflow,
            "title_only_documents": sum(1 for document in graph_documents if not document.is_event),
        },
    )


def record_to_dict(record: Any) -> dict[str, Any]:
    return asdict(record)


def _events(documents: list[ParsedDocument]) -> list[ParsedEvent]:
    return [document.event for document in documents if document.event is not None]


def _document_record(document: ParsedDocument) -> GraphDocument:
    return GraphDocument(
        id=document.doc_id,
        title=document.title or "",
        url=document.url or "",
        wikidata_qid=document.wikidata_qid or document.doc_id,
        wikipedia_pageid=int(document.wikipedia_pageid or 0),
        approx_tokens=int(document.approx_tokens) if document.approx_tokens is not None else -1,
        has_olympic_infobox=bool(document.has_olympic_infobox),
        kind=document.kind,
        rejection_reason=document.rejection_reason or "",
        is_event=document.is_event,
        event_id=document.event.event_id if document.event is not None else "",
    )


def _chunk_record(chunk: TextChunk) -> GraphChunk:
    return GraphChunk(
        id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_title=chunk.document_title or "",
        source_url=chunk.source_url or "",
        text=chunk.text,
        indexed_text=chunk.indexed_text,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        token_count=chunk.token_count,
        chunk_index=chunk.chunk_index,
        section=chunk.section or "",
        kind=chunk.kind,
        event_id=chunk.event_id or "",
        document_kind=chunk.document_kind,
    )


def _event_record(document: ParsedDocument, event: ParsedEvent) -> GraphEvent:
    gold = _medal(event, "gold")
    silver = _medal(event, "silver")
    bronze = _medal(event, "bronze")
    overflow = bool(event.competitors is not None and event.competitors > 2_147_483_647)
    date = event.date
    return GraphEvent(
        id=event.event_id,
        document_id=event.document_id,
        title=event.title or "",
        title_folded=fold_text(event.title),
        sport=event.sport or "",
        sport_raw=event.sport_raw or "",
        event_name_raw=event.event_name_raw or "",
        event_name_folded=fold_text(event.event_name_raw),
        event_name_normalized=normalize_event_name(event.event_name_raw),
        event_name_core=event_name_core(event.event_name_raw),
        event_name_gender=event_name_gender(event.event_name_raw),
        infobox_event=event.infobox_event or "",
        year=event.year or 0,
        season=event.season or "",
        games_id=event.games_id or games_id(event.year, event.season) or "",
        infobox_games_raw=event.infobox_games_raw or "",
        year_source=event.year_source or "",
        venue_raw=event.venue_raw or "",
        venue_key=event.venue_key or venue_id(event.venue_raw) or "",
        venue_compact=compact_venue(event.venue_raw),
        venue_tokens=venue_token_blob(event.venue_raw),
        date_raw=date.raw if date is not None else "",
        date_years=packed_ints(date.years if date is not None else ()),
        date_months=packed_ints(date.months if date is not None else ()),
        date_days=packed_ints(date.days if date is not None else ()),
        date_compact=compact_alnum(date.raw if date is not None else ""),
        competitors=clamp_int32(event.competitors),
        has_competitors=event.competitors is not None,
        competitors_raw=event.competitors_raw or "",
        nations=clamp_int32(event.nations),
        has_nations=event.nations is not None,
        nations_raw=event.nations_raw or "",
        gold_raw=gold[0],
        gold_noc=gold[1],
        silver_raw=silver[0],
        bronze_raw=bronze[0],
        has_gold=bool(gold[0]),
        prev_year=event.prev_year or 0,
        next_year=event.next_year or 0,
        prev_year_raw=event.prev_year_raw or "",
        next_year_raw=event.next_year_raw or "",
        flags="|".join(event.flags),
        url=document.url or "",
        competitors_overflow=overflow,
    )


def _medal(event: ParsedEvent, place: str) -> tuple[str, str]:
    for medal in event.medals:
        if medal.place == place:
            return medal.name_raw, medal.noc or ""
    return "", ""
