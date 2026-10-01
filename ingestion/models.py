"""Parsed corpus records for Phase 1 (no TigerGraph schema)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True)
class Provenance:
    """Where a single extracted value came from."""

    document_id: str
    page_title: str
    field_name: str
    raw_text: str


@dataclass(frozen=True)
class TitleParse:
    """Structured fields parsed from a Wikipedia page title."""

    raw: str
    sport: str | None
    year: int | None
    season: str | None
    event_name: str | None
    parsed: bool


@dataclass(frozen=True)
class DateSpan:
    """Conservative date representation. Missing parts stay None."""

    raw: str
    source_field: str
    years: tuple[int, ...] = ()
    months: tuple[int, ...] = ()
    days: tuple[int, ...] = ()

    @property
    def year(self) -> int | None:
        return self.years[0] if len(set(self.years)) == 1 else None


@dataclass(frozen=True)
class Medal:
    place: str
    name_raw: str
    noc_raw: str | None
    noc: str | None
    name_provenance: Provenance
    noc_provenance: Provenance | None = None


@dataclass
class ParsedEvent:
    """Structured Olympic event extracted from an infobox document."""

    event_id: str
    document_id: str
    title: str
    sport_raw: str | None
    sport: str | None
    event_name_raw: str | None
    infobox_event: str | None
    year: int | None
    season: str | None
    games_id: str | None
    infobox_games_raw: str | None
    year_source: str
    venue_raw: str | None
    venue_key: str | None
    date: DateSpan | None
    competitors_raw: str | None
    competitors: int | None
    nations_raw: str | None
    nations: int | None
    medals: list[Medal]
    prev_year_raw: str | None
    prev_year: int | None
    next_year_raw: str | None
    next_year: int | None
    flags: tuple[str, ...] = ()
    provenance: dict[str, Provenance] = field(default_factory=dict)

    def gold_raw(self) -> str | None:
        for medal in self.medals:
            if medal.place == "gold":
                return medal.name_raw
        return None

    def field_status(self) -> dict[str, str]:
        def status(present: bool) -> str:
            return "present" if present else "missing"

        return {
            "sport": status(self.sport is not None),
            "year": status(self.year is not None),
            "season": status(self.season is not None),
            "event_name": status(bool(self.event_name_raw)),
            "venue": status(bool(self.venue_raw)),
            "date": status(self.date is not None),
            "competitors": status(self.competitors is not None),
            "nations": status(self.nations is not None),
            "gold": status(self.gold_raw() is not None),
            "prev_year": status(self.prev_year is not None),
            "next_year": status(self.next_year is not None),
        }


@dataclass
class ParsedDocument:
    """Every corpus row becomes a ParsedDocument. Events are optional."""

    doc_id: str
    title: str
    url: str | None
    wikidata_qid: str | None
    wikipedia_pageid: int | None
    approx_tokens: int | None
    text: str
    has_olympic_infobox: bool
    title_parse: TitleParse
    infobox: dict[str, str]
    kind: str
    rejection_reason: str | None = None
    event: ParsedEvent | None = None
    flags: tuple[str, ...] = ()

    @property
    def is_event(self) -> bool:
        return self.kind == "event" and self.event is not None


@dataclass
class ParseCorpusResult:
    documents: list[ParsedDocument]

    def events(self) -> Iterator[ParsedEvent]:
        for document in self.documents:
            if document.event is not None:
                yield document.event

    def event_by_id(self) -> dict[str, ParsedEvent]:
        return {event.event_id: event for event in self.events()}

    def document_by_id(self) -> dict[str, ParsedDocument]:
        return {document.doc_id: document for document in self.documents}

    def summary(self) -> dict[str, Any]:
        events = [document for document in self.documents if document.is_event]
        return {
            "total_documents": len(self.documents),
            "event_documents": len(events),
            "document_only": len(self.documents) - len(events),
        }


@dataclass(frozen=True)
class ChunkingConfig:
    """Deterministic infobox / lead / section chunking knobs."""

    max_chars: int = 1400
    min_chars: int = 60
    hard_max_chars: int = 1800
    include_title_in_index: bool = True


@dataclass
class TextChunk:
    """A retrievable text span. Distinct from Event records."""

    chunk_id: str
    document_id: str
    document_title: str
    source_url: str | None
    text: str
    indexed_text: str
    start_char: int
    end_char: int
    token_count: int
    chunk_index: int
    section: str
    kind: str
    event_id: str | None = None
    document_kind: str = "document"

    @property
    def is_event_document(self) -> bool:
        return self.event_id is not None
