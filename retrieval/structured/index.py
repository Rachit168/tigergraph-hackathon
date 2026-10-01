"""In-memory Event indexes. Only infobox-gated Event records are indexed."""

from __future__ import annotations

from collections import defaultdict

from ingestion.models import ParseCorpusResult, ParsedDocument, ParsedEvent
from ingestion.parse import normalize_sport_key
from retrieval.structured.match import event_names_match, fold_text


class StructuredIndex:
    """Lookup structures that map to later GSQL predicates."""

    def __init__(self, result: ParseCorpusResult) -> None:
        self.documents = list(result.documents)
        self.events = [document.event for document in self.documents if document.event is not None]
        self.event_by_id = {event.event_id: event for event in self.events}
        self.document_by_id = {document.doc_id: document for document in self.documents}
        self.by_title: dict[str, list[ParsedEvent]] = defaultdict(list)
        self.by_title_folded: dict[str, list[ParsedEvent]] = defaultdict(list)
        self.by_identity: dict[tuple, list[ParsedEvent]] = defaultdict(list)
        self.by_sport_games: dict[tuple, list[ParsedEvent]] = defaultdict(list)
        self.by_games: dict[tuple, list[ParsedEvent]] = defaultdict(list)
        self.sports: set[str] = set()
        for event in self.events:
            self.by_title[event.title].append(event)
            folded = fold_text(event.title)
            if folded:
                self.by_title_folded[folded].append(event)
            identity = (
                event.sport,
                event.year,
                event.season,
                fold_text(event.event_name_raw),
            )
            if event.sport and event.year is not None and event.season and event.event_name_raw:
                self.by_identity[identity].append(event)
            if event.sport and event.year is not None and event.season:
                self.by_sport_games[(event.sport, event.year, event.season)].append(event)
            if event.year is not None and event.season:
                self.by_games[(event.year, event.season)].append(event)
            if event.sport_raw:
                self.sports.add(event.sport_raw)
            if event.sport:
                self.sports.add(event.sport)

    @classmethod
    def from_documents(cls, documents: list[ParsedDocument]) -> StructuredIndex:
        return cls(ParseCorpusResult(documents=documents))

    def lookup_title(self, title: str | None) -> list[ParsedEvent]:
        if not title:
            return []
        exact = list(self.by_title.get(title, []))
        if exact:
            return exact
        return list(self.by_title_folded.get(fold_text(title), []))

    def events_of_sport_games(
        self,
        sport: str | None,
        year: int | None,
        season: str | None,
    ) -> list[ParsedEvent]:
        key = (normalize_sport_key(sport), year, season)
        return list(self.by_sport_games.get(key, []))

    def events_matching_name(
        self,
        sport: str | None,
        year: int | None,
        season: str | None,
        event_name: str | None,
    ) -> list[ParsedEvent]:
        pool = self.events_of_sport_games(sport, year, season) if sport and year and season else []
        if not pool and year is not None and season:
            pool = list(self.by_games.get((year, season), []))
            if sport:
                sport_key = normalize_sport_key(sport)
                pool = [event for event in pool if event.sport == sport_key]
        if not event_name:
            return pool
        identity_key = (
            normalize_sport_key(sport),
            year,
            season,
            fold_text(event_name),
        )
        exact = list(self.by_identity.get(identity_key, []))
        if exact:
            return exact
        return [event for event in pool if event_names_match(event_name, event.event_name_raw)]

    def document(self, doc_id: str) -> ParsedDocument | None:
        return self.document_by_id.get(doc_id)
