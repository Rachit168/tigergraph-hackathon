"""Deterministic structured solver over the Phase 1 Event table."""

from __future__ import annotations

import time
from collections.abc import Iterable

from ingestion.models import ParsedEvent, Provenance
from retrieval.structured.calendar import resolve_related_year
from retrieval.structured.index import StructuredIndex
from retrieval.structured.match import (
    date_raw_score,
    date_specificity,
    dates_overlap,
    parse_question_date,
    venue_score,
)
from retrieval.structured.models import EvidenceItem, QuerySpec, SolverResult
from retrieval.structured.question import QuestionParser


class StructuredSolver:
    def __init__(self, index: StructuredIndex) -> None:
        self.index = index
        self.parser = QuestionParser(sorted(index.sports, key=len, reverse=True))

    def solve(self, question: str, qtype: str | None = None) -> SolverResult:
        started = time.perf_counter()
        spec = self.parser.parse(question, qtype=qtype)
        result = self.solve_spec(spec)
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return result

    def solve_spec(self, spec: QuerySpec) -> SolverResult:
        if not spec.matched_template:
            return _result(spec, "unresolved", method="parse", reason="question_template_unparsed")
        dispatch = {
            "lookup_nations": self._lookup,
            "count_over_threshold": self._aggregation,
            "argmax_competitors": self._superlative,
            "previous_event_gold": self._temporal,
            "next_event_gold": self._temporal,
            "events_at_venue_date": self._multi_hop,
        }
        handler = dispatch.get(spec.operation)
        if handler is None:
            return _result(spec, "unresolved", method="parse", reason="unsupported_operation")
        return handler(spec)

    def _lookup(self, spec: QuerySpec) -> SolverResult:
        events = self.index.lookup_title(spec.event_title)
        method = "lookup_event"
        if not events:
            return _result(spec, "not_found", method=method, reason="event_title_not_found")
        if len(events) > 1:
            return _result(
                spec,
                "ambiguous",
                method=method,
                reason="multiple_title_matches",
                events=events,
                evidence=_event_evidence(events, method, field="title"),
            )
        event = events[0]
        if event.nations is None:
            return _result(
                spec,
                "unresolved",
                method=method,
                reason="missing_field:nations",
                events=events,
                evidence=_event_evidence(events, method, field="nations"),
                notes={"limitation": "data"},
            )
        return _result(
            spec,
            "supported",
            method=method,
            answer=[str(event.nations)],
            events=events,
            evidence=_event_evidence(events, method, field="nations", value=str(event.nations)),
        )

    def _aggregation(self, spec: QuerySpec) -> SolverResult:
        pool = self.index.events_of_sport_games(spec.sport, spec.year, spec.season)
        method = "count_over_threshold"
        if not pool:
            return _result(spec, "not_found", method=method, reason="no_sport_games_events")
        if spec.threshold is None:
            return _result(spec, "unresolved", method=method, reason="missing_threshold", events=pool)
        usable = [event for event in pool if event.competitors is not None]
        missing = len(pool) - len(usable)
        hits = [event for event in usable if event.competitors is not None and event.competitors > spec.threshold]
        evidence = _event_evidence(hits, method, field="competitors")
        evidence.append(
            EvidenceItem(
                evidence_id="agg-count",
                evidence_type="query_row",
                document_id=None,
                event_id=None,
                field_name="count",
                value=str(len(hits)),
                raw_text=None,
                retrieval_method=method,
            )
        )
        return _result(
            spec,
            "supported",
            method=method,
            answer=[str(len(hits))],
            events=hits,
            evidence=evidence,
            notes={
                "pool_size": len(pool),
                "usable_with_competitors": len(usable),
                "missing_competitors": missing,
            },
        )

    def _superlative(self, spec: QuerySpec) -> SolverResult:
        pool = self.index.events_of_sport_games(spec.sport, spec.year, spec.season)
        method = "argmax_competitors"
        usable = [event for event in pool if event.competitors is not None]
        if not usable:
            status = "not_found" if not pool else "unresolved"
            reason = "no_sport_games_events" if not pool else "missing_field:competitors"
            return _result(
                spec,
                status,
                method=method,
                reason=reason,
                events=pool,
                notes={"limitation": "data" if pool else None},
            )
        best = max(event.competitors or 0 for event in usable)
        winners = [event for event in usable if event.competitors == best]
        if len(winners) > 1:
            return _result(
                spec,
                "ambiguous",
                method=method,
                reason="tied_max_competitors",
                events=winners,
                evidence=_event_evidence(winners, method, field="competitors"),
                notes={"max_competitors": best, "pool_size": len(usable)},
            )
        winner = winners[0]
        return _result(
            spec,
            "supported",
            method=method,
            answer=[winner.title],
            events=winners,
            evidence=_event_evidence(winners, method, field="competitors", value=str(winner.competitors)),
            notes={"max_competitors": best, "pool_size": len(usable)},
        )

    def _temporal(self, spec: QuerySpec) -> SolverResult:
        method = spec.operation
        relation = spec.temporal_relation or "previous"
        named = self.index.events_matching_name(spec.sport, spec.named_year, spec.season, spec.event_name)
        if len(named) > 1:
            return _result(
                spec,
                "ambiguous",
                method=method,
                reason="multiple_named_year_events",
                events=named,
                evidence=_event_evidence(named, method, field="event_name"),
            )
        infobox_year = None
        named_event = named[0] if named else None
        if named_event is not None:
            infobox_year = named_event.prev_year if relation == "previous" else named_event.next_year
        target_year, year_source = resolve_related_year(
            spec.named_year or 0,
            spec.season or "",
            relation,
            infobox_year=infobox_year,
        )
        if target_year is None:
            return _result(
                spec,
                "unresolved",
                method=method,
                reason="related_year_unresolved",
                events=named,
                notes={"year_source": year_source, "limitation": "data"},
            )
        same_name = None
        if named_event is not None and named_event.event_name_raw:
            same_name = self.index.events_matching_name(
                named_event.sport,
                target_year,
                named_event.season,
                named_event.event_name_raw,
            )
        related = same_name if same_name else self.index.events_matching_name(
            spec.sport,
            target_year,
            spec.season,
            spec.event_name,
        )
        notes = {
            "named_year_hits": len(named),
            "target_year": target_year,
            "year_source": year_source,
            "same_name_hits": len(same_name or []),
            "related_hits": len(related),
        }
        if not related:
            reason = "previous_event_name_missing" if relation == "previous" else "next_event_name_missing"
            return _result(
                spec,
                "unresolved",
                method=method,
                reason=reason,
                events=named,
                notes={**notes, "limitation": "data"},
            )
        if len(related) > 1:
            return _result(
                spec,
                "ambiguous",
                method=method,
                reason="multiple_related_events",
                events=related,
                evidence=_event_evidence(related, method, field="gold"),
                notes=notes,
            )
        event = related[0]
        gold = event.gold_raw()
        if not gold:
            return _result(
                spec,
                "unresolved",
                method=method,
                reason="missing_field:gold",
                events=related,
                evidence=_event_evidence(related, method, field="gold"),
                notes={**notes, "limitation": "data"},
            )
        path_events = [item for item in (named_event, event) if item is not None]
        return _result(
            spec,
            "supported",
            method=method,
            answer=[gold],
            events=related,
            evidence=_path_evidence(path_events, method, event, gold),
            notes=notes,
        )

    def _multi_hop(self, spec: QuerySpec) -> SolverResult:
        method = "events_at_venue_date"
        question_date = parse_question_date(spec.date_text, fallback_year=spec.year)
        scored: list[tuple[int, ParsedEvent]] = []
        for event in self.index.events:
            if spec.year is not None and event.year != spec.year:
                continue
            if spec.season and event.season and event.season != spec.season:
                continue
            score = venue_score(spec.venue, event.venue_raw)
            if score:
                scored.append((score, event))
        if not scored:
            return _result(spec, "not_found", method=method, reason="venue_not_found")
        best_venue = max(score for score, _event in scored)
        venue_hits = [event for score, event in scored if score == best_venue]
        overlapping = [
            event
            for event in venue_hits
            if question_date is None or dates_overlap(event.date, question_date)
        ]
        notes = {
            "venue_best_score": best_venue,
            "venue_hits": len(venue_hits),
            "date_overlap_hits": len(overlapping),
        }
        if not overlapping:
            return _result(
                spec,
                "unresolved",
                method=method,
                reason="date_does_not_overlap",
                events=venue_hits,
                notes={**notes, "limitation": "data"},
            )
        narrowed = _narrow_venue_date(overlapping, question_date, spec.date_text)
        notes["after_date_tighten"] = len(narrowed)
        if not narrowed:
            return _result(
                spec,
                "unresolved",
                method=method,
                reason="date_does_not_overlap",
                events=overlapping,
                notes={**notes, "limitation": "data"},
            )
        if len(narrowed) > 1:
            return _result(
                spec,
                "ambiguous",
                method=method,
                reason="multiple_events_same_venue_date",
                events=narrowed,
                evidence=_event_evidence(narrowed, method, field="gold"),
                notes={**notes, "limitation": "data"},
            )
        event = narrowed[0]
        gold = event.gold_raw()
        if not gold:
            return _result(
                spec,
                "unresolved",
                method=method,
                reason="missing_field:gold",
                events=narrowed,
                evidence=_event_evidence(narrowed, method, field="gold"),
                notes={**notes, "limitation": "data"},
            )
        return _result(
            spec,
            "supported",
            method=method,
            answer=[gold],
            events=narrowed,
            evidence=_event_evidence(narrowed, method, field="gold", value=gold),
            notes=notes,
        )


def _narrow_venue_date(
    events: list[ParsedEvent],
    question_date,
    date_text: str | None,
) -> list[ParsedEvent]:
    if len(events) <= 1:
        return events
    by_days = {}
    for event in events:
        by_days.setdefault(date_specificity(event, question_date), []).append(event)
    tight = by_days[max(by_days)]
    if len(tight) == 1:
        return tight
    by_raw = {}
    for event in tight:
        by_raw.setdefault(date_raw_score(event, date_text), []).append(event)
    raw_best = by_raw[max(by_raw)]
    if max(by_raw) > 0:
        return raw_best
    return tight


def _event_evidence(
    events: Iterable[ParsedEvent],
    method: str,
    field: str,
    value: str | None = None,
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for index, event in enumerate(events):
        provenance = event.provenance.get(field)
        raw = provenance.raw_text if provenance else None
        field_value = value
        if field_value is None:
            field_value = _field_value(event, field)
        items.append(
            EvidenceItem(
                evidence_id=f"{method}:{event.event_id}:{field}:{index}",
                evidence_type="event_attribute",
                document_id=event.document_id,
                event_id=event.event_id,
                field_name=field,
                value=field_value,
                raw_text=raw,
                retrieval_method=method,
                provenance=provenance,
            )
        )
    return items


def _path_evidence(
    path_events: list[ParsedEvent],
    method: str,
    answer_event: ParsedEvent,
    gold: str,
) -> list[EvidenceItem]:
    items = _event_evidence(path_events, method, field="title")
    gold_prov = answer_event.provenance.get("gold")
    items.append(
        EvidenceItem(
            evidence_id=f"{method}:{answer_event.event_id}:gold",
            evidence_type="path",
            document_id=answer_event.document_id,
            event_id=answer_event.event_id,
            field_name="gold",
            value=gold,
            raw_text=gold_prov.raw_text if gold_prov else gold,
            retrieval_method=method,
            provenance=gold_prov or Provenance(answer_event.document_id, answer_event.title, "gold", gold),
        )
    )
    return items


def _field_value(event: ParsedEvent, field: str) -> str | None:
    if field == "nations":
        return str(event.nations) if event.nations is not None else event.nations_raw
    if field == "competitors":
        return str(event.competitors) if event.competitors is not None else event.competitors_raw
    if field == "gold":
        return event.gold_raw()
    if field == "title":
        return event.title
    if field == "event_name":
        return event.event_name_raw
    if field == "venue":
        return event.venue_raw
    return None


def _result(
    spec: QuerySpec,
    status: str,
    method: str,
    reason: str | None = None,
    answer: list[str] | None = None,
    events: list[ParsedEvent] | None = None,
    evidence: list[EvidenceItem] | None = None,
    notes: dict | None = None,
) -> SolverResult:
    return SolverResult(
        status=status,  # type: ignore[arg-type]
        spec=spec,
        method=method,
        answer=answer,
        reason=reason,
        events=events or [],
        evidence=evidence or [],
        notes=notes or {},
    )
