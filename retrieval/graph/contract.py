"""GSQL-portable structured operations over exported Event attributes.

This is the Phase 2 query contract expressed against graph records. It is the
offline reference for installed GSQL, not a substitute TigerGraph backend.
"""

from __future__ import annotations

from ingestion.graph_ids import gsql_venue_score, names_fuzzy_match, names_identity_match, packed_overlap, unpack_ints
from ingestion.graph_records import GraphEvent, GraphExport
from ingestion.models import DateSpan, ParsedEvent
from retrieval.graph.params import params_for_spec
from retrieval.graph.results import GraphQueryResult
from retrieval.structured.calendar import resolve_related_year
from retrieval.structured.match import parse_question_date
from retrieval.structured.models import QuerySpec
from retrieval.structured.solver import _narrow_venue_date


class GraphStore:
    def __init__(self, events: list[GraphEvent]) -> None:
        self.events = list(events)
        self.by_id = {event.id: event for event in self.events}

    @classmethod
    def from_export(cls, graph: GraphExport) -> GraphStore:
        return cls(graph.events)

    def execute(self, spec: QuerySpec) -> GraphQueryResult:
        if not spec.matched_template:
            return GraphQueryResult(operation=spec.operation or "unknown", status="unresolved", reason="question_template_unparsed")
        name, params = params_for_spec(spec)
        if name == "lookup_event":
            return self.lookup_event(params["title"], params["title_folded"])
        if name == "count_over_threshold":
            return self.count_over_threshold(params["sport"], params["year"], params["season"], params["threshold"])
        if name == "argmax_competitors":
            return self.argmax_competitors(params["sport"], params["year"], params["season"])
        if name == "previous_event_gold":
            return self.previous_event_gold(params, spec)
        if name == "events_at_venue_date":
            return self.events_at_venue_date(params, spec)
        return GraphQueryResult(operation=name, status="unresolved", reason="unsupported_operation")

    def lookup_event(self, title: str, title_folded: str) -> GraphQueryResult:
        exact = [event for event in self.events if event.title == title]
        hits = exact if exact else [event for event in self.events if title_folded and event.title_folded == title_folded]
        result = _from_events("lookup_event", hits)
        if not hits:
            result.status = "not_found"
            result.reason = "event_title_not_found"
            return result
        if len(hits) > 1:
            result.status = "ambiguous"
            result.reason = "multiple_title_matches"
            return result
        if not hits[0].has_nations:
            result.status = "unresolved"
            result.reason = "missing_field:nations"
            return result
        result.status = "supported"
        result.nations = [hits[0].nations]
        return result

    def count_over_threshold(self, sport: str, year: int, season: str, threshold: int) -> GraphQueryResult:
        pool = self.events_of_sport_games(sport, year, season)
        result = GraphQueryResult(operation="count_over_threshold", status="supported")
        if not pool:
            result.status = "not_found"
            result.reason = "no_sport_games_events"
            result.notes = {"pool_size": 0, "usable_with_competitors": 0, "missing_competitors": 0}
            result.count = None
            return result
        if threshold < 0:
            result.status = "unresolved"
            result.reason = "missing_threshold"
            result.event_ids = [event.id for event in pool]
            return result
        usable = [event for event in pool if event.has_competitors]
        hits = [event for event in usable if event.competitors > threshold]
        result = _from_events("count_over_threshold", hits)
        result.status = "supported"
        result.count = len(hits)
        result.notes = {
            "pool_size": len(pool),
            "usable_with_competitors": len(usable),
            "missing_competitors": len(pool) - len(usable),
        }
        return result

    def argmax_competitors(self, sport: str, year: int, season: str) -> GraphQueryResult:
        pool = self.events_of_sport_games(sport, year, season)
        usable = [event for event in pool if event.has_competitors]
        if not usable:
            status = "not_found" if not pool else "unresolved"
            reason = "no_sport_games_events" if not pool else "missing_field:competitors"
            return GraphQueryResult(
                operation="argmax_competitors",
                status=status,
                reason=reason,
                event_ids=[event.id for event in pool],
                notes={"limitation": "data" if pool else None, "pool_size": len(pool)},
            )
        best = max(event.competitors for event in usable)
        winners = [event for event in usable if event.competitors == best]
        result = _from_events("argmax_competitors", winners)
        result.max_competitors = best
        result.notes = {"max_competitors": best, "pool_size": len(usable)}
        if len(winners) > 1:
            result.status = "ambiguous"
            result.reason = "tied_max_competitors"
            return result
        result.status = "supported"
        return result

    def previous_event_gold(self, params: dict, spec: QuerySpec) -> GraphQueryResult:
        relation = params.get("relation") or "previous"
        named = self.events_matching_name(
            params["sport"],
            params["named_year"],
            params["season"],
            params["event_name_folded"],
            params["event_name_normalized"],
            params["event_name_core"],
            params["event_name_gender"],
        )
        if len(named) > 1:
            result = _from_events("previous_event_gold", named)
            result.status = "ambiguous"
            result.reason = "multiple_named_year_events"
            result.named_ids = [event.id for event in named]
            return result
        infobox_year = None
        named_event = named[0] if named else None
        if named_event is not None:
            infobox_year = named_event.prev_year if relation == "previous" else named_event.next_year
            if infobox_year == 0:
                infobox_year = None
        target_year, year_source = resolve_related_year(
            spec.named_year or 0,
            spec.season or "",
            relation,
            infobox_year=infobox_year,
        )
        if target_year is None:
            result = _from_events("previous_event_gold", named)
            result.status = "unresolved"
            result.reason = "related_year_unresolved"
            result.named_ids = [event.id for event in named]
            result.year_source = year_source
            result.notes = {"year_source": year_source, "limitation": "data"}
            return result
        same_name: list[GraphEvent] = []
        if named_event is not None and named_event.event_name_raw:
            same_name = self.events_matching_name(
                named_event.sport,
                target_year,
                named_event.season,
                named_event.event_name_folded,
                named_event.event_name_normalized,
                named_event.event_name_core,
                named_event.event_name_gender,
            )
        related = same_name if same_name else self.events_matching_name(
            params["sport"],
            target_year,
            params["season"],
            params["event_name_folded"],
            params["event_name_normalized"],
            params["event_name_core"],
            params["event_name_gender"],
        )
        notes = {
            "named_year_hits": len(named),
            "target_year": target_year,
            "year_source": year_source,
            "same_name_hits": len(same_name),
            "related_hits": len(related),
        }
        result = _from_events("previous_event_gold", related)
        result.named_ids = [event.id for event in named]
        result.target_year = target_year
        result.year_source = year_source
        result.notes = notes
        if not related:
            result.status = "unresolved"
            result.reason = "previous_event_name_missing" if relation == "previous" else "next_event_name_missing"
            result.event_ids = [event.id for event in named]
            result.notes = {**notes, "limitation": "data"}
            return result
        if len(related) > 1:
            result.status = "ambiguous"
            result.reason = "multiple_related_events"
            return result
        if not related[0].has_gold:
            result.status = "unresolved"
            result.reason = "missing_field:gold"
            result.notes = {**notes, "limitation": "data"}
            return result
        result.status = "supported"
        return result

    def events_at_venue_date(self, params: dict, spec: QuerySpec) -> GraphQueryResult:
        question_date = parse_question_date(spec.date_text, fallback_year=spec.year)
        tokens = tuple(params.get(f"tok{index}", "") for index in range(1, 9) if params.get(f"tok{index}"))
        scored: list[tuple[int, GraphEvent]] = []
        for event in self.events:
            if params["year"] and event.year != params["year"]:
                continue
            if params["season"] and event.season and event.season != params["season"]:
                continue
            score = gsql_venue_score(params["venue_compact"], tokens, event.venue_compact, event.venue_tokens)
            if score:
                scored.append((score, event))
        if not scored:
            return GraphQueryResult(operation="events_at_venue_date", status="not_found", reason="venue_not_found")
        best = max(score for score, _event in scored)
        venue_hits = [event for score, event in scored if score == best]
        overlapping = [
            event
            for event in venue_hits
            if question_date is None
            or (
                packed_overlap(event.date_years, params["q_years"])
                and packed_overlap(event.date_months, params["q_months"])
                and packed_overlap(event.date_days, params["q_days"])
            )
        ]
        notes = {
            "venue_best_score": best,
            "venue_hits": len(venue_hits),
            "date_overlap_hits": len(overlapping),
        }
        if not overlapping:
            result = _from_events("events_at_venue_date", venue_hits)
            result.status = "unresolved"
            result.reason = "date_does_not_overlap"
            result.notes = {**notes, "limitation": "data"}
            return result
        parsed = [_as_parsed(event) for event in overlapping]
        narrowed_parsed = _narrow_venue_date(parsed, question_date, spec.date_text)
        narrowed_ids = {event.event_id for event in narrowed_parsed}
        narrowed = [event for event in overlapping if event.id in narrowed_ids]
        notes["after_date_tighten"] = len(narrowed)
        if not narrowed:
            result = _from_events("events_at_venue_date", overlapping)
            result.status = "unresolved"
            result.reason = "date_does_not_overlap"
            result.notes = {**notes, "limitation": "data"}
            return result
        result = _from_events("events_at_venue_date", narrowed)
        result.notes = notes
        if len(narrowed) > 1:
            result.status = "ambiguous"
            result.reason = "multiple_events_same_venue_date"
            result.notes = {**notes, "limitation": "data"}
            return result
        if not narrowed[0].has_gold:
            result.status = "unresolved"
            result.reason = "missing_field:gold"
            result.notes = {**notes, "limitation": "data"}
            return result
        result.status = "supported"
        return result

    def events_of_sport_games(self, sport: str, year: int, season: str) -> list[GraphEvent]:
        return [
            event
            for event in self.events
            if event.sport == sport and event.year == year and event.season == season
        ]

    def events_matching_name(
        self,
        sport: str,
        year: int,
        season: str,
        folded: str,
        normalized: str,
        core: str,
        gender: str,
    ) -> list[GraphEvent]:
        pool = self.events_of_sport_games(sport, year, season)
        if not pool and year:
            pool = [event for event in self.events if event.year == year and (not season or event.season == season)]
            if sport:
                pool = [event for event in pool if event.sport == sport]
        identity = [event for event in pool if names_identity_match(folded, event.event_name_folded)]
        if identity:
            return identity
        return [
            event
            for event in pool
            if names_fuzzy_match(
                normalized,
                core,
                gender,
                event.event_name_normalized,
                event.event_name_core,
                event.event_name_gender,
            )
        ]


def interpret_installed_result(operation: str, spec: QuerySpec, payload: dict, events_by_id: dict[str, GraphEvent] | None = None) -> GraphQueryResult:
    """Map a raw installed-query payload onto the Phase 2 status contract."""
    event_ids = _as_list(payload.get("event_ids"))
    result = GraphQueryResult(
        operation=operation,
        status="supported",
        event_ids=event_ids,
        titles=_as_list(payload.get("titles")),
        gold=_as_list(payload.get("gold")),
        named_ids=_as_list(payload.get("named_ids")),
        count=payload.get("count") if payload.get("count") is not None else payload.get("hit_count"),
        max_competitors=payload.get("max_competitors"),
        target_year=payload.get("target_year"),
        year_source=payload.get("year_source"),
        notes={
            key: payload.get(key)
            for key in (
                "pool_size",
                "usable_with_competitors",
                "missing_competitors",
                "named_year_hits",
                "related_hits",
                "venue_best_score",
                "venue_hits",
                "date_overlap_hits",
            )
            if key in payload
        },
        source="tigergraph",
    )
    if operation == "lookup_event":
        return _status_lookup(result, payload)
    if operation == "count_over_threshold":
        if result.count is None:
            result.count = len(event_ids)
        if payload.get("pool_size") == 0:
            result.status = "not_found"
            result.reason = "no_sport_games_events"
            result.count = None
        elif spec.threshold is None:
            result.status = "unresolved"
            result.reason = "missing_threshold"
        else:
            result.status = "supported"
        return result
    if operation == "argmax_competitors":
        usable = payload.get("usable_with_competitors") or 0
        pool = payload.get("pool_size") or 0
        if usable == 0:
            result.status = "not_found" if pool == 0 else "unresolved"
            result.reason = "no_sport_games_events" if pool == 0 else "missing_field:competitors"
            return result
        if len(event_ids) > 1:
            result.status = "ambiguous"
            result.reason = "tied_max_competitors"
            return result
        result.status = "supported"
        return result
    if operation == "previous_event_gold":
        named_n = payload.get("named_year_hits")
        if named_n is None:
            named_n = len(result.named_ids)
        if named_n > 1:
            result.status = "ambiguous"
            result.reason = "multiple_named_year_events"
            result.event_ids = result.named_ids or result.event_ids
            return result
        if not payload.get("target_year"):
            result.status = "unresolved"
            result.reason = "related_year_unresolved"
            return result
        if not event_ids:
            relation = spec.temporal_relation or "previous"
            result.status = "unresolved"
            result.reason = "previous_event_name_missing" if relation == "previous" else "next_event_name_missing"
            result.event_ids = result.named_ids
            return result
        if len(event_ids) > 1:
            result.status = "ambiguous"
            result.reason = "multiple_related_events"
            return result
        gold = result.gold[0] if result.gold else ""
        if not gold:
            result.status = "unresolved"
            result.reason = "missing_field:gold"
            return result
        result.status = "supported"
        return result
    if operation == "events_at_venue_date":
        return _status_venue_date(result, spec, payload, events_by_id)
    return result


def _status_lookup(result: GraphQueryResult, payload: dict) -> GraphQueryResult:
    has_nations = payload.get("has_nations") or []
    nations = payload.get("nations") or []
    result.has_nations = [bool(value) for value in _as_list(has_nations)]
    result.nations = [int(value) for value in _as_list(nations)]
    if not result.event_ids:
        result.status = "not_found"
        result.reason = "event_title_not_found"
        return result
    if len(result.event_ids) > 1:
        result.status = "ambiguous"
        result.reason = "multiple_title_matches"
        return result
    present = result.has_nations[0] if result.has_nations else True
    if not present:
        result.status = "unresolved"
        result.reason = "missing_field:nations"
        return result
    result.status = "supported"
    return result


def _status_venue_date(
    result: GraphQueryResult,
    spec: QuerySpec,
    payload: dict,
    events_by_id: dict[str, GraphEvent] | None,
) -> GraphQueryResult:
    if not result.event_ids and not payload.get("venue_hits"):
        result.status = "not_found"
        result.reason = "venue_not_found"
        return result
    if not result.event_ids:
        result.status = "unresolved"
        result.reason = "date_does_not_overlap"
        result.notes = {**result.notes, "limitation": "data"}
        return result
    original_ids = list(result.event_ids)
    original_titles = list(result.titles)
    original_gold = list(result.gold)
    parsed = _venue_candidates_as_parsed(result, payload, events_by_id)
    if parsed:
        question_date = parse_question_date(spec.date_text, fallback_year=spec.year)
        narrowed_parsed = _narrow_venue_date(parsed, question_date, spec.date_text)
        index = {event_id: i for i, event_id in enumerate(original_ids)}
        keep = [item.event_id for item in narrowed_parsed if item.event_id in index]
        result.event_ids = keep
        result.titles = [
            original_titles[index[event_id]] if index[event_id] < len(original_titles) else ""
            for event_id in keep
        ]
        result.gold = [
            original_gold[index[event_id]] if index[event_id] < len(original_gold) else ""
            for event_id in keep
        ]
        if "date_overlap_hits" not in result.notes:
            result.notes["date_overlap_hits"] = len(original_ids)
        result.notes["after_date_tighten"] = len(keep)
    if not result.event_ids:
        result.status = "unresolved"
        result.reason = "date_does_not_overlap"
        return result
    if len(result.event_ids) > 1:
        result.status = "ambiguous"
        result.reason = "multiple_events_same_venue_date"
        result.notes = {**result.notes, "limitation": "data"}
        return result
    gold = result.gold[0] if result.gold else ""
    if not gold:
        result.status = "unresolved"
        result.reason = "missing_field:gold"
        result.notes = {**result.notes, "limitation": "data"}
        return result
    result.status = "supported"
    return result


def _venue_candidates_as_parsed(
    result: GraphQueryResult,
    payload: dict,
    events_by_id: dict[str, GraphEvent] | None,
) -> list[ParsedEvent]:
    if events_by_id and all(event_id in events_by_id for event_id in result.event_ids):
        return [_as_parsed(events_by_id[event_id]) for event_id in result.event_ids]
    date_raw = _as_list(payload.get("date_raw"))
    if not date_raw or len(date_raw) != len(result.event_ids):
        return []
    date_years = _as_list(payload.get("date_years"))
    date_months = _as_list(payload.get("date_months"))
    date_days = _as_list(payload.get("date_days"))
    parsed: list[ParsedEvent] = []
    for index, event_id in enumerate(result.event_ids):
        raw = str(date_raw[index] or "")
        date = None
        if raw:
            date = DateSpan(
                raw=raw,
                source_field="date",
                years=unpack_ints(_packed_field(date_years[index] if index < len(date_years) else "")),
                months=unpack_ints(_packed_field(date_months[index] if index < len(date_months) else "")),
                days=unpack_ints(_packed_field(date_days[index] if index < len(date_days) else "")),
            )
        parsed.append(
            ParsedEvent(
                event_id=str(event_id),
                document_id=str(event_id),
                title=result.titles[index] if index < len(result.titles) else str(event_id),
                sport_raw=None,
                sport=None,
                event_name_raw=None,
                infobox_event=None,
                year=None,
                season=None,
                games_id=None,
                infobox_games_raw=None,
                year_source="payload",
                venue_raw=None,
                venue_key=None,
                date=date,
                competitors_raw=None,
                competitors=None,
                nations_raw=None,
                nations=None,
                medals=[],
                prev_year_raw=None,
                prev_year=None,
                next_year_raw=None,
                next_year=None,
            )
        )
    return parsed


def compare_solver_to_graph(solver_result, graph_result: GraphQueryResult) -> dict:
    solver_ids = sorted(event.event_id for event in solver_result.events)
    graph_ids = sorted(graph_result.event_ids)
    solver_count = None
    if solver_result.method == "count_over_threshold" and solver_result.status == "supported":
        solver_count = len(solver_result.events)
        if solver_result.answer:
            solver_count = int(solver_result.answer[0])
    mismatches: list[str] = []
    if solver_result.status != graph_result.status:
        mismatches.append("status")
    if (solver_result.reason or None) != (graph_result.reason or None):
        mismatches.append("reason")
    if solver_ids != graph_ids:
        mismatches.append("event_ids")
    if solver_result.method == "count_over_threshold" and solver_count != graph_result.count:
        mismatches.append("count")
    solver_target = solver_result.notes.get("target_year") if solver_result.notes else None
    if solver_target is not None and graph_result.target_year not in {None, solver_target}:
        mismatches.append("target_year")
    return {
        "ok": not mismatches,
        "mismatches": mismatches,
        "solver": {
            "status": solver_result.status,
            "reason": solver_result.reason,
            "event_ids": solver_ids,
            "count": solver_count,
            "target_year": solver_target,
        },
        "graph": graph_result.to_compare_dict(),
    }


def _from_events(operation: str, events: list[GraphEvent]) -> GraphQueryResult:
    return GraphQueryResult(
        operation=operation,
        status="supported",
        event_ids=[event.id for event in events],
        titles=[event.title for event in events],
        gold=[event.gold_raw for event in events],
        nations=[event.nations for event in events],
        has_nations=[event.has_nations for event in events],
        competitors=[event.competitors for event in events],
    )


def _as_parsed(event: GraphEvent) -> ParsedEvent:
    date = None
    if event.date_raw:
        date = DateSpan(
            raw=event.date_raw,
            source_field="date",
            years=unpack_ints(event.date_years),
            months=unpack_ints(event.date_months),
            days=unpack_ints(event.date_days),
        )
    parsed = ParsedEvent(
        event_id=event.id,
        document_id=event.document_id,
        title=event.title,
        sport_raw=event.sport_raw or None,
        sport=event.sport or None,
        event_name_raw=event.event_name_raw or None,
        infobox_event=event.infobox_event or None,
        year=event.year or None,
        season=event.season or None,
        games_id=event.games_id or None,
        infobox_games_raw=event.infobox_games_raw or None,
        year_source=event.year_source,
        venue_raw=event.venue_raw or None,
        venue_key=event.venue_key or None,
        date=date,
        competitors_raw=event.competitors_raw or None,
        competitors=event.competitors if event.has_competitors else None,
        nations_raw=event.nations_raw or None,
        nations=event.nations if event.has_nations else None,
        medals=[],
        prev_year_raw=event.prev_year_raw or None,
        prev_year=event.prev_year or None,
        next_year_raw=event.next_year_raw or None,
        next_year=event.next_year or None,
    )
    return parsed


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _packed_field(value) -> str:
    if value is None:
        return ""
    return str(value)


