"""Question-to-document gold mapping for retrieval eval.

Uses public gold_doc_ids plus structured Event slots from the question text.
Does not inject answer strings into retrieval queries.
"""

from __future__ import annotations

from ingestion.models import ParseCorpusResult
from retrieval.structured.calendar import resolve_related_year
from retrieval.structured.index import StructuredIndex
from retrieval.structured.match import dates_overlap, parse_question_date, venue_score
from retrieval.structured.question import QuestionParser


def official_gold_ids(question: dict) -> list[str]:
    return [str(doc_id) for doc_id in (question.get("gold_doc_ids") or [])]


def structured_gold_ids(
    question: dict,
    index: StructuredIndex,
    parser: QuestionParser,
) -> list[str]:
    spec = parser.parse(str(question.get("question") or ""), qtype=question.get("qtype"))
    qtype = spec.qtype
    if qtype == "lookup":
        return [event.event_id for event in index.lookup_title(spec.event_title)]
    if qtype in {"aggregation", "superlative"}:
        return [
            event.event_id
            for event in index.events_of_sport_games(spec.sport, spec.year, spec.season)
        ]
    if qtype == "temporal":
        named = index.events_matching_name(spec.sport, spec.named_year, spec.season, spec.event_name)
        infobox_year = None
        if len(named) == 1:
            infobox_year = named[0].prev_year if spec.temporal_relation != "next" else named[0].next_year
        target_year, _source = resolve_related_year(
            spec.named_year or 0,
            spec.season or "",
            spec.temporal_relation or "previous",
            infobox_year=infobox_year,
        )
        related = []
        if target_year is not None:
            seed_name = named[0].event_name_raw if len(named) == 1 else spec.event_name
            seed_sport = named[0].sport if len(named) == 1 else spec.sport
            seed_season = named[0].season if len(named) == 1 else spec.season
            related = index.events_matching_name(seed_sport, target_year, seed_season, seed_name)
        ids: list[str] = []
        for event in named + related:
            if event.event_id not in ids:
                ids.append(event.event_id)
        return ids
    if qtype == "multi_hop":
        qdate = parse_question_date(spec.date_text, fallback_year=spec.year)
        scored = []
        for event in index.events:
            if spec.year is not None and event.year != spec.year:
                continue
            score = venue_score(spec.venue, event.venue_raw)
            if not score:
                continue
            if qdate is not None and event.date is not None and not dates_overlap(event.date, qdate):
                continue
            scored.append((score, event.event_id))
        if not scored:
            return []
        best = max(score for score, _doc in scored)
        return [doc_id for score, doc_id in scored if score == best]
    return []


def build_parser(corpus: ParseCorpusResult) -> tuple[StructuredIndex, QuestionParser]:
    index = StructuredIndex(corpus)
    parser = QuestionParser(sorted(index.sports, key=len, reverse=True))
    return index, parser
