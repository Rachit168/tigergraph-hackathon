"""Encode QuerySpec into installed-query parameters. No ad-hoc GSQL."""

from __future__ import annotations

from ingestion.graph_ids import compact_venue, matching_keys, packed_ints, query_venue_tokens
from ingestion.parse import normalize_sport_key
from retrieval.structured.match import fold_text, parse_question_date
from retrieval.structured.models import QuerySpec

TOKEN_SLOTS = 8


def params_for_spec(spec: QuerySpec) -> tuple[str, dict]:
    operation = spec.operation
    if operation == "lookup_nations":
        return "lookup_event", lookup_params(spec.event_title)
    if operation == "count_over_threshold":
        return "count_over_threshold", count_params(spec)
    if operation == "argmax_competitors":
        return "argmax_competitors", argmax_params(spec)
    if operation in {"previous_event_gold", "next_event_gold"}:
        return "previous_event_gold", previous_params(spec)
    if operation == "events_at_venue_date":
        return "events_at_venue_date", venue_date_params(spec)
    raise ValueError(f"unsupported operation: {operation}")


def lookup_params(title: str | None) -> dict:
    return {"title": title or "", "title_folded": fold_text(title)}


def count_params(spec: QuerySpec) -> dict:
    return {
        "sport": normalize_sport_key(spec.sport) or "",
        "year": int(spec.year or 0),
        "season": spec.season or "",
        "threshold": int(spec.threshold if spec.threshold is not None else -1),
    }


def argmax_params(spec: QuerySpec) -> dict:
    return {
        "sport": normalize_sport_key(spec.sport) or "",
        "year": int(spec.year or 0),
        "season": spec.season or "",
    }


def previous_params(spec: QuerySpec) -> dict:
    keys = matching_keys(spec.event_name)
    return {
        "sport": normalize_sport_key(spec.sport) or "",
        "named_year": int(spec.named_year or 0),
        "season": spec.season or "",
        "event_name_folded": keys["event_name_folded"],
        "event_name_normalized": keys["event_name_normalized"],
        "event_name_core": keys["event_name_core"],
        "event_name_gender": keys["event_name_gender"],
        "relation": spec.temporal_relation or "previous",
    }


def venue_date_params(spec: QuerySpec) -> dict:
    question_date = parse_question_date(spec.date_text, fallback_year=spec.year)
    tokens = list(query_venue_tokens(spec.venue))[:TOKEN_SLOTS]
    while len(tokens) < TOKEN_SLOTS:
        tokens.append("")
    payload = {
        "year": int(spec.year or 0),
        "season": spec.season or "",
        "venue_compact": compact_venue(spec.venue),
        "q_years": packed_ints(question_date.years if question_date is not None else ()),
        "q_months": packed_ints(question_date.months if question_date is not None else ()),
        "q_days": packed_ints(question_date.days if question_date is not None else ()),
    }
    for index, token in enumerate(tokens, start=1):
        payload[f"tok{index}"] = token
    return payload
