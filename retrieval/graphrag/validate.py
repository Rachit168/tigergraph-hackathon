"""Argument construction and validation for GraphRAG retrieval operations."""

from __future__ import annotations

from retrieval.graph.params import params_for_spec
from retrieval.structured.models import QuerySpec

SUPPORTED_SPEC_OPERATIONS = frozenset(
    {
        "lookup_nations",
        "count_over_threshold",
        "argmax_competitors",
        "previous_event_gold",
        "next_event_gold",
        "events_at_venue_date",
    }
)


def packed_event_ids(event_ids: list[str] | tuple[str, ...] | None) -> str:
    cleaned = [str(event_id).strip() for event_id in (event_ids or []) if str(event_id).strip()]
    if not cleaned:
        return ""
    return "|" + "|".join(cleaned) + "|"


def chunk_query_params(event_ids: list[str], max_extra: int = 0) -> dict:
    if max_extra < 0:
        raise ValueError("max_extra must be >= 0")
    return {"event_ids": packed_event_ids(event_ids), "max_extra": int(max_extra)}


def neighborhood_params(event_id: str | None) -> dict:
    return {"event_id": (event_id or "").strip()}


def validate_spec(spec: QuerySpec) -> str | None:
    if not spec.matched_template or spec.operation not in SUPPORTED_SPEC_OPERATIONS:
        return "question_template_unparsed"
    if spec.operation == "lookup_nations":
        if not (spec.event_title or "").strip():
            return "missing_event_title"
        return None
    if spec.operation == "count_over_threshold":
        if not (spec.sport or "").strip() or spec.year is None or not (spec.season or "").strip():
            return "missing_sport_games_args"
        return None
    if spec.operation == "argmax_competitors":
        if not (spec.sport or "").strip() or spec.year is None or not (spec.season or "").strip():
            return "missing_sport_games_args"
        return None
    if spec.operation in {"previous_event_gold", "next_event_gold"}:
        if not (spec.sport or "").strip() or spec.named_year is None:
            return "missing_temporal_args"
        if not (spec.event_name or spec.event_desc or "").strip():
            return "missing_event_name"
        return None
    if spec.operation == "events_at_venue_date":
        if not (spec.venue or "").strip() or not (spec.date_text or "").strip():
            return "missing_venue_date_args"
        return None
    return "unsupported_operation"


def validate_event_id(event_id: str | None) -> str | None:
    if not (event_id or "").strip():
        return "missing_event_id"
    return None


def spec_query_binding(spec: QuerySpec) -> tuple[str, dict]:
    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    return params_for_spec(spec)
