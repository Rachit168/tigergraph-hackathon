"""Olympiad calendar helpers. These do not invent Event attributes."""

from __future__ import annotations

from ingestion.parse import previous_olympiad_year


def next_olympiad_year(year: int, season: str) -> int | None:
    if season == "Summer":
        return year + 4
    if season == "Winter":
        if year == 1992:
            return 1994
        return year + 4
    return None


def resolve_related_year(
    named_year: int,
    season: str,
    relation: str,
    infobox_year: int | None = None,
) -> tuple[int | None, str]:
    """Prefer the event infobox prev/next year; fall back to the calendar."""
    if infobox_year is not None:
        return infobox_year, "infobox"
    if relation == "previous":
        return previous_olympiad_year(named_year, season), "calendar"
    if relation == "next":
        return next_olympiad_year(named_year, season), "calendar"
    return None, "unknown"
