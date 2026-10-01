"""Deterministic name, venue, and date matching for Event lookup."""

from __future__ import annotations

import re
import unicodedata

from ingestion.models import DateSpan, ParsedEvent
from ingestion.parse import parse_date_span

GENDER_MAP = {
    "men": "men",
    "man's": "men",
    "mens": "men",
    "men's": "men",
    "male": "men",
    "women": "women",
    "woman": "women",
    "womens": "women",
    "women's": "women",
    "female": "women",
    "mixed": "mixed",
}

UNIT_MAP = {
    "metres": "metre",
    "meters": "metre",
    "meter": "metre",
    "kilometres": "kilometre",
    "kilometers": "kilometre",
    "kilometer": "kilometre",
    "kms": "km",
}

STOP_VENUE = {"the", "and", "at", "of", "olympic", "olympics", "centre", "center"}


def fold_text(value: str | None) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))
    text = text.replace("–", " ").replace("—", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def compact_alnum(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", fold_text(value))


def normalize_event_name(value: str | None) -> str:
    text = fold_text(value)
    text = text.replace(",", "")
    text = text.replace("'", "").replace("’", "")
    for source, dest in UNIT_MAP.items():
        text = re.sub(rf"\b{re.escape(source)}\b", dest, text)
    return re.sub(r"\s+", " ", text).strip()


def name_tokens(value: str | None) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9+]+", normalize_event_name(value)))


def gender_of(tokens: tuple[str, ...]) -> str | None:
    for token in tokens:
        mapped = GENDER_MAP.get(token)
        if mapped:
            return mapped
    joined = " ".join(tokens)
    for raw, mapped in GENDER_MAP.items():
        if raw in joined.split():
            return mapped
    return None


def event_names_match(query: str | None, event_name: str | None) -> bool:
    if not query or not event_name:
        return False
    query_norm = normalize_event_name(query)
    event_norm = normalize_event_name(event_name)
    if query_norm == event_norm:
        return True
    query_tokens = name_tokens(query)
    event_tokens = name_tokens(event_name)
    if not query_tokens or not event_tokens:
        return False
    query_gender = gender_of(query_tokens)
    event_gender = gender_of(event_tokens)
    if query_gender and event_gender and query_gender != event_gender:
        return False

    def stripped(tokens: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(token for token in tokens if token not in GENDER_MAP)

    query_core = stripped(query_tokens)
    event_core = stripped(event_tokens)
    if query_core == event_core:
        return True
    return False


def venue_score(query: str | None, venue_raw: str | None) -> int:
    if not query or not venue_raw:
        return 0
    query_compact = compact_alnum(query)
    venue_compact = compact_alnum(venue_raw)
    if not query_compact or not venue_compact:
        return 0
    if query_compact == venue_compact:
        return 100
    if query_compact in venue_compact:
        return 70
    if venue_compact in query_compact:
        return 60
    query_tokens = {token for token in name_tokens(query) if token not in STOP_VENUE}
    venue_tokens = set(name_tokens(venue_raw))
    if query_tokens and query_tokens <= venue_tokens:
        return 40
    return 0


def parse_question_date(date_text: str | None, fallback_year: int | None = None) -> DateSpan | None:
    if not date_text:
        return None
    cleaned = re.sub(r"\bat the \d{4} (?:Summer|Winter) Olympics\b", "", date_text, flags=re.I)
    return parse_date_span(cleaned.strip() or date_text, "question", fallback_year=fallback_year)


def dates_overlap(event_date: DateSpan | None, question_date: DateSpan | None) -> bool:
    if event_date is None or question_date is None:
        return False
    months_ok = (
        not event_date.months
        or not question_date.months
        or not set(event_date.months).isdisjoint(question_date.months)
    )
    days_ok = (
        not event_date.days
        or not question_date.days
        or not set(event_date.days).isdisjoint(question_date.days)
    )
    years_ok = (
        not event_date.years
        or not question_date.years
        or not set(event_date.years).isdisjoint(question_date.years)
    )
    return months_ok and days_ok and years_ok


def date_specificity(event: ParsedEvent, question_date: DateSpan | None) -> int:
    if event.date is None or question_date is None:
        return 0
    event_days = set(event.date.days)
    question_days = set(question_date.days)
    if event_days and question_days and event_days == question_days:
        return 3
    if event_days and question_days and question_days <= event_days:
        return 2
    if dates_overlap(event.date, question_date):
        return 1
    return 0


def date_raw_score(event: ParsedEvent, date_text: str | None) -> int:
    if event.date is None or not date_text:
        return 0
    question_compact = compact_alnum(date_text)
    event_compact = compact_alnum(event.date.raw)
    if not question_compact or not event_compact:
        return 0
    if question_compact == event_compact:
        return 3
    if question_compact in event_compact or event_compact in question_compact:
        return 2
    return 0


def extract_games_from_text(text: str) -> tuple[int | None, str | None]:
    match = re.search(r"\b(\d{4})\s+(Summer|Winter)\s+Olympics\b", text, flags=re.I)
    if not match:
        return None, None
    season = "Summer" if match.group(2).casefold() == "summer" else "Winter"
    return int(match.group(1)), season
