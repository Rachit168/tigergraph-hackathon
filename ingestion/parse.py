"""Deterministic Olympic event parser.

Event records are created only when an ``[Infobox Olympic event]`` block is
present. Olympic-looking titles without that infobox remain document-only.
Competitors and nations are taken only from infobox fields, never from body
tables. No LLM, network, or TigerGraph calls.
"""

from __future__ import annotations

import re
import unicodedata

from ingestion.models import (
    DateSpan,
    Medal,
    ParsedDocument,
    ParsedEvent,
    Provenance,
    TitleParse,
)

INFOBOX_HEADER_RE = re.compile(r"\[Infobox Olympic event\b[^\]]*\]", re.IGNORECASE)
INFOBOX_LINE_RE = re.compile(r"^[ \t]{2,}(?P<key>[^:\n]+):\s?(?P<value>.*)$")
TITLE_RE = re.compile(
    r"^(?P<sport>.+?) at the (?P<year>\d{4}) (?P<season>Summer|Winter) Olympics"
    r"(?:\s+[–—-]\s+(?P<event>.+))?$",
    re.UNICODE,
)
GAMES_RE = re.compile(r"(?P<year>\d{4})\s+(?P<season>Summer|Winter)", re.IGNORECASE)
YEAR_RE = re.compile(r"\b((?:18|19|20)\d{2})\b")
INT_RE = re.compile(r"-?\d+")
NOC_RE = re.compile(r"^[A-Za-z]{3}$")

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

MEDAL_SPECS = (
    ("gold", "gold", "goldNOC"),
    ("silver", "silver", "silverNOC"),
    ("bronze", "bronze", "bronzeNOC"),
    ("gold2", "gold2", "goldNOC2"),
    ("silver2", "silver2", "silverNOC2"),
    ("bronze2", "bronze2", "bronzeNOC2"),
    ("silver3", "silver3", "silverNOC3"),
)

def parse_title(title: str) -> TitleParse:
    raw = (title or "").strip()
    match = TITLE_RE.match(raw)
    if not match:
        return TitleParse(raw=raw, sport=None, year=None, season=None, event_name=None, parsed=False)
    sport = match.group("sport").strip() or None
    event_name = match.group("event")
    event_name = event_name.strip() if event_name else None
    return TitleParse(
        raw=raw,
        sport=sport,
        year=int(match.group("year")),
        season=_title_case_season(match.group("season")),
        event_name=event_name,
        parsed=True,
    )


def looks_like_olympic_event_title(title: str) -> bool:
    parsed = parse_title(title)
    return parsed.parsed and parsed.event_name is not None


def extract_infobox(text: str) -> tuple[bool, dict[str, str], str | None]:
    """Return (has_header, fields, header_quote)."""
    if not text:
        return False, {}, None
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    header = INFOBOX_HEADER_RE.search(normalized)
    if not header:
        return False, {}, None
    start = header.end()
    if start < len(normalized) and normalized[start] == "\n":
        start += 1
    fields: dict[str, str] = {}
    for line in normalized[start:].split("\n"):
        if not line.strip():
            break
        match = INFOBOX_LINE_RE.match(line)
        if not match:
            break
        key = match.group("key").strip()
        value = match.group("value").strip()
        if key:
            fields[key] = value
    return True, fields, header.group(0)


def parse_games_field(raw: str | None) -> tuple[int | None, str | None]:
    if not raw:
        return None, None
    match = GAMES_RE.search(raw)
    if not match:
        return None, None
    return int(match.group("year")), _title_case_season(match.group("season"))


def parse_year_value(raw: str | None) -> int | None:
    if not raw:
        return None
    match = YEAR_RE.search(raw.strip())
    if not match:
        return None
    return int(match.group(1))


def parse_int_value(raw: str | None) -> int | None:
    if not raw:
        return None
    match = INT_RE.search(raw.strip())
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def normalize_venue_key(raw: str | None) -> str | None:
    if not raw:
        return None
    text = unicodedata.normalize("NFKC", raw)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text or None


def normalize_sport_key(raw: str | None) -> str | None:
    if not raw:
        return None
    return re.sub(r"\s+", " ", raw.strip()).casefold() or None


_DATE_PHRASES = (
    (re.compile(r"\bnew\s+year'?s\s+day\b", re.IGNORECASE), "1 January"),
)


def parse_date_span(raw: str | None, source_field: str, fallback_year: int | None = None) -> DateSpan | None:
    if not raw or not raw.strip():
        return None
    original = raw.strip()
    text = original
    for pattern, replacement in _DATE_PHRASES:
        text = pattern.sub(replacement, text)
    years = [int(match.group(1)) for match in YEAR_RE.finditer(text)]
    months: list[int] = []
    days: list[int] = []
    for start, end in re.findall(r"\b(\d{1,2})\s*(?:to|–|—|-)\s*(\d{1,2})\b", text, flags=re.IGNORECASE):
        left, right = int(start), int(end)
        if 1 <= left <= 31 and 1 <= right <= 31 and left <= right:
            days.extend(range(left, right + 1))
    tokens = re.findall(r"[A-Za-z]+|\d+", text)
    for token in tokens:
        month = MONTHS.get(token.casefold())
        if month is not None:
            months.append(month)
            continue
        if token.isdigit():
            number = int(token)
            if 1000 <= number <= 2100:
                continue
            if 1 <= number <= 31:
                days.append(number)
    if fallback_year is not None and not years:
        years = [fallback_year]
    return DateSpan(
        raw=raw.strip(),
        source_field=source_field,
        years=tuple(dict.fromkeys(years)),
        months=tuple(dict.fromkeys(months)),
        days=tuple(dict.fromkeys(days)),
    )


def parse_noc(raw: str | None) -> str | None:
    if not raw:
        return None
    value = raw.strip()
    if NOC_RE.match(value):
        return value.upper()
    return None


def parse_document(record: dict) -> ParsedDocument:
    doc_id = str(record.get("doc_id") or "").strip()
    title = str(record.get("title") or "")
    text = str(record.get("text") or "")
    title_parse = parse_title(title)
    has_infobox, infobox, _header = extract_infobox(text)
    olympic_title = looks_like_olympic_event_title(title)
    flags: list[str] = []

    if olympic_title:
        flags.append("olympic_event_title")
    elif title_parse.parsed:
        flags.append("games_overview_title")

    if not has_infobox:
        reason = "no_olympic_event_infobox"
        if olympic_title:
            reason = "olympic_title_without_infobox"
            flags.append("rejected_event_title")
        return ParsedDocument(
            doc_id=doc_id,
            title=title,
            url=record.get("url"),
            wikidata_qid=record.get("wikidata_qid"),
            wikipedia_pageid=_optional_int(record.get("wikipedia_pageid")),
            approx_tokens=_optional_int(record.get("approx_tokens")),
            text=text,
            has_olympic_infobox=False,
            title_parse=title_parse,
            infobox={},
            kind="document",
            rejection_reason=reason,
            event=None,
            flags=tuple(flags),
        )

    event = _event_from_infobox(doc_id, title, title_parse, infobox)
    flags.extend(event.flags)
    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        url=record.get("url"),
        wikidata_qid=record.get("wikidata_qid"),
        wikipedia_pageid=_optional_int(record.get("wikipedia_pageid")),
        approx_tokens=_optional_int(record.get("approx_tokens")),
        text=text,
        has_olympic_infobox=True,
        title_parse=title_parse,
        infobox=infobox,
        kind="event",
        rejection_reason=None,
        event=event,
        flags=tuple(dict.fromkeys(flags)),
    )


def _event_from_infobox(
    doc_id: str,
    title: str,
    title_parse: TitleParse,
    infobox: dict[str, str],
) -> ParsedEvent:
    flags: list[str] = []
    infobox_games_raw = _first_present(infobox, ("games",))
    infobox_year, infobox_season = parse_games_field(infobox_games_raw)

    year = title_parse.year
    season = title_parse.season
    year_source = "title" if year is not None else "missing"
    if year is None and infobox_year is not None:
        year = infobox_year
        season = infobox_season
        year_source = "infobox"
    elif (
        title_parse.year is not None
        and infobox_year is not None
        and (title_parse.year != infobox_year or title_parse.season != infobox_season)
    ):
        flags.append("title_infobox_games_conflict")
        year_source = "title_preferred"

    if not title_parse.parsed:
        flags.append("infobox_without_title_parse")

    venue_raw = _first_present(infobox, ("venue", "venues"))
    date_raw, date_field = _date_field(infobox)
    competitors_raw = _first_present(infobox, ("competitors",))
    nations_raw = _first_present(infobox, ("nations",))
    prev_raw = _first_present(infobox, ("prev",))
    next_raw = _first_present(infobox, ("next",))
    infobox_event = _first_present(infobox, ("event",))

    competitors = parse_int_value(competitors_raw)
    nations = parse_int_value(nations_raw)
    if competitors_raw and competitors is None:
        flags.append("malformed_competitors")
    if nations_raw and nations is None:
        flags.append("malformed_nations")
    if competitors_raw and competitors is not None and not competitors_raw.strip().lstrip("-").isdigit():
        flags.append("competitors_has_extra_text")
    if nations_raw and nations is not None and not nations_raw.strip().lstrip("-").isdigit():
        flags.append("nations_has_extra_text")

    date = parse_date_span(date_raw, date_field or "date", fallback_year=year)
    provenance = _build_provenance(doc_id, title, title_parse, infobox, venue_raw, date_raw, date_field)
    medals = _parse_medals(doc_id, title, infobox)

    sport_raw = title_parse.sport
    event_name_raw = title_parse.event_name or infobox_event
    games_id = f"{year}_{season}" if year is not None and season else None

    if not venue_raw:
        flags.append("missing_venue")
    if date is None:
        flags.append("missing_date")
    if competitors is None:
        flags.append("missing_competitors")
    if nations is None:
        flags.append("missing_nations")
    if not medals:
        flags.append("missing_gold")
    elif all(medal.place != "gold" for medal in medals):
        flags.append("missing_gold")

    return ParsedEvent(
        event_id=doc_id,
        document_id=doc_id,
        title=title,
        sport_raw=sport_raw,
        sport=normalize_sport_key(sport_raw),
        event_name_raw=event_name_raw,
        infobox_event=infobox_event,
        year=year,
        season=season,
        games_id=games_id,
        infobox_games_raw=infobox_games_raw,
        year_source=year_source,
        venue_raw=venue_raw,
        venue_key=normalize_venue_key(venue_raw),
        date=date,
        competitors_raw=competitors_raw,
        competitors=competitors,
        nations_raw=nations_raw,
        nations=nations,
        medals=medals,
        prev_year_raw=prev_raw,
        prev_year=parse_year_value(prev_raw),
        next_year_raw=next_raw,
        next_year=parse_year_value(next_raw),
        flags=tuple(flags),
        provenance=provenance,
    )


def _parse_medals(doc_id: str, title: str, infobox: dict[str, str]) -> list[Medal]:
    medals: list[Medal] = []
    for place, name_key, noc_key in MEDAL_SPECS:
        name_raw = infobox.get(name_key)
        if not name_raw or not name_raw.strip():
            continue
        noc_raw = infobox.get(noc_key)
        medals.append(
            Medal(
                place=place,
                name_raw=name_raw.strip(),
                noc_raw=noc_raw.strip() if noc_raw else None,
                noc=parse_noc(noc_raw),
                name_provenance=Provenance(doc_id, title, name_key, name_raw.strip()),
                noc_provenance=(
                    Provenance(doc_id, title, noc_key, noc_raw.strip()) if noc_raw and noc_raw.strip() else None
                ),
            )
        )
    return medals


def _build_provenance(
    doc_id: str,
    title: str,
    title_parse: TitleParse,
    infobox: dict[str, str],
    venue_raw: str | None,
    date_raw: str | None,
    date_field: str | None,
) -> dict[str, Provenance]:
    provenance: dict[str, Provenance] = {
        "title": Provenance(doc_id, title, "title", title),
    }
    if title_parse.parsed:
        provenance["sport"] = Provenance(doc_id, title, "title", title)
        provenance["year"] = Provenance(doc_id, title, "title", title)
        provenance["season"] = Provenance(doc_id, title, "title", title)
        if title_parse.event_name:
            provenance["event_name"] = Provenance(doc_id, title, "title", title_parse.event_name)
    mapping = {
        "infobox_event": "event",
        "infobox_games": "games",
        "competitors": "competitors",
        "nations": "nations",
        "prev_year": "prev",
        "next_year": "next",
        "gold": "gold",
        "goldNOC": "goldNOC",
    }
    for dest, src in mapping.items():
        raw = infobox.get(src)
        if raw:
            provenance[dest] = Provenance(doc_id, title, src, raw)
    if venue_raw:
        field_name = "venue" if "venue" in infobox else "venues"
        provenance["venue"] = Provenance(doc_id, title, field_name, venue_raw)
    if date_raw and date_field:
        provenance["date"] = Provenance(doc_id, title, date_field, date_raw)
    return provenance


def _date_field(infobox: dict[str, str]) -> tuple[str | None, str | None]:
    if infobox.get("date"):
        return infobox["date"], "date"
    if infobox.get("dates"):
        return infobox["dates"], "dates"
    return None, None


def _first_present(infobox: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = infobox.get(key)
        if value and value.strip():
            return value.strip()
    return None


def _title_case_season(season: str) -> str:
    lowered = season.strip().casefold()
    if lowered == "summer":
        return "Summer"
    if lowered == "winter":
        return "Winter"
    return season.strip()


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def previous_olympiad_year(year: int, season: str) -> int | None:
    """Calendar fallback only. Not used to invent Event.competitors."""
    if season == "Summer":
        return year - 4
    if season == "Winter":
        if year == 1994:
            return 1992
        if year == 1992:
            return 1988
        return year - 4
    return None
