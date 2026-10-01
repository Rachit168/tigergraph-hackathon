"""Deterministic question → QuerySpec parser. No LLM, no question ids.

Closed synonym and shape classes around the five existing operations.
The operations themselves are unchanged.
"""

from __future__ import annotations

import re
import unicodedata

from ingestion.parse import normalize_sport_key, previous_olympiad_year
from retrieval.structured.match import extract_games_from_text
from retrieval.structured.models import QuerySpec

# Closed synonym classes. These must make sense for an arbitrary future
# Olympic question of the same operation, not a specific benchmark item.
NATION_PLURAL = r"(?:nations|countries|nocs)"
NATION_SINGULAR = r"(?:nation|country|noc)"
COMPETE_VERB = r"(?:competed|participated|took\s+part|entered)"
MORE_THAN = r"(?:more\s+than|over|greater\s+than)"
MOST_COMPETITORS = (
    r"(?:the\s+highest\s+number\s+of\s+competitors"
    r"|the\s+most\s+competitors"
    r"|the\s+largest(?:\s+competitor)?\s+field)"
)
GOLD_WIN = r"(?:(?:won|took|earned)\s+(?:the\s+)?gold(?:\s+medal)?|was\s+the\s+gold\s+medalist)"
TEMPORAL_REL = r"(?:immediately\s+|just\s+)?(?:before|after|prior\s+to)"
COUNT_LEAD = r"(?:how\s+many|count(?:\s+the)?|what(?:\s+is|\s+was)\s+the\s+(?:number|count)\s+of)"

FRAMING_RE = re.compile(
    r"^(?:"
    r"according\s+to\s+the\s+(?:provided\s+)?(?:corpus|documents?|text|context)\s*,\s*"
    r"|(?:can|could)\s+you\s+(?:please\s+)?(?:tell\s+me|say)\s*,?\s*"
    r"|please\s+(?:tell\s+me\s+)?"
    r"|tell\s+me\s*,?\s*"
    r")+",
    re.IGNORECASE,
)

LOOKUP_RE = re.compile(
    r"^(?:"
    + r"how\s+many\s+"
    + NATION_PLURAL
    + r"\s+"
    + COMPETE_VERB
    + r"\s+in\s+(?P<title>.+)"
    + r"|what(?:\s+is|\s+was)\s+the\s+"
    + NATION_SINGULAR
    + r"\s+count\s+for\s+(?P<title_count>.+)"
    + r"|what(?:\s+is|\s+was)\s+the\s+(?:number|count)\s+of\s+"
    + NATION_PLURAL
    + r"\s+(?:that\s+)?"
    + COMPETE_VERB
    + r"\s+in\s+(?P<title_number>.+)"
    + r")$",
    re.IGNORECASE,
)

AGG_RE = re.compile(
    r"(?:"
    + r"(?:"
    + COUNT_LEAD
    + r")\s+(?P<sport>.+?)\s+events\s+at\s+the\s+"
    + r"(?P<year>\d{4})\s+(?P<season>Summer|Winter)\s+Olympics"
    + r"|at\s+the\s+(?P<year_front>\d{4})\s+(?P<season_front>Summer|Winter)\s+Olympics,?\s+"
    + r"(?:"
    + COUNT_LEAD
    + r")\s+(?P<sport_front>.+?)\s+events"
    + r")"
    + r"\s+(?:"
    + r"(?:that\s+)?(?:had|with)\s+"
    + MORE_THAN
    + r"\s+(?P<threshold>\d+)\s+competitors"
    + r"|exceeded\s+(?P<threshold_ex>\d+)\s+competitors"
    + r")",
    re.IGNORECASE,
)

SUP_RE = re.compile(
    r"(?:"
    + r"which\s+(?P<sport>.+?)\s+event\s+at\s+the\s+(?P<year>\d{4})\s+"
    + r"(?P<season>Summer|Winter)\s+Olympics\s+had\s+"
    + MOST_COMPETITORS
    + r"|at\s+the\s+(?P<year_front>\d{4})\s+(?P<season_front>Summer|Winter)\s+Olympics,?\s+"
    + r"which\s+(?P<sport_front>.+?)\s+event\s+had\s+"
    + MOST_COMPETITORS
    + r"|which\s+(?P<year_mid>\d{4})\s+(?P<season_mid>Summer|Winter)\s+Olympics\s+"
    + r"(?P<sport_mid>.+?)\s+event\s+had\s+"
    + MOST_COMPETITORS
    + r")",
    re.IGNORECASE,
)

TEMP_RE = re.compile(
    r"(?:"
    + r"(?:who\s+)?(?:"
    + GOLD_WIN
    + r")\s+in\s+(?:the\s+)?(?P<event>.+?)\s+at\s+the\s+"
    + r"(?P<season>Summer|Winter)\s+Olympics\s+(?:held\s+)?"
    + TEMPORAL_REL
    + r"\s+(?P<year>\d{4})"
    + r"|(?P<rel_front>before|after|prior\s+to)\s+the\s+(?P<year_front>\d{4})\s+"
    + r"(?P<season_front>Summer|Winter)\s+Olympics,?\s+who\s+(?:"
    + GOLD_WIN
    + r")\s+"
    + r"in\s+(?:the\s+)?(?P<event_front>.+)"
    + r")",
    re.IGNORECASE,
)

MH_RE = re.compile(
    r"(?:"
    + r"(?:who|which(?:\s+athlete)?)\s+(?:"
    + GOLD_WIN
    + r")(?:\s+for)?"
    + r"\s+(?:in\s+)?(?:the\s+)?(?:event\s+)?(?:held\s+)?at\s+(?P<venue>.+?)\s+"
    + r"(?:on|in|from)\s+(?P<date>.+)"
    + r"|event\s+held\s+at\s+(?P<venue_legacy>.+?)\s+on\s+(?P<date_legacy>.+)"
    + r")",
    re.IGNORECASE,
)

FAMILY_ORDER = ("lookup", "aggregation", "superlative", "temporal", "multi_hop")


def normalize_question_surface(text: str) -> str:
    """Whitespace, quotes, contractions, and harmless framing. Not a rewrite of intent."""

    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = cleaned.replace("\u2019", "'").replace("\u2018", "'")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\bwhat's\b", "what is", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bwho's\b", "who is", cleaned, flags=re.IGNORECASE)
    cleaned = FRAMING_RE.sub("", cleaned).strip()
    return cleaned.rstrip(" ?!.")


class QuestionParser:
    def __init__(self, sports: list[str] | None = None) -> None:
        unique = {sport for sport in (sports or []) if sport}
        self.sports = sorted(unique, key=len, reverse=True)

    def parse(self, question: str, qtype: str | None = None) -> QuerySpec:
        raw = (question or "").strip()
        surface = normalize_question_surface(raw)
        inferred = self.infer_qtype(surface)
        resolved_type = qtype or inferred or "unknown"
        if resolved_type == "lookup":
            return self._lookup(raw, surface, resolved_type, inferred)
        if resolved_type == "aggregation":
            return self._aggregation(raw, surface, resolved_type, inferred)
        if resolved_type == "superlative":
            return self._superlative(raw, surface, resolved_type, inferred)
        if resolved_type == "temporal":
            return self._temporal(raw, surface, resolved_type, inferred)
        if resolved_type == "multi_hop":
            return self._multi_hop(raw, surface, resolved_type, inferred)
        return QuerySpec(
            qtype=resolved_type,
            operation="unknown",
            raw_question=raw,
            matched_template=False,
            inferred_qtype=inferred,
        )

    def matching_families(self, text: str) -> tuple[str, ...]:
        surface = normalize_question_surface(text)
        hits = []
        if LOOKUP_RE.search(surface):
            hits.append("lookup")
        if AGG_RE.search(surface):
            hits.append("aggregation")
        if SUP_RE.search(surface):
            hits.append("superlative")
        if TEMP_RE.search(surface):
            hits.append("temporal")
        if MH_RE.search(surface) and _multi_hop_match(surface) is not None:
            hits.append("multi_hop")
        return tuple(hits)

    def infer_qtype(self, text: str) -> str | None:
        hits = self.matching_families(text)
        if len(hits) != 1:
            return None
        return hits[0]

    def _lookup(self, raw: str, surface: str, qtype: str, inferred: str | None) -> QuerySpec:
        match = LOOKUP_RE.search(surface)
        title = _named(match, "title", "title_count", "title_number") if match else None
        return QuerySpec(
            qtype=qtype,
            operation="lookup_nations",
            raw_question=raw,
            matched_template=bool(match and title),
            event_title=title,
            requested_field="nations",
            inferred_qtype=inferred,
        )

    def _aggregation(self, raw: str, surface: str, qtype: str, inferred: str | None) -> QuerySpec:
        match = AGG_RE.search(surface)
        if not match:
            return QuerySpec(
                qtype=qtype,
                operation="count_over_threshold",
                raw_question=raw,
                matched_template=False,
                aggregation="count",
                comparison="gt",
                requested_field="count",
                inferred_qtype=inferred,
            )
        sport = _named(match, "sport", "sport_front")
        year = _named_int(match, "year", "year_front")
        season = _named(match, "season", "season_front")
        threshold = _named_int(match, "threshold", "threshold_ex")
        return QuerySpec(
            qtype=qtype,
            operation="count_over_threshold",
            raw_question=raw,
            matched_template=True,
            sport=normalize_sport_key(sport),
            year=year,
            season=_title_season(season or ""),
            threshold=threshold,
            aggregation="count",
            comparison="gt",
            requested_field="count",
            inferred_qtype=inferred,
        )

    def _superlative(self, raw: str, surface: str, qtype: str, inferred: str | None) -> QuerySpec:
        match = SUP_RE.search(surface)
        if not match:
            return QuerySpec(
                qtype=qtype,
                operation="argmax_competitors",
                raw_question=raw,
                matched_template=False,
                aggregation="argmax",
                comparison="max",
                requested_field="title",
                inferred_qtype=inferred,
            )
        sport = _named(match, "sport", "sport_front", "sport_mid")
        year = _named_int(match, "year", "year_front", "year_mid")
        season = _named(match, "season", "season_front", "season_mid")
        return QuerySpec(
            qtype=qtype,
            operation="argmax_competitors",
            raw_question=raw,
            matched_template=True,
            sport=normalize_sport_key(sport),
            year=year,
            season=_title_season(season or ""),
            aggregation="argmax",
            comparison="max",
            requested_field="title",
            inferred_qtype=inferred,
        )

    def _temporal(self, raw: str, surface: str, qtype: str, inferred: str | None) -> QuerySpec:
        match = TEMP_RE.search(surface)
        if not match:
            return QuerySpec(
                qtype=qtype,
                operation="previous_event_gold",
                raw_question=raw,
                matched_template=False,
                requested_field="gold",
                inferred_qtype=inferred,
            )
        rel_text = _named(match, "rel_front") or match.group(0)
        relation = _temporal_relation(rel_text)
        event_desc = (_named(match, "event", "event_front") or "").strip()
        season = _title_season(_named(match, "season", "season_front") or "")
        named_year = _named_int(match, "year", "year_front")
        sport, event_name = split_event_description(event_desc, self.sports)
        operation = "previous_event_gold" if relation == "previous" else "next_event_gold"
        return QuerySpec(
            qtype=qtype,
            operation=operation,
            raw_question=raw,
            matched_template=True,
            sport=sport,
            year=previous_olympiad_year(named_year, season) if relation == "previous" and named_year else None,
            season=season,
            event_name=event_name,
            event_desc=event_desc,
            temporal_relation=relation,
            requested_field="gold",
            named_year=named_year,
            inferred_qtype=inferred,
        )

    def _multi_hop(self, raw: str, surface: str, qtype: str, inferred: str | None) -> QuerySpec:
        extracted = _multi_hop_match(surface)
        if extracted is None:
            return QuerySpec(
                qtype=qtype,
                operation="events_at_venue_date",
                raw_question=raw,
                matched_template=False,
                requested_field="gold",
                inferred_qtype=inferred,
            )
        venue, date_text = extracted
        year, season = extract_games_from_text(surface)
        if year is None:
            year_match = re.search(r"\b((?:19|20)\d{2})\b", date_text)
            if year_match:
                year = int(year_match.group(1))
        return QuerySpec(
            qtype=qtype,
            operation="events_at_venue_date",
            raw_question=raw,
            matched_template=True,
            year=year,
            season=season,
            venue=venue,
            date_text=date_text,
            requested_field="gold",
            inferred_qtype=inferred,
        )


def split_event_description(event_desc: str, sports: list[str]) -> tuple[str | None, str | None]:
    text = event_desc.strip()
    folded = text.casefold()
    for sport in sports:
        suffix = f"{sport} event"
        if folded.endswith(suffix):
            body = text[: -len(suffix)].strip()
            return normalize_sport_key(sport), body or None
        if folded.endswith(sport):
            body = text[: -len(sport)].strip()
            return normalize_sport_key(sport), body or None
    if folded.endswith(" event"):
        return None, text[: -len(" event")].strip() or None
    return None, text or None


def _multi_hop_match(surface: str) -> tuple[str, str] | None:
    match = MH_RE.search(surface)
    if not match:
        return None
    venue = _named(match, "venue", "venue_legacy")
    date_text = _named(match, "date", "date_legacy")
    if not venue or not date_text:
        return None
    date_text = date_text.strip().rstrip(" ?!.")
    if _is_games_not_venue(venue):
        return None
    return venue.strip(), date_text


def _is_games_not_venue(venue: str) -> bool:
    return bool(re.fullmatch(r"the\s+(?:summer|winter)\s+olympics", venue.strip(), flags=re.IGNORECASE))


def _temporal_relation(value: str) -> str:
    folded = value.casefold()
    if re.search(r"\bafter\b", folded):
        return "next"
    return "previous"


def _named(match: re.Match[str] | None, *names: str) -> str | None:
    if match is None:
        return None
    groups = match.groupdict()
    for name in names:
        value = groups.get(name)
        if value and str(value).strip():
            return str(value).strip()
    return None


def _named_int(match: re.Match[str] | None, *names: str) -> int | None:
    value = _named(match, *names)
    return int(value) if value is not None else None


def _title_season(value: str) -> str:
    return "Winter" if value.strip().casefold() == "winter" else "Summer"
