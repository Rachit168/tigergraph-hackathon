"""Deterministic lexical query plans from the user question. No LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass

from retrieval.rag.sparse import tokenize

YEAR_RE = re.compile(r"\b((?:18|19|20)\d{2})\b")
SEASON_RE = re.compile(r"\b(summer|winter)\b", re.IGNORECASE)
SET_COUNT_RE = re.compile(
    r"how many\s+(.+?)\s+events?\s+at\b.{0,80}?\b(?:more than|fewer than|at least|over)\s+\d+",
    re.IGNORECASE | re.DOTALL,
)
SET_EXTREME_RE = re.compile(
    r"which\s+(.+?)\s+events?\s+at\b.{0,80}?\b(?:highest|lowest|most|fewest|largest|smallest)\b",
    re.IGNORECASE | re.DOTALL,
)
TEMPORAL_RE = re.compile(r"\b(?:previous|prior|before|next|following|after)\b", re.IGNORECASE)
BOILERPLATE_RE = re.compile(r"^\s*according to the provided corpus,?\s*", re.IGNORECASE)
_STOP = frozenset(
    {
        "according",
        "provided",
        "corpus",
        "how",
        "many",
        "which",
        "what",
        "who",
        "whom",
        "had",
        "have",
        "has",
        "was",
        "were",
        "did",
        "the",
        "a",
        "an",
        "of",
        "in",
        "at",
        "on",
        "for",
        "with",
        "from",
        "than",
        "more",
        "most",
        "less",
        "least",
        "highest",
        "lowest",
        "largest",
        "smallest",
        "fewest",
        "number",
        "numbers",
        "count",
        "events",
        "event",
        "competed",
        "competitors",
        "nations",
        "gold",
        "won",
        "winner",
        "previous",
        "prior",
        "before",
        "next",
        "following",
        "after",
        "and",
        "or",
        "to",
        "by",
    }
)


@dataclass(frozen=True)
class QueryPlan:
    original: str
    cleaned: str
    variants: tuple[str, ...]
    years: tuple[int, ...]
    seasons: tuple[str, ...]
    sport_tokens: tuple[str, ...]
    content_tokens: tuple[str, ...]
    intent: str


def plan_query(question: str) -> QueryPlan:
    original = question or ""
    cleaned = BOILERPLATE_RE.sub("", original).strip() or original.strip()
    years = tuple(int(match) for match in YEAR_RE.findall(cleaned))
    seasons = tuple(dict.fromkeys(part.casefold() for part in SEASON_RE.findall(cleaned)))
    intent = _intent(cleaned)
    sport_tokens = _sport_tokens(cleaned, intent)
    content_tokens = tuple(
        token for token in tokenize(cleaned) if token not in _STOP and not token.isdigit()
    )
    variants = _variants(cleaned, years, seasons, sport_tokens, content_tokens)
    return QueryPlan(
        original=original,
        cleaned=cleaned,
        variants=variants,
        years=years,
        seasons=seasons,
        sport_tokens=sport_tokens,
        content_tokens=content_tokens,
        intent=intent,
    )


def _intent(text: str) -> str:
    if SET_COUNT_RE.search(text):
        return "set_count"
    if SET_EXTREME_RE.search(text):
        return "set_extreme"
    if TEMPORAL_RE.search(text):
        return "temporal"
    return "lookup"


def _sport_tokens(text: str, intent: str) -> tuple[str, ...]:
    match = SET_COUNT_RE.search(text) if intent == "set_count" else None
    if match is None and intent == "set_extreme":
        match = SET_EXTREME_RE.search(text)
    phrase = match.group(1) if match is not None else ""
    tokens = tuple(token for token in tokenize(phrase) if token not in _STOP and not token.isdigit())
    if tokens:
        return tokens
    return tuple(token for token in tokenize(text) if token not in _STOP and not token.isdigit())[:4]


def _variants(
    cleaned: str,
    years: tuple[int, ...],
    seasons: tuple[str, ...],
    sport_tokens: tuple[str, ...],
    content_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    parts = list(sport_tokens or content_tokens[:6])
    parts.extend(str(year) for year in years)
    parts.extend(seasons)
    if any(token in {"olympics", "olympic"} for token in tokenize(cleaned)):
        parts.append("olympics")
    title_like = " ".join(dict.fromkeys(parts))
    variants = [cleaned]
    if title_like and title_like.casefold() != cleaned.casefold():
        variants.append(title_like)
    if sport_tokens and years:
        variants.append(" ".join((*sport_tokens, str(years[0]), "olympics", "competitors")))
    ordered: list[str] = []
    seen: set[str] = set()
    for item in variants:
        key = item.casefold()
        if not item.strip() or key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return tuple(ordered)
