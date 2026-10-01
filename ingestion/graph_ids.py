"""Stable deterministic identities for the Phase 4 TigerGraph schema."""

from __future__ import annotations

from ingestion.parse import normalize_sport_key, normalize_venue_key
from retrieval.structured.match import (
    GENDER_MAP,
    compact_alnum,
    fold_text,
    gender_of,
    name_tokens,
    normalize_event_name,
)

INT32_MAX = 2_147_483_647


def document_id(doc_id: str) -> str:
    return doc_id


def event_id(doc_id: str) -> str:
    return doc_id


def chunk_id(document_id_value: str, chunk_index: int) -> str:
    return f"{document_id_value}::c{chunk_index:03d}"


def games_id(year: int | None, season: str | None) -> str | None:
    if year is None or not season:
        return None
    return f"{year}_{season}"


def sport_id(sport: str | None) -> str | None:
    return normalize_sport_key(sport)


def venue_id(venue_raw: str | None) -> str | None:
    return normalize_venue_key(venue_raw)


def event_name_core(value: str | None) -> str:
    tokens = name_tokens(value)
    stripped = tuple(token for token in tokens if token not in GENDER_MAP)
    return " ".join(stripped)


def event_name_gender(value: str | None) -> str:
    return gender_of(name_tokens(value)) or ""


def packed_ints(values: tuple[int, ...] | list[int] | None) -> str:
    if not values:
        return ""
    unique = tuple(dict.fromkeys(int(value) for value in values))
    return "|" + "|".join(str(value) for value in unique) + "|"


def unpack_ints(packed: str | None) -> tuple[int, ...]:
    if not packed:
        return ()
    return tuple(int(part) for part in packed.split("|") if part)


def packed_overlap(left: str | None, right: str | None) -> bool:
    left_vals = set(unpack_ints(left))
    right_vals = set(unpack_ints(right))
    if not left_vals or not right_vals:
        return True
    return not left_vals.isdisjoint(right_vals)


def venue_token_blob(venue_raw: str | None) -> str:
    return " ".join(name_tokens(venue_raw))


def query_venue_tokens(venue_query: str | None) -> tuple[str, ...]:
    from retrieval.structured.match import STOP_VENUE

    return tuple(token for token in name_tokens(venue_query) if token not in STOP_VENUE)


def gsql_contains(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return haystack != haystack.replace(needle, "", 1)


def gsql_venue_score(
    query_compact: str,
    query_tokens: tuple[str, ...] | list[str],
    venue_compact: str,
    venue_tokens_blob: str,
) -> int:
    if not query_compact or not venue_compact:
        return 0
    if query_compact == venue_compact:
        return 100
    if gsql_contains(venue_compact, query_compact):
        return 70
    if gsql_contains(query_compact, venue_compact):
        return 60
    tokens = [token for token in query_tokens if token]
    if not tokens:
        return 0
    padded = f" {venue_tokens_blob} "
    if all(f" {token} " in padded for token in tokens):
        return 40
    return 0


def matching_keys(event_name: str | None) -> dict[str, str]:
    return {
        "event_name_folded": fold_text(event_name),
        "event_name_normalized": normalize_event_name(event_name),
        "event_name_core": event_name_core(event_name),
        "event_name_gender": event_name_gender(event_name),
    }


def names_identity_match(query_folded: str, event_folded: str) -> bool:
    return bool(query_folded) and query_folded == event_folded


def names_fuzzy_match(
    query_normalized: str,
    query_core: str,
    query_gender: str,
    event_normalized: str,
    event_core: str,
    event_gender: str,
) -> bool:
    if query_normalized and query_normalized == event_normalized:
        return True
    if not query_core or query_core != event_core:
        return False
    if query_gender and event_gender and query_gender != event_gender:
        return False
    return True


def clamp_int32(value: int | None, *, missing: int = -1) -> int:
    if value is None:
        return missing
    if value > INT32_MAX:
        return INT32_MAX
    if value < -INT32_MAX - 1:
        return -INT32_MAX - 1
    return int(value)


def compact_venue(value: str | None) -> str:
    return compact_alnum(value)
