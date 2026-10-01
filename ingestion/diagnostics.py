"""Phase 1 parser diagnostics: corpus profile and public-question coverage.

This module does not implement retrieval or answer generation. It only checks
whether parsed Event records contain the fields a question family needs.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from ingestion.models import DateSpan, ParseCorpusResult, ParsedEvent
from ingestion.parse import looks_like_olympic_event_title, normalize_sport_key, previous_olympiad_year

LOOKUP_RE = re.compile(r"^How many nations competed in (.+)\?\s*$")
AGG_RE = re.compile(
    r"how many (.+?) events at the (\d{4}) (Summer|Winter) Olympics had more than (\d+) competitors",
    re.IGNORECASE,
)
SUP_RE = re.compile(
    r"which (.+?) event at the (\d{4}) (Summer|Winter) Olympics had the highest number of competitors",
    re.IGNORECASE,
)
TEMP_RE = re.compile(
    r"gold medal in the (.+?) at the (Summer|Winter) Olympics held immediately before (\d{4})",
    re.IGNORECASE,
)
MH_RE = re.compile(r"event held at (.+?) on (.+?)(?:\?\s*)?$", re.IGNORECASE)

SAMPLE_LIMIT = 5


def corpus_profile(result: ParseCorpusResult) -> dict[str, Any]:
    documents = result.documents
    events = [document.event for document in documents if document.event is not None]
    event_docs = [document for document in documents if document.is_event]
    infobox_count = sum(1 for document in documents if document.has_olympic_infobox)
    olympic_titles = [document for document in documents if looks_like_olympic_event_title(document.title)]
    rejected_titles = [
        document
        for document in documents
        if document.rejection_reason == "olympic_title_without_infobox"
    ]
    identity_groups: dict[tuple, list[str]] = defaultdict(list)
    for event in events:
        key = (event.sport, event.year, event.season, _fold(event.event_name_raw))
        if all(part is not None for part in key[:3]) and key[3]:
            identity_groups[key].append(event.event_id)
    duplicates = {key: ids for key, ids in identity_groups.items() if len(ids) > 1}

    missing = {
        field: sum(1 for event in events if event.field_status()[field] == "missing")
        for field in (
            "sport",
            "year",
            "season",
            "event_name",
            "venue",
            "date",
            "competitors",
            "nations",
            "gold",
            "prev_year",
            "next_year",
        )
    }
    flag_counts = Counter()
    for document in documents:
        flag_counts.update(document.flags)

    sports = Counter(event.sport_raw for event in events if event.sport_raw)
    seasons = Counter(event.season for event in events if event.season)
    years = Counter(event.year for event in events if event.year is not None)

    years_pre_1988 = [
        {"doc_id": event.event_id, "title": event.title, "year": event.year, "year_source": event.year_source}
        for event in events
        if event.year is not None and event.year < 1988
    ]
    suspicious = {
        "olympic_title_without_infobox": _samples(rejected_titles),
        "title_infobox_games_conflict": _event_samples(events, "title_infobox_games_conflict"),
        "infobox_without_title_parse": _event_samples(events, "infobox_without_title_parse"),
        "missing_competitors": _event_samples_by_missing(events, "competitors"),
        "missing_nations": _event_samples_by_missing(events, "nations"),
        "missing_venue": _event_samples_by_missing(events, "venue"),
        "missing_date": _event_samples_by_missing(events, "date"),
        "malformed_competitors": _event_samples(events, "malformed_competitors"),
        "years_before_1988": years_pre_1988[:SAMPLE_LIMIT],
        "duplicate_identity": [
            {"key": list(key), "doc_ids": ids[:8], "count": len(ids)}
            for key, ids in sorted(duplicates.items(), key=lambda item: -len(item[1]))[:SAMPLE_LIMIT]
        ],
    }

    return {
        "total_documents": len(documents),
        "event_documents": len(event_docs),
        "document_only": len(documents) - len(event_docs),
        "documents_with_olympic_event_infobox": infobox_count,
        "olympic_looking_titles": len(olympic_titles),
        "rejected_despite_olympic_title": len(rejected_titles),
        "parse_failures": 0,
        "missing_values": missing,
        "missing_value_rates": {
            field: round(100.0 * count / len(events), 1) if events else 0.0
            for field, count in missing.items()
        },
        "sport_distribution": sports.most_common(),
        "season_distribution": seasons.most_common(),
        "year_distribution": sorted(years.items()),
        "duplicate_identity_groups": len(duplicates),
        "flag_counts": flag_counts.most_common(),
        "suspicious": suspicious,
        "rejected_title_samples": [
            {"doc_id": document.doc_id, "title": document.title}
            for document in rejected_titles[:SAMPLE_LIMIT]
        ],
    }


def question_coverage(result: ParseCorpusResult, questions: list[dict]) -> dict[str, Any]:
    events_by_id = result.event_by_id()
    documents_by_id = result.document_by_id()
    events = list(result.events())
    rows = []
    by_type: dict[str, Counter] = defaultdict(Counter)

    for question in questions:
        qtype = str(question.get("qtype") or "unknown")
        row = _cover_question(question, events, events_by_id, documents_by_id)
        rows.append(row)
        by_type[qtype][row["support"]] += 1
        by_type[qtype]["n"] += 1

    summary = {}
    for qtype, counts in sorted(by_type.items()):
        n = counts["n"]
        summary[qtype] = {
            "n": n,
            "supported": counts["supported"],
            "partial": counts["partial"],
            "unsupported": counts["unsupported"],
            "supported_pct": round(100.0 * counts["supported"] / n, 1) if n else 0.0,
        }
    overall_n = len(rows)
    overall_supported = sum(1 for row in rows if row["support"] == "supported")
    return {
        "n": overall_n,
        "supported": overall_supported,
        "supported_pct": round(100.0 * overall_supported / overall_n, 1) if overall_n else 0.0,
        "by_qtype": summary,
        "unsupported_or_partial": [
            {
                "qid": row["qid"],
                "qtype": row["qtype"],
                "support": row["support"],
                "reasons": row["reasons"],
            }
            for row in rows
            if row["support"] != "supported"
        ],
        "rows": rows,
    }


def schema_validation(result: ParseCorpusResult) -> dict[str, Any]:
    events = list(result.events())
    n = len(events) or 1
    field_availability = {}
    for field in (
        "sport",
        "year",
        "season",
        "event_name",
        "venue",
        "date",
        "competitors",
        "nations",
        "gold",
        "prev_year",
        "next_year",
    ):
        present = sum(1 for event in events if event.field_status()[field] == "present")
        rate = 100.0 * present / n
        if rate >= 95:
            bucket = "reliably_available"
        elif rate >= 70:
            bucket = "partially_available"
        else:
            bucket = "sparse"
        field_availability[field] = {
            "present": present,
            "missing": len(events) - present,
            "pct": round(rate, 1),
            "bucket": bucket,
        }

    noc_present = sum(
        1
        for event in events
        if any(medal.place == "gold" and medal.noc for medal in event.medals)
    )
    prev_matchable = 0
    prev_with_year = 0
    index = _event_index(events)
    for event in events:
        if event.prev_year is None or event.sport is None or not event.event_name_raw:
            continue
        prev_with_year += 1
        if index.get((event.sport, event.prev_year, event.season, _fold(event.event_name_raw))):
            prev_matchable += 1

    return {
        "proposed_vertices": {
            "Document": {
                "status": "reliably_available",
                "note": "Every corpus row has doc_id, title, url, text.",
            },
            "Chunk": {
                "status": "deferred",
                "note": "Phase 1 does not chunk. Infobox text is recoverable from provenance raw_text.",
            },
            "Event": {
                "status": "reliably_available",
                "note": f"{len(events)} infobox-gated Event records.",
            },
            "Games": {
                "status": "reliably_available",
                "note": "year+season present on almost all Events; infobox games can conflict with title.",
            },
            "Sport": {
                "status": "reliably_available",
                "note": "Taken from title grammar, not infobox event prose.",
            },
            "Venue": {
                "status": field_availability["venue"]["bucket"],
                "note": "Raw venue strings plus a light venue_key. Distinct halls must not be merged.",
            },
            "Athlete": {
                "status": "ambiguous",
                "note": "gold/silver/bronze raw strings are present; concatenated team names should not be split yet.",
            },
            "Nation": {
                "status": "partially_available" if noc_present / n >= 0.7 else "sparse",
                "note": f"{noc_present} Events have a parseable gold NOC.",
            },
        },
        "event_fields": field_availability,
        "prev_event_representable": {
            "events_with_prev_year": prev_with_year,
            "matched_same_sport_event_name": prev_matchable,
            "pct_of_prev_year": round(100.0 * prev_matchable / prev_with_year, 1) if prev_with_year else 0.0,
        },
    }


def extract_question_slots(question_text: str, qtype: str) -> dict[str, Any]:
    text = question_text.strip()
    if qtype == "lookup":
        match = LOOKUP_RE.match(text)
        return {"event_title": match.group(1).strip() if match else None, "matched_template": bool(match)}
    if qtype == "aggregation":
        match = AGG_RE.search(text)
        if not match:
            return {"matched_template": False}
        return {
            "matched_template": True,
            "sport": match.group(1),
            "year": int(match.group(2)),
            "season": match.group(3),
            "threshold": int(match.group(4)),
        }
    if qtype == "superlative":
        match = SUP_RE.search(text)
        if not match:
            return {"matched_template": False}
        return {
            "matched_template": True,
            "sport": match.group(1),
            "year": int(match.group(2)),
            "season": match.group(3),
        }
    if qtype == "temporal":
        match = TEMP_RE.search(text)
        if not match:
            return {"matched_template": False}
        return {
            "matched_template": True,
            "event_desc": match.group(1),
            "season": match.group(2),
            "named_year": int(match.group(3)),
            "previous_year": previous_olympiad_year(int(match.group(3)), match.group(2)),
        }
    if qtype == "multi_hop":
        match = MH_RE.search(text)
        if not match:
            return {"matched_template": False}
        return {
            "matched_template": True,
            "venue": match.group(1).strip(),
            "date_text": match.group(2).strip().rstrip("?"),
        }
    return {"matched_template": False}


def _cover_question(
    question: dict,
    events: list[ParsedEvent],
    events_by_id: dict[str, ParsedEvent],
    documents_by_id: dict[str, Any],
) -> dict[str, Any]:
    qtype = str(question.get("qtype") or "unknown")
    text = str(question.get("question") or "")
    gold_ids = list(question.get("gold_doc_ids") or [])
    slots = extract_question_slots(text, qtype)
    reasons: list[str] = []
    gold_events = [events_by_id[doc_id] for doc_id in gold_ids if doc_id in events_by_id]
    missing_gold = [doc_id for doc_id in gold_ids if doc_id in documents_by_id and doc_id not in events_by_id]
    unknown_gold = [doc_id for doc_id in gold_ids if doc_id not in documents_by_id]

    if unknown_gold:
        reasons.append(f"gold_doc_not_in_corpus:{','.join(unknown_gold)}")
    if missing_gold:
        reasons.append(f"gold_doc_not_event:{','.join(missing_gold)}")
    if not slots.get("matched_template", False):
        reasons.append("question_template_unparsed")

    support = "unsupported"
    extra: dict[str, Any] = {"slots": slots, "gold_event_count": len(gold_events), "gold_doc_count": len(gold_ids)}

    if qtype == "lookup":
        title = slots.get("event_title")
        title_hit = next((event for event in events if event.title == title), None)
        extra["title_resolved"] = title_hit.event_id if title_hit else None
        if title_hit is None:
            reasons.append("event_title_not_found")
        elif title_hit.nations is None:
            reasons.append("nations_missing")
            support = "partial"
        else:
            support = "supported"
        if gold_ids and title_hit and gold_ids[0] != title_hit.event_id:
            reasons.append("title_event_id_differs_from_gold")
            support = "partial"
    elif qtype in {"aggregation", "superlative"}:
        sport = normalize_sport_key(slots.get("sport"))
        year = slots.get("year")
        season = slots.get("season")
        pool = [
            event
            for event in events
            if event.sport == sport and event.year == year and event.season == season
        ]
        pool_with_competitors = [event for event in pool if event.competitors is not None]
        extra["candidate_pool"] = len(pool)
        extra["candidates_with_competitors"] = len(pool_with_competitors)
        extra["gold_with_competitors"] = sum(1 for event in gold_events if event.competitors is not None)
        if not slots.get("matched_template"):
            support = "unsupported"
        elif not pool_with_competitors:
            reasons.append("no_sport_games_events_with_competitors")
            support = "unsupported"
        elif gold_ids and extra["gold_with_competitors"] < len(gold_ids):
            reasons.append("some_gold_events_missing_competitors")
            support = "partial"
        else:
            support = "supported"
        if gold_ids and len(pool) != len(gold_ids) and support == "supported":
            extra["pool_vs_gold"] = {"pool": len(pool), "gold": len(gold_ids)}
            reasons.append("candidate_pool_size_differs_from_gold_set")
            support = "partial"
    elif qtype == "temporal":
        extra["gold_with_gold_medal"] = sum(1 for event in gold_events if event.gold_raw())
        extra["gold_years"] = sorted({event.year for event in gold_events if event.year is not None})
        named = slots.get("named_year")
        previous = slots.get("previous_year")
        named_event = next((event for event in gold_events if event.year == named), None)
        prev_event = next((event for event in gold_events if event.year == previous), None)
        extra["named_event_has_prev_year"] = bool(named_event and named_event.prev_year)
        extra["prev_matches_calendar"] = bool(
            named_event and previous and named_event.prev_year == previous
        )
        if len(gold_events) < 2:
            reasons.append("fewer_than_two_temporal_events")
            support = "unsupported"
        elif prev_event is None or not prev_event.gold_raw():
            reasons.append("previous_event_missing_gold")
            support = "partial"
        else:
            support = "supported"
        if named_event is None:
            reasons.append("named_year_event_missing")
            if support == "supported":
                support = "partial"
    elif qtype == "multi_hop":
        if len(gold_events) != 1:
            reasons.append("expected_one_gold_event")
            support = "unsupported"
        else:
            event = gold_events[0]
            extra["gold_has_venue"] = bool(event.venue_raw)
            extra["gold_has_date"] = event.date is not None
            extra["gold_has_gold_medal"] = bool(event.gold_raw())
            extra["date_has_month"] = bool(event.date and event.date.months)
            question_date = DateSpan(raw=str(slots.get("date_text") or ""), source_field="question")
            if slots.get("date_text"):
                from ingestion.parse import parse_date_span

                parsed_qdate = parse_date_span(slots["date_text"], "question", fallback_year=event.year)
                extra["question_date_parsed"] = bool(parsed_qdate and (parsed_qdate.months or parsed_qdate.days))
                extra["date_token_overlap"] = _date_overlap(event.date, parsed_qdate)
            else:
                extra["question_date_parsed"] = False
                extra["date_token_overlap"] = False
                parsed_qdate = question_date
            venue_q = _fold(slots.get("venue"))
            extra["venue_key_match"] = bool(
                venue_q and event.venue_key and (venue_q in event.venue_key or event.venue_key in venue_q)
            )
            if not event.venue_raw:
                reasons.append("venue_missing")
            if event.date is None:
                reasons.append("date_missing")
            if not event.gold_raw():
                reasons.append("gold_missing")
            if reasons:
                support = "partial" if event.gold_raw() else "unsupported"
            else:
                support = "supported"
    else:
        reasons.append(f"unknown_qtype:{qtype}")
        support = "unsupported"

    if not gold_ids and qtype != "lookup":
        # Questions without gold ids stay slot-only for existence checks.
        if qtype in {"aggregation", "superlative"} and extra.get("candidates_with_competitors"):
            support = "supported"
            reasons = [reason for reason in reasons if not reason.startswith("gold")]
        elif qtype in {"temporal", "multi_hop"}:
            reasons.append("no_gold_doc_ids_for_referenced_event_check")
            if support == "supported":
                support = "partial"

    return {
        "qid": question.get("qid"),
        "qtype": qtype,
        "support": support,
        "reasons": reasons,
        **extra,
    }


def _event_index(events: list[ParsedEvent]) -> dict[tuple, list[str]]:
    index: dict[tuple, list[str]] = defaultdict(list)
    for event in events:
        key = (event.sport, event.year, event.season, _fold(event.event_name_raw))
        index[key].append(event.event_id)
    return index


def _date_overlap(event_date: DateSpan | None, question_date: DateSpan | None) -> bool:
    if event_date is None or question_date is None:
        return False
    months_ok = not event_date.months or not question_date.months or not set(event_date.months).isdisjoint(
        question_date.months
    )
    days_ok = not event_date.days or not question_date.days or not set(event_date.days).isdisjoint(
        question_date.days
    )
    years_ok = not event_date.years or not question_date.years or not set(event_date.years).isdisjoint(
        question_date.years
    )
    return months_ok and days_ok and years_ok


def _samples(documents: list) -> list[dict[str, str]]:
    return [{"doc_id": document.doc_id, "title": document.title} for document in documents[:SAMPLE_LIMIT]]


def _event_samples(events: list[ParsedEvent], flag: str) -> list[dict[str, str]]:
    matched = [event for event in events if flag in event.flags]
    return [
        {"doc_id": event.event_id, "title": event.title, "flags": ",".join(event.flags)}
        for event in matched[:SAMPLE_LIMIT]
    ]


def _event_samples_by_missing(events: list[ParsedEvent], field: str) -> list[dict[str, str]]:
    matched = [event for event in events if event.field_status()[field] == "missing"]
    return [{"doc_id": event.event_id, "title": event.title} for event in matched[:SAMPLE_LIMIT]]


def _fold(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"\s+", " ", value).strip().casefold()
