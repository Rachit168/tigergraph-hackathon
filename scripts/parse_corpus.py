"""Parse the hackathon corpus and write a Phase 1 diagnostic report.

Usage:
    python -m scripts.parse_corpus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingestion.diagnostics import corpus_profile, question_coverage, schema_validation
from ingestion.loader import load_questions, parse_corpus
from ingestion.paths import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_PUBLIC_QUESTIONS_PATH,
    REPO_ROOT,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 corpus parser diagnostics")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--questions", type=Path, default=DEFAULT_PUBLIC_QUESTIONS_PATH)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "_research" / "parser_ceiling_report.md",
    )
    args = parser.parse_args()

    result = parse_corpus(args.corpus)
    profile = corpus_profile(result)
    questions = load_questions(args.questions) if args.questions.exists() else []
    coverage = question_coverage(result, questions) if questions else {}
    schema = schema_validation(result)
    report = render_report(profile, coverage, schema)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {args.report}")
    return 0


def render_report(profile: dict, coverage: dict, schema: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Parser Ceiling Report (Phase 1)")
    add("")
    add("**Status:** analysis of the deterministic infobox/title parser. Not retrieval. Not TigerGraph.")
    add("")
    add("Event records are created only when `[Infobox Olympic event]` is present. Competitors and nations come only from infobox fields.")
    add("")
    add("## 1. Corpus parsing statistics")
    add("")
    add(f"| Metric | Count |")
    add("|---|---:|")
    add(f"| Total documents | {profile['total_documents']} |")
    add(f"| Event documents | {profile['event_documents']} |")
    add(f"| Document-only | {profile['document_only']} |")
    add(f"| Documents with Olympic event infobox | {profile['documents_with_olympic_event_infobox']} |")
    add(f"| Olympic-looking event titles | {profile['olympic_looking_titles']} |")
    add(f"| Rejected despite Olympic-looking title | {profile['rejected_despite_olympic_title']} |")
    add(f"| Parser crashes / JSON failures | {profile['parse_failures']} |")
    add(f"| Duplicate Event identity groups | {profile['duplicate_identity_groups']} |")
    add("")
    add("## 2. Field coverage (among Event records)")
    add("")
    add("| Field | Missing | Missing % |")
    add("|---|---:|---:|")
    missing = profile["missing_values"]
    rates = profile["missing_value_rates"]
    for field, count in missing.items():
        add(f"| {field} | {count} | {rates[field]}% |")
    add("")
    add("### Sport distribution")
    add("")
    add("| Sport | Events |")
    add("|---|---:|")
    for sport, count in profile["sport_distribution"]:
        add(f"| {sport} | {count} |")
    add("")
    add("### Season distribution")
    add("")
    add("| Season | Events |")
    add("|---|---:|")
    for season, count in profile["season_distribution"]:
        add(f"| {season} | {count} |")
    add("")
    add("### Year distribution")
    add("")
    add("| Year | Events |")
    add("|---|---:|")
    for year, count in profile["year_distribution"]:
        add(f"| {year} | {count} |")
    add("")
    add("## 3. Rejected / suspicious categories")
    add("")
    add("Flag counts:")
    add("")
    add("| Flag | Count |")
    add("|---|---:|")
    for flag, count in profile["flag_counts"]:
        add(f"| {flag} | {count} |")
    add("")
    suspicious = profile["suspicious"]
    for category, samples in suspicious.items():
        add(f"### {category}")
        add("")
        if not samples:
            add("None.")
            add("")
            continue
        add("```json")
        add(json.dumps(samples, indent=2, ensure_ascii=False))
        add("```")
        add("")
    add("## 4. Public-question structural coverage")
    add("")
    if not coverage:
        add("Public questions file was not found.")
        add("")
    else:
        add("This is **not** retrieval accuracy. A question is `supported` when the parsed Event table contains the fields that family needs.")
        add("")
        add(f"Overall supported: **{coverage['supported']} / {coverage['n']}** ({coverage['supported_pct']}%).")
        add("")
        add("## 5. Coverage by qtype")
        add("")
        add("| qtype | n | supported | partial | unsupported | supported % |")
        add("|---|---:|---:|---:|---:|---:|")
        for qtype, stats in coverage["by_qtype"].items():
            add(
                f"| {qtype} | {stats['n']} | {stats['supported']} | {stats['partial']} | "
                f"{stats['unsupported']} | {stats['supported_pct']}% |"
            )
        add("")
        leftover = coverage.get("unsupported_or_partial") or []
        add(f"Partial/unsupported items: {len(leftover)}")
        add("")
        if leftover:
            add("| qid | qtype | support | reasons |")
            add("|---|---|---|---|")
            for row in leftover:
                reasons = "; ".join(row["reasons"]) if row["reasons"] else ""
                add(f"| {row['qid']} | {row['qtype']} | {row['support']} | {reasons} |")
            add("")
    add("## 6. Schema validation")
    add("")
    add("| Proposed vertex | Status | Note |")
    add("|---|---|---|")
    for name, info in schema["proposed_vertices"].items():
        add(f"| {name} | {info['status']} | {info['note']} |")
    add("")
    add("| Event field | Present | Missing | % | Bucket |")
    add("|---|---:|---:|---:|---|")
    for field, info in schema["event_fields"].items():
        add(
            f"| {field} | {info['present']} | {info['missing']} | {info['pct']}% | {info['bucket']} |"
        )
    add("")
    prev = schema["prev_event_representable"]
    add(
        f"PREV_EVENT alignment via sport+event_name+prev_year: "
        f"{prev['matched_same_sport_event_name']} / {prev['events_with_prev_year']} "
        f"({prev['pct_of_prev_year']}%) of Events that have `prev`."
    )
    add("")
    add("## 7. Problems discovered")
    add("")
    add(_problems_section(profile, coverage, schema))
    add("")
    add("## 8. Recommended changes to v0.2")
    add("")
    add(_recommendations(profile, coverage, schema))
    add("")
    add("## 9. Proceed to TigerGraph ingestion?")
    add("")
    add(_proceed(profile, coverage))
    add("")
    return "\n".join(lines)


def _problems_section(profile: dict, coverage: dict, schema: dict) -> str:
    bullets = [
        f"- Infobox-gated Events: {profile['event_documents']}; document-only: {profile['document_only']}.",
        f"- Olympic-looking event titles rejected for missing infobox: {profile['rejected_despite_olympic_title']} (team-sport tournament pages and similar body-only articles; they stay Document-only).",
        f"- Title vs infobox games conflicts: {dict(profile['flag_counts']).get('title_infobox_games_conflict', 0)}. Title year/season is preferred.",
        f"- Competitors missing on {profile['missing_values']['competitors']} Events; nations missing on {profile['missing_values']['nations']}.",
        f"- Venue missing on {profile['missing_values']['venue']} Events; date missing on {profile['missing_values']['date']}.",
        f"- Duplicate identity groups: {profile['duplicate_identity_groups']}.",
    ]
    leftover = (coverage or {}).get("unsupported_or_partial") or []
    if leftover:
        bullets.append(f"- Public questions not fully structurally supported: {len(leftover)}.")
    prev = schema["prev_event_representable"]
    bullets.append(
        f"- Same-name PREV_EVENT match rate {prev['pct_of_prev_year']}% — event program changes will need calendar fallback in Phase 2, not extra parser invention."
    )
    return "\n".join(bullets)


def _recommendations(profile: dict, coverage: dict, schema: dict) -> str:
    return "\n".join(
        [
            "Keep the infobox gate. Do not promote Olympic-looking titles without infoboxes to Event vertices.",
            "",
            "- **Document / Event split:** confirmed. No schema change required.",
            "- **Event.competitors / Event.nations:** infobox-only is the right default. Do not mix body-table counts.",
            "- **Games year/season:** prefer title over infobox `games` when they conflict; store both.",
            "- **Athlete vertex:** defer splitting concatenated medal names. Store `gold_raw` for answers.",
            "- **Chunk:** still deferred to Phase 3; provenance quotes already exist from infobox lines.",
            "- **PREV_EVENT:** keep `prev_year` on Event; Phase 2 should combine same-name match + olympiad calendar, not parser-invented edges.",
            "- **Venue/date:** keep raw strings plus light keys/tokens. Do not over-normalize hall names.",
        ]
    )


def _proceed(profile: dict, coverage: dict) -> str:
    supported_pct = (coverage or {}).get("supported_pct")
    if profile["event_documents"] >= 2000 and (supported_pct is None or supported_pct >= 80):
        return (
            "**Yes, proceed to Phase 2 (offline structured solver), not yet to TigerGraph load.** "
            "The parser validates the Event table bet. Ingest into TigerGraph after Phase 2 confirms query logic on this table."
        )
    return (
        "**Not yet.** Fix parser coverage gaps listed above before spending time on Savanna/GSQL."
    )


if __name__ == "__main__":
    raise SystemExit(main())
