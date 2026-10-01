"""Deterministic candidate expansion, grouping, reranking, and compaction."""

from __future__ import annotations

import re
from collections import defaultdict

from ingestion.models import TextChunk
from retrieval.rag.models import RetrievalHit, RetrievalResult
from retrieval.rag.query import QueryPlan
from retrieval.rag.sparse import tokenize

INFOBOX_KEEP_RE = re.compile(
    r"^\s*(?:event|games|date|venue|competitors|nations|gold|silver|bronze|prev|next)\s*:",
    re.IGNORECASE,
)
PREV_YEAR_RE = re.compile(r"(?:^|\n)\s*prev:\s*((?:18|19|20)\d{2})", re.IGNORECASE)
NEXT_YEAR_RE = re.compile(r"(?:^|\n)\s*next:\s*((?:18|19|20)\d{2})", re.IGNORECASE)

_SET_INTENTS = frozenset({"set_count", "set_extreme"})
_SET_CAP = 64
_POOL_K = 120


def select_evidence(
    *,
    query: str,
    plan: QueryPlan,
    first_hits: list[RetrievalHit],
    chunks_by_document: dict[str, list[TextChunk]],
    infoboxes: list[TextChunk],
    top_k: int,
    method: str,
) -> RetrievalResult:
    scored = {hit.chunk_id: hit.score for hit in first_hits}
    for hit in first_hits:
        scored[hit.chunk_id] = max(scored.get(hit.chunk_id, 0.0), hit.score)

    selected_chunks: list[TextChunk]
    if plan.intent in _SET_INTENTS and plan.years and plan.sport_tokens:
        selected_chunks = _set_infoboxes(plan, infoboxes)
        if not selected_chunks:
            selected_chunks = _expand_documents(first_hits[: max(top_k, 20)], chunks_by_document, prefer_infobox=True)
    else:
        seed = first_hits[: max(top_k * 2, 12)]
        selected_chunks = _expand_documents(seed, chunks_by_document, prefer_infobox=True)
        if plan.intent == "temporal":
            selected_chunks = _temporal_expand(plan, selected_chunks, infoboxes)

    grouped = _group_chunks(selected_chunks)
    ordered_docs = _rank_groups(grouped, plan, scored, first_hits)
    final_k = _final_k(plan, top_k, len(ordered_docs))
    hits = _compact_hits(ordered_docs, grouped, plan, scored, final_k, method)
    return RetrievalResult(
        query=query,
        method=f"{method}+candidates",
        hits=hits,
        params={
            "top_k": top_k,
            "final_k": final_k,
            "intent": plan.intent,
            "n_variants": len(plan.variants),
            "n_groups": len(ordered_docs),
            "sport_tokens": list(plan.sport_tokens),
            "years": list(plan.years),
        },
    )


def evidence_text(chunk: TextChunk, *, compact: bool) -> str:
    body = chunk.text or ""
    title = chunk.document_title or ""
    if compact:
        body = _compact_infobox(body, title)
    if title and title not in body:
        return f"{title}\n{body}".strip()
    return body.strip() or title


def _set_infoboxes(plan: QueryPlan, infoboxes: list[TextChunk]) -> list[TextChunk]:
    matched = [chunk for chunk in infoboxes if _title_matches_set(chunk.document_title, plan)]
    matched.sort(key=lambda chunk: (chunk.document_title, chunk.chunk_id))
    return matched[:_SET_CAP]


def _title_matches_set(title: str, plan: QueryPlan) -> bool:
    folded = (title or "").casefold()
    if not folded:
        return False
    if plan.years and not any(str(year) in folded for year in plan.years):
        return False
    return all(_has_word(folded, token) for token in plan.sport_tokens)


def _has_word(text: str, token: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) is not None


def _expand_documents(
    hits: list[RetrievalHit],
    chunks_by_document: dict[str, list[TextChunk]],
    *,
    prefer_infobox: bool,
) -> list[TextChunk]:
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.document_id in seen:
            continue
        seen.add(hit.document_id)
        ordered_ids.append(hit.document_id)
    expanded: list[TextChunk] = []
    for document_id in ordered_ids:
        members = list(chunks_by_document.get(document_id) or [])
        if prefer_infobox:
            members.sort(key=lambda chunk: (0 if chunk.kind == "infobox" else 1, chunk.chunk_index, chunk.chunk_id))
        expanded.extend(members[:3])
    return expanded


def _temporal_expand(plan: QueryPlan, chunks: list[TextChunk], infoboxes: list[TextChunk]) -> list[TextChunk]:
    extra_years: list[int] = []
    for chunk in chunks:
        if chunk.kind != "infobox":
            continue
        text = chunk.text or ""
        if re.search(r"\b(?:previous|prior|before)\b", plan.cleaned, flags=re.IGNORECASE):
            extra_years.extend(int(value) for value in PREV_YEAR_RE.findall(text))
        if re.search(r"\b(?:next|following|after)\b", plan.cleaned, flags=re.IGNORECASE):
            extra_years.extend(int(value) for value in NEXT_YEAR_RE.findall(text))
    years = tuple(dict.fromkeys(extra_years))
    if not years:
        return chunks
    shifted = QueryPlan(
        original=plan.original,
        cleaned=plan.cleaned,
        variants=plan.variants,
        years=years,
        seasons=plan.seasons,
        sport_tokens=plan.sport_tokens or plan.content_tokens[:4],
        content_tokens=plan.content_tokens,
        intent=plan.intent,
    )
    extras = _set_infoboxes(shifted, infoboxes)
    seen = {chunk.chunk_id for chunk in chunks}
    merged = list(chunks)
    for chunk in extras:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        merged.append(chunk)
    return merged


def _group_chunks(chunks: list[TextChunk]) -> dict[str, list[TextChunk]]:
    grouped: dict[str, list[TextChunk]] = defaultdict(list)
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        grouped[chunk.document_id].append(chunk)
    return grouped


def _rank_groups(
    grouped: dict[str, list[TextChunk]],
    plan: QueryPlan,
    scores: dict[str, float],
    first_hits: list[RetrievalHit],
) -> list[str]:
    first_rank: dict[str, int] = {}
    for hit in first_hits:
        first_rank.setdefault(hit.document_id, hit.rank)

    def key(document_id: str) -> tuple:
        members = grouped[document_id]
        title = members[0].document_title if members else ""
        max_score = max((scores.get(chunk.chunk_id, 0.0) for chunk in members), default=0.0)
        coverage = _constraint_coverage(plan, title, " ".join(chunk.text or "" for chunk in members))
        infobox = 1 if any(chunk.kind == "infobox" for chunk in members) else 0
        rank = first_rank.get(document_id, 10_000)
        if plan.intent in _SET_INTENTS:
            return (-coverage, -infobox, -max_score, rank, title, document_id)
        return (rank, -infobox, -max_score, -coverage, title, document_id)

    return sorted(grouped.keys(), key=key)


def _constraint_coverage(plan: QueryPlan, title: str, text: str) -> int:
    haystack = f"{title}\n{text}".casefold()
    score = 0
    for year in plan.years:
        if str(year) in haystack:
            score += 3
    for token in plan.sport_tokens:
        if _has_word(haystack, token):
            score += 4
    for season in plan.seasons:
        if season in haystack:
            score += 1
    query_tokens = [token for token in tokenize(plan.cleaned) if token not in {"according", "provided", "corpus"}]
    if query_tokens:
        present = sum(1 for token in query_tokens if _has_word(haystack, token))
        score += present
    if "competitors:" in haystack and any(token in plan.cleaned.casefold() for token in ("competitor", "highest", "more than", "most")):
        score += 2
    if "nations:" in haystack and "nation" in plan.cleaned.casefold():
        score += 2
    if "gold:" in haystack and "gold" in plan.cleaned.casefold():
        score += 2
    return score


def _final_k(plan: QueryPlan, top_k: int, n_docs: int) -> int:
    if plan.intent in _SET_INTENTS:
        return max(top_k, min(_SET_CAP, n_docs))
    if plan.intent == "temporal":
        return min(top_k + 4, max(top_k, n_docs))
    return max(1, top_k)


def _compact_hits(
    ordered_docs: list[str],
    grouped: dict[str, list[TextChunk]],
    plan: QueryPlan,
    scores: dict[str, float],
    final_k: int,
    method: str,
) -> list[RetrievalHit]:
    compact = plan.intent in _SET_INTENTS
    hits: list[RetrievalHit] = []
    for document_id in ordered_docs:
        members = sorted(
            grouped[document_id],
            key=lambda chunk: (0 if chunk.kind == "infobox" else 1, chunk.chunk_index, chunk.chunk_id),
        )
        take = members[:1] if compact else members[:2]
        for chunk in take:
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    score=scores.get(chunk.chunk_id, 0.0),
                    rank=len(hits) + 1,
                    text=evidence_text(chunk, compact=compact),
                    section=chunk.section,
                    kind=chunk.kind,
                    event_id=chunk.event_id,
                    retrieval_method=method,
                    source_url=chunk.source_url,
                )
            )
            if len(hits) >= final_k:
                return hits
    return hits


def _compact_infobox(text: str, title: str = "") -> str:
    lines = [line.rstrip() for line in (text or "").splitlines() if line.strip()]
    kept: list[str] = []
    title_fold = title.casefold()
    for line in lines:
        if "infobox olympic event" in line.casefold():
            kept.append(line)
            continue
        match = INFOBOX_KEEP_RE.match(line)
        if not match:
            continue
        key, _, value = line.partition(":")
        if key.strip().casefold() == "event" and title_fold and value.strip().casefold() in title_fold:
            continue
        kept.append(line)
    return "\n".join(kept) if kept else (text or "").strip()
