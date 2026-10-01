"""Reciprocal Rank Fusion and optional deterministic MMR."""

from __future__ import annotations

from retrieval.rag.models import RetrievalHit, RetrievalResult


def rrf_fuse(
    ranked_lists: list[list[RetrievalHit]],
    *,
    rrf_k: int = 60,
    top_k: int = 10,
    method: str = "hybrid_rrf",
    query: str = "",
) -> RetrievalResult:
    scores: dict[str, float] = {}
    reps: dict[str, RetrievalHit] = {}
    for hits in ranked_lists:
        for hit in hits:
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + hit.rank)
            reps.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    fused: list[RetrievalHit] = []
    for rank, (chunk_id, score) in enumerate(ordered[: max(top_k, 0)], start=1):
        source = reps[chunk_id]
        fused.append(
            RetrievalHit(
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                document_title=source.document_title,
                score=score,
                rank=rank,
                text=source.text,
                section=source.section,
                kind=source.kind,
                event_id=source.event_id,
                retrieval_method=method,
                source_url=source.source_url,
            )
        )
    return RetrievalResult(
        query=query,
        method=method,
        hits=fused,
        params={"rrf_k": rrf_k, "top_k": top_k, "n_lists": len(ranked_lists)},
    )


def mmr_rerank(
    hits: list[RetrievalHit],
    *,
    lambda_mult: float = 0.7,
    top_k: int | None = None,
) -> list[RetrievalHit]:
    """Optional diversity rerank. Token Jaccard vs selected items. Deterministic."""
    if not hits:
        return []
    limit = top_k or len(hits)
    remaining = list(hits)
    selected: list[RetrievalHit] = []
    token_sets = [_tokens(hit.text) for hit in remaining]
    while remaining and len(selected) < limit:
        best_i = 0
        best_score = float("-inf")
        for index, hit in enumerate(remaining):
            relevance = hit.score
            if selected:
                diversity = max(
                    _jaccard(token_sets[index], _tokens(item.text)) for item in selected
                )
            else:
                diversity = 0.0
            mmr = lambda_mult * relevance - (1.0 - lambda_mult) * diversity
            key = (mmr, -index, hit.chunk_id)
            current = (best_score, -best_i, remaining[best_i].chunk_id)
            if key > current:
                best_score = mmr
                best_i = index
        chosen = remaining.pop(best_i)
        token_sets.pop(best_i)
        selected.append(chosen)
    return [
        RetrievalHit(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            document_title=hit.document_title,
            score=hit.score,
            rank=rank,
            text=hit.text,
            section=hit.section,
            kind=hit.kind,
            event_id=hit.event_id,
            retrieval_method="mmr",
            source_url=hit.source_url,
        )
        for rank, hit in enumerate(selected, start=1)
    ]


def _tokens(text: str) -> set[str]:
    return {part for part in (text or "").casefold().split() if part}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left & right)
    union = len(left | right)
    return inter / union if union else 0.0
