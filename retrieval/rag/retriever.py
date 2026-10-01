"""Offline text retriever: sparse BM25, optional dense, RRF hybrid."""

from __future__ import annotations

import time
from collections import defaultdict

from ingestion.models import TextChunk
from retrieval.rag.dense import DenseRetriever
from retrieval.rag.hybrid import mmr_rerank, rrf_fuse
from retrieval.rag.models import RetrievalHit, RetrievalResult
from retrieval.rag.query import plan_query
from retrieval.rag.select import select_evidence
from retrieval.rag.sparse import BM25Index


class TextRetriever:
    def __init__(
        self,
        chunks: list[TextChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
        dense: DenseRetriever | None = None,
    ) -> None:
        self.chunks = chunks
        self.sparse = BM25Index(chunks, k1=k1, b=b)
        self.dense = dense or DenseRetriever()
        self.rrf_k = rrf_k
        self._by_document: dict[str, list[TextChunk]] = defaultdict(list)
        self._infoboxes: list[TextChunk] = []
        for chunk in chunks:
            self._by_document[chunk.document_id].append(chunk)
            if chunk.kind == "infobox":
                self._infoboxes.append(chunk)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        method: str = "sparse",
        sparse_top_k: int | None = None,
        dense_top_k: int | None = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
    ) -> RetrievalResult:
        started = time.perf_counter()
        sparse_k = sparse_top_k or max(top_k, 50)
        dense_k = dense_top_k or max(top_k, 50)
        if method == "sparse":
            result = self._sparse_candidates(query, top_k=top_k)
        elif method == "dense":
            result = self.dense.retrieve(query, top_k=top_k)
        elif method == "hybrid":
            sparse = self.sparse.retrieve(query, top_k=sparse_k)
            dense = self.dense.retrieve(query, top_k=dense_k)
            lists = [sparse.hits]
            if self.dense.available and dense.hits:
                lists.append(dense.hits)
            result = rrf_fuse(lists, rrf_k=self.rrf_k, top_k=top_k, query=query)
            result.params["dense_available"] = self.dense.available
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        if use_mmr and result.hits:
            result.hits = mmr_rerank(result.hits, lambda_mult=mmr_lambda, top_k=top_k)
            result.method = f"{result.method}+mmr"
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return result

    def _sparse_candidates(self, query: str, *, top_k: int) -> RetrievalResult:
        plan = plan_query(query)
        pool_k = max(top_k * 8, 80) if plan.intent in {"set_count", "set_extreme"} else max(top_k * 3, 20)
        merged: dict[str, RetrievalHit] = {}
        for variant in plan.variants:
            for hit in self.sparse.retrieve(variant, top_k=pool_k).hits:
                current = merged.get(hit.chunk_id)
                if current is None or hit.score > current.score:
                    merged[hit.chunk_id] = hit
        first_hits = sorted(merged.values(), key=lambda hit: (-hit.score, hit.chunk_id))
        first_hits = [
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
                retrieval_method=hit.retrieval_method,
                source_url=hit.source_url,
            )
            for rank, hit in enumerate(first_hits, start=1)
        ]
        result = select_evidence(
            query=query,
            plan=plan,
            first_hits=first_hits,
            chunks_by_document=self._by_document,
            infoboxes=self._infoboxes,
            top_k=top_k,
            method="bm25",
        )
        result.params["backend"] = "bm25"
        result.params["candidate_chunk_ids"] = [hit.chunk_id for hit in first_hits]
        result.params["candidate_document_ids"] = _unique([hit.document_id for hit in first_hits])
        return result


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered

