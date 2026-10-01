"""Deterministic BM25 + TigerGraph Vector fusion. Both backends must run."""

from __future__ import annotations

import time
from typing import Any

from retrieval.graph.vector import VectorBackendError
from retrieval.rag.hybrid import rrf_fuse
from retrieval.rag.models import RetrievalResult
from retrieval.rag.retriever import TextRetriever
from retrieval.rag.vector import TigerGraphVectorRetriever

HYBRID_METHODS = frozenset({"hybrid", "hybrid_rrf", "bm25_vector"})
SPARSE_METHODS = frozenset({"sparse", "bm25"})
VECTOR_METHODS = frozenset({"dense", "vector", "tigergraph_vector"})


class HybridBM25VectorRetriever:
    def __init__(
        self,
        sparse: TextRetriever,
        vector: TigerGraphVectorRetriever,
        *,
        rrf_k: int = 60,
    ) -> None:
        self.sparse = sparse
        self.vector = vector
        self.rrf_k = rrf_k
        self.available = True

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        method: str = "hybrid",
        sparse_top_k: int | None = None,
        dense_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        del kwargs
        started = time.perf_counter()
        if method in SPARSE_METHODS:
            raise VectorBackendError("hybrid retriever refuses BM25-only mode")
        if method in VECTOR_METHODS:
            raise VectorBackendError("hybrid retriever refuses vector-only mode")
        if method not in HYBRID_METHODS:
            raise VectorBackendError(f"unsupported hybrid method: {method}")
        sparse_k = sparse_top_k or max(top_k, 50)
        vector_k = dense_top_k or max(top_k, 50)
        sparse_result = self.sparse.retrieve(query, top_k=sparse_k, method="sparse")
        vector_result = self.vector.retrieve(query, top_k=vector_k, method="tigergraph_vector")
        if sparse_result.params.get("backend") != "bm25":
            raise VectorBackendError("hybrid sparse arm did not run BM25")
        if vector_result.params.get("backend") != "tigergraph_vector":
            raise VectorBackendError("hybrid vector arm did not run TigerGraph vector search")
        fused = rrf_fuse(
            [sparse_result.hits, vector_result.hits],
            rrf_k=self.rrf_k,
            top_k=top_k,
            method="hybrid_rrf",
            query=query,
        )
        candidate_chunks = _unique(
            list(sparse_result.params.get("candidate_chunk_ids") or [hit.chunk_id for hit in sparse_result.hits])
            + list(vector_result.params.get("candidate_chunk_ids") or [hit.chunk_id for hit in vector_result.hits])
        )
        candidate_docs = _unique(
            list(sparse_result.params.get("candidate_document_ids") or [hit.document_id for hit in sparse_result.hits])
            + list(vector_result.params.get("candidate_document_ids") or [hit.document_id for hit in vector_result.hits])
        )
        fused.params.update(
            {
                "backend": "hybrid_rrf",
                "backends": ["bm25", "tigergraph_vector"],
                "rrf_k": self.rrf_k,
                "sparse_hits": len(sparse_result.hits),
                "vector_hits": len(vector_result.hits),
                "sparse_empty": not sparse_result.hits,
                "vector_empty": not vector_result.hits,
                "candidate_chunk_ids": candidate_chunks,
                "candidate_document_ids": candidate_docs,
                "graph_operations": False,
            }
        )
        fused.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return fused


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
