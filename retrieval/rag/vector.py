"""TigerGraph Vector RAG retriever. No BM25 fallback."""

from __future__ import annotations

import time
from typing import Any

from ingestion.embeddings import EmbeddingError, OpenAICompatibleEmbedder
from ingestion.models import TextChunk
from retrieval.graph.vector import VectorBackendError, map_vector_hits_to_chunks
from retrieval.graph.vector_store import TigerGraphVectorStore
from retrieval.rag.models import RetrievalResult

VECTOR_METHODS = frozenset({"dense", "vector", "tigergraph_vector"})
FORBIDDEN_METHODS = frozenset({"sparse", "bm25", "hybrid", "hybrid_rrf"})


class TigerGraphVectorRetriever:
    def __init__(
        self,
        store: TigerGraphVectorStore,
        chunks: list[TextChunk],
        embedder: OpenAICompatibleEmbedder,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunks = chunks
        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.available = True

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        method: str = "tigergraph_vector",
        **kwargs: Any,
    ) -> RetrievalResult:
        del kwargs
        started = time.perf_counter()
        if method in FORBIDDEN_METHODS:
            raise VectorBackendError(f"TigerGraph vector retriever refuses method={method}")
        if method not in VECTOR_METHODS:
            raise VectorBackendError(f"unsupported vector method: {method}")
        try:
            query_vector = self.embedder.embed_query(query)
            hits = self.store.search(query_vector, top_k=top_k)
        except EmbeddingError as exc:
            raise VectorBackendError(f"embedding failed: {exc}") from exc
        mapped = map_vector_hits_to_chunks(hits, self.chunks_by_id)
        unmapped = [hit.chunk_id for hit in hits if hit.chunk_id not in self.chunks_by_id]
        result = RetrievalResult(
            query=query,
            method="tigergraph_vector",
            hits=mapped[: max(top_k, 0)],
            params={
                "backend": "tigergraph_vector",
                "top_k": top_k,
                "raw_hit_count": len(hits),
                "mapped_hit_count": len(mapped),
                "unmapped_chunk_ids": unmapped,
                "candidate_chunk_ids": [hit.chunk_id for hit in mapped],
                "candidate_document_ids": _unique([hit.document_id for hit in mapped]),
                "embedding_model": self.embedder.model,
                "embedding_dimension": self.embedder.dimension,
                "graph_operations": False,
                "bm25": False,
            },
        )
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
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
