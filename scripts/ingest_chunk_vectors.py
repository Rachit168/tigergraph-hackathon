"""Ingest Chunk embeddings into TigerGraph Vector attributes."""

from __future__ import annotations

import argparse
from typing import Any

from config.embeddings import embedding_cache_path, load_embedding_settings
from ingestion.chunker import ChunkingConfig, chunk_corpus
from ingestion.embeddings import EmbeddingCache, EmbeddingError, build_embedder, text_digest
from ingestion.loader import parse_corpus
from ingestion.models import TextChunk
from ingestion.paths import DEFAULT_CORPUS_PATH
from retrieval.graph.client import TigerGraphClient
from retrieval.graph.vector import VectorBackendError
from retrieval.graph.vector_store import TigerGraphVectorStore


def ingest_chunk_vectors(
    *,
    client: TigerGraphClient,
    chunks: list[TextChunk],
    embedder: Any,
    cache: EmbeddingCache | None,
    skip_upsert: bool = False,
    skip_ensure: bool = False,
) -> dict[str, Any]:
    store = TigerGraphVectorStore(client, dimension=embedder.dimension)
    if skip_ensure:
        schema = store.require_schema()
        query = store.require_search_query()
        index_status = store.require_index()
        return {
            "n_chunks": len(chunks),
            "cache_hits": 0,
            "embedded": 0,
            "upserted": 0,
            "schema": schema,
            "query": query,
            "index": index_status,
            "model": embedder.model,
            "dimension": embedder.dimension,
            "vector_attribute": "embedding",
            "search_query": "search_chunk_embedding",
            "embedding_backend": getattr(embedder, "backend", "http"),
            "skip_ensure": True,
            "skip_upsert": True,
        }
    schema = store.ensure_schema()
    query = store.ensure_search_query()
    pending: list[TextChunk] = []
    ready: list[tuple[str, list[float]]] = []
    for chunk in chunks:
        digest = text_digest(chunk.indexed_text or chunk.text)
        vector = cache.get(chunk.chunk_id, digest, embedder.model, embedder.dimension)
        if vector is None:
            pending.append(chunk)
        else:
            ready.append((chunk.chunk_id, vector))
    print(f"cache_hits {len(ready)} pending {len(pending)} model {embedder.model}", flush=True)
    embedded = 0
    batch_size = 256 if getattr(embedder, "backend", "") == "fastembed" else max(1, embedder.settings.batch_size)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        try:
            vectors = embedder.embed_texts([chunk.indexed_text or chunk.text for chunk in batch])
        except Exception:
            raise
        staged: list[tuple[str, str, str, int, list[float]]] = []
        for chunk, vector in zip(batch, vectors, strict=True):
            digest = text_digest(chunk.indexed_text or chunk.text)
            staged.append((chunk.chunk_id, digest, embedder.model, embedder.dimension, vector))
            ready.append((chunk.chunk_id, vector))
        cache.put_many(staged)
        embedded += len(batch)
        print(f"embedded {embedded}/{len(pending)}", flush=True)
    upserted = 0
    index_status: dict[str, Any] = {}
    if not skip_upsert:
        print(f"upserting {len(ready)} vectors", flush=True)
        upserted = store.upsert_vectors(ready)
        index_status = store.wait_until_ready()
    return {
        "n_chunks": len(chunks),
        "cache_hits": len(chunks) - len(pending),
        "embedded": embedded,
        "upserted": upserted,
        "schema": schema,
        "query": query,
        "index": index_status,
        "model": embedder.model,
        "dimension": embedder.dimension,
        "vector_attribute": "embedding",
        "search_query": "search_chunk_embedding",
        "embedding_backend": getattr(embedder, "backend", "http"),
        "skip_ensure": False,
        "skip_upsert": skip_upsert,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed chunks and upsert TigerGraph vectors")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--skip-upsert", action="store_true")
    args = parser.parse_args()
    settings = load_embedding_settings()
    client = TigerGraphClient()
    if not client.connect():
        raise SystemExit(client.error or "TigerGraph is not reachable")
    corpus = parse_corpus(args.corpus)
    chunks = chunk_corpus(corpus.documents, ChunkingConfig())
    embedder = build_embedder(settings)
    cache = EmbeddingCache(embedding_cache_path(embedder.model))
    try:
        summary = ingest_chunk_vectors(
            client=client,
            chunks=chunks,
            embedder=embedder,
            cache=cache,
            skip_upsert=args.skip_upsert,
        )
    except (VectorBackendError, EmbeddingError) as exc:
        raise SystemExit(f"vector ingest failed: {exc}") from exc
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
