"""Unit tests for TigerGraph Vector RAG helpers. No live graph or LLM."""

from __future__ import annotations

import unittest

from ingestion.embeddings import build_embedding_request, parse_embedding_response, text_digest
from ingestion.models import TextChunk
from retrieval.graph.vector import (
    VectorBackendError,
    VectorHit,
    map_vector_hits_to_chunks,
    parse_vector_attributes_from_ls,
    parse_vector_search_result,
)
from retrieval.rag.hybrid import rrf_fuse
from retrieval.rag.hybrid_vector import HybridBM25VectorRetriever
from retrieval.rag.models import RetrievalHit, RetrievalResult
from retrieval.rag.vector import TigerGraphVectorRetriever


def _chunk(chunk_id: str, document_id: str, text: str = "body") -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=document_id,
        source_url=None,
        text=text,
        indexed_text=f"{document_id}\n{text}",
        start_char=0,
        end_char=len(text),
        token_count=1,
        chunk_index=0,
        section="lead",
        kind="lead",
        event_id=document_id if document_id.startswith("Q") else None,
        document_kind="event",
    )


def _hit(chunk_id: str, document_id: str, rank: int, score: float = 1.0, method: str = "t") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=document_id,
        score=score,
        rank=rank,
        text="x",
        section="lead",
        kind="lead",
        event_id=None,
        retrieval_method=method,
    )


class EmbeddingRequestTests(unittest.TestCase):
    def test_build_embedding_request_single_and_batch(self) -> None:
        single = build_embedding_request("@cf/baai/bge-small-en-v1.5", "judo 2016")
        self.assertEqual(single["model"], "@cf/baai/bge-small-en-v1.5")
        self.assertEqual(single["input"], "judo 2016")
        batch = build_embedding_request("@cf/baai/bge-small-en-v1.5", ["a", "b"])
        self.assertEqual(batch["input"], ["a", "b"])

    def test_parse_embedding_response_orders_by_index(self) -> None:
        raw = {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }
        vectors = parse_embedding_response(raw, expected=2)
        self.assertEqual(vectors[0], [1.0, 0.0])
        self.assertEqual(vectors[1], [0.0, 1.0])

    def test_text_digest_is_stable(self) -> None:
        self.assertEqual(text_digest("  Judo   2016 "), text_digest("Judo 2016"))


class VectorParseTests(unittest.TestCase):
    def test_parse_ls_vector_attributes(self) -> None:
        ls = """
Using graph 'OlympicGraph'
Vertex Types:
  - VERTEX Chunk(...)
Vector Embeddings:
  - Chunk:
    - embedding(Dimension=384, IndexType="HNSW", DataType="FLOAT", Metric="COSINE")
Queries:
  - lookup_event()
"""
        attrs = parse_vector_attributes_from_ls(ls)
        self.assertEqual(len(attrs), 1)
        self.assertEqual(attrs[0]["vertex_type"], "Chunk")
        self.assertEqual(attrs[0]["vector_name"], "embedding")
        self.assertEqual(attrs[0]["dimension"], 384)
        self.assertEqual(attrs[0]["metric"], "COSINE")

    def test_parse_vector_search_result_and_duplicates(self) -> None:
        raw = [
            {
                "v": [
                    {"v_id": "Q1::c000", "attributes": {"document_id": "Q1"}},
                    {"v_id": "Q1::c000", "attributes": {"document_id": "Q1"}},
                    {"v_id": "Q2::c000", "attributes": {"document_id": "Q2"}},
                ]
            },
            {"distances": {"Q2::c000": 0.1, "Q1::c000": 0.4}},
        ]
        hits = parse_vector_search_result(raw)
        self.assertEqual([hit.chunk_id for hit in hits], ["Q2::c000", "Q1::c000"])
        self.assertEqual(hits[0].rank, 1)

    def test_empty_vector_results(self) -> None:
        self.assertEqual(parse_vector_search_result([{"v": []}, {"distances": {}}]), [])

    def test_chunk_identity_mapping_drops_unknown_ids(self) -> None:
        chunks = {"Q1::c000": _chunk("Q1::c000", "Q1", "infobox")}
        hits = [
            VectorHit("Q1::c000", 0.9, 1, {}),
            VectorHit("missing::c000", 0.8, 2, {}),
        ]
        mapped = map_vector_hits_to_chunks(hits, chunks)
        self.assertEqual([hit.chunk_id for hit in mapped], ["Q1::c000"])
        self.assertEqual(mapped[0].document_id, "Q1")
        self.assertEqual(mapped[0].text, "infobox")
        self.assertEqual(mapped[0].retrieval_method, "tigergraph_vector")

    def test_stable_tie_order(self) -> None:
        raw = [
            {"v": [{"v_id": "b::c000"}, {"v_id": "a::c000"}]},
            {"distances": {"a::c000": 0.2, "b::c000": 0.2}},
        ]
        hits = parse_vector_search_result(raw)
        self.assertEqual([hit.chunk_id for hit in hits], ["a::c000", "b::c000"])


class HybridFusionTests(unittest.TestCase):
    def test_rrf_duplicate_and_tie_handling(self) -> None:
        left = [_hit("c1", "d1", 1), _hit("c2", "d2", 2)]
        right = [_hit("c2", "d2", 1), _hit("c1", "d1", 2)]
        fused = rrf_fuse([left, right], rrf_k=60, top_k=2, query="q")
        self.assertEqual(len(fused.hits), 2)
        ids = [hit.chunk_id for hit in fused.hits]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, sorted(ids))

    def test_hybrid_refuses_single_backend_modes(self) -> None:
        hybrid = HybridBM25VectorRetriever(FakeSparse(), FakeVector())
        with self.assertRaises(VectorBackendError):
            hybrid.retrieve("q", method="sparse")
        with self.assertRaises(VectorBackendError):
            hybrid.retrieve("q", method="tigergraph_vector")

    def test_hybrid_records_both_backends(self) -> None:
        hybrid = HybridBM25VectorRetriever(FakeSparse(), FakeVector())
        result = hybrid.retrieve("q", method="hybrid", top_k=2)
        self.assertEqual(result.params["backends"], ["bm25", "tigergraph_vector"])
        self.assertEqual(result.method, "hybrid_rrf")
        self.assertTrue(result.hits)

    def test_vector_retriever_refuses_bm25(self) -> None:
        retriever = TigerGraphVectorRetriever(FakeStore(), [_chunk("Q1::c000", "Q1")], FakeEmbedder())
        with self.assertRaises(VectorBackendError):
            retriever.retrieve("q", method="sparse")

    def test_vector_backend_failure(self) -> None:
        retriever = TigerGraphVectorRetriever(FailingStore(), [_chunk("Q1::c000", "Q1")], FakeEmbedder())
        with self.assertRaises(VectorBackendError):
            retriever.retrieve("q", method="tigergraph_vector")

    def test_empty_vector_search_does_not_become_bm25(self) -> None:
        retriever = TigerGraphVectorRetriever(FakeStore(hits=[]), [_chunk("Q1::c000", "Q1")], FakeEmbedder())
        result = retriever.retrieve("q", method="tigergraph_vector")
        self.assertEqual(result.hits, [])
        self.assertEqual(result.params["backend"], "tigergraph_vector")
        self.assertFalse(result.params["bm25"])


class FakeSparse:
    def retrieve(self, query, *, top_k=10, method="sparse", **kwargs):
        del query, kwargs
        hits = [_hit("c1", "d1", 1, method="bm25"), _hit("c2", "d2", 2, method="bm25")][:top_k]
        return RetrievalResult(
            query="q",
            method="bm25+candidates",
            hits=hits,
            params={"backend": "bm25", "candidate_chunk_ids": [hit.chunk_id for hit in hits], "candidate_document_ids": [hit.document_id for hit in hits]},
        )


class FakeVector:
    def retrieve(self, query, *, top_k=10, method="tigergraph_vector", **kwargs):
        del query, kwargs, method
        hits = [_hit("c2", "d2", 1, method="tigergraph_vector"), _hit("c3", "d3", 2, method="tigergraph_vector")][:top_k]
        return RetrievalResult(
            query="q",
            method="tigergraph_vector",
            hits=hits,
            params={"backend": "tigergraph_vector", "candidate_chunk_ids": [hit.chunk_id for hit in hits], "candidate_document_ids": [hit.document_id for hit in hits]},
        )


class FakeEmbedder:
    model = "test-embed"
    dimension = 3

    def embed_query(self, text: str) -> list[float]:
        del text
        return [0.1, 0.2, 0.3]


class FakeStore:
    def __init__(self, hits=None) -> None:
        self.hits = hits if hits is not None else [VectorHit("Q1::c000", 0.9, 1, {})]

    def search(self, query_vector, *, top_k: int):
        del query_vector
        return list(self.hits)[:top_k]


class FailingStore:
    def search(self, query_vector, *, top_k: int):
        del query_vector, top_k
        raise VectorBackendError("index missing")


class _Settings:
    graphname = "OlympicGraph"


class _RecordingStore:
    def __init__(self, client, *, dimension: int) -> None:
        del client
        self.dimension = dimension
        self.calls: list[str] = []
        type(self).last = self
        _RecordingStore.last = self

    last = None

    def ensure_schema(self):
        self.calls.append("ensure_schema")
        return {"created": False, "spec": {"dimension": 384}}

    def ensure_search_query(self):
        self.calls.append("ensure_search_query")
        return {"query": "search_chunk_embedding", "installed": True}

    def require_schema(self):
        self.calls.append("require_schema")
        return {"created": False, "spec": {"dimension": 384}, "ensured": False}

    def require_search_query(self):
        self.calls.append("require_search_query")
        return {"query": "search_chunk_embedding", "installed": True, "ensured": False}

    def require_index(self):
        self.calls.append("require_index")
        return {"ready": True, "status": {"NeedRebuildServers": []}, "ensured": False}

    def upsert_vectors(self, rows):
        self.calls.append("upsert_vectors")
        return len(list(rows))

    def wait_until_ready(self):
        self.calls.append("wait_until_ready")
        return {"ready": True}


class _MissingQueryStore(_RecordingStore):
    def require_search_query(self):
        self.calls.append("require_search_query")
        raise VectorBackendError(
            "search_chunk_embedding is not installed; "
            "refusing CREATE OR REPLACE / INSTALL because skip-ensure is set"
        )


class _IngestEmbedder:
    model = "BAAI/bge-small-en-v1.5"
    dimension = 384
    backend = "fastembed"
    settings = type("S", (), {"batch_size": 16})()

    def embed_texts(self, texts):
        del texts
        raise AssertionError("chunk embedding generation must not run")


class SkipEnsureIngestTests(unittest.TestCase):
    def test_default_ingest_still_ensures_schema_and_query(self) -> None:
        from unittest.mock import patch

        from scripts.ingest_chunk_vectors import ingest_chunk_vectors

        with patch("scripts.ingest_chunk_vectors.TigerGraphVectorStore", _RecordingStore):
            summary = ingest_chunk_vectors(
                client=object(),
                chunks=[],
                embedder=_IngestEmbedder(),
                cache=object(),
                skip_upsert=True,
                skip_ensure=False,
            )
        store = _RecordingStore.last
        self.assertIsNotNone(store)
        self.assertEqual(store.calls, ["ensure_schema", "ensure_search_query"])
        self.assertFalse(summary["skip_ensure"])
        self.assertEqual(summary["upserted"], 0)

    def test_skip_ensure_performs_zero_ddl_install_or_upsert(self) -> None:
        from unittest.mock import patch

        from scripts.ingest_chunk_vectors import ingest_chunk_vectors

        with patch("scripts.ingest_chunk_vectors.TigerGraphVectorStore", _RecordingStore):
            summary = ingest_chunk_vectors(
                client=object(),
                chunks=[_chunk("Q1::c000", "Q1")],
                embedder=_IngestEmbedder(),
                cache=None,
                skip_upsert=True,
                skip_ensure=True,
            )
        store = _RecordingStore.last
        self.assertEqual(store.calls, ["require_schema", "require_search_query", "require_index"])
        self.assertNotIn("ensure_schema", store.calls)
        self.assertNotIn("ensure_search_query", store.calls)
        self.assertNotIn("upsert_vectors", store.calls)
        self.assertNotIn("wait_until_ready", store.calls)
        self.assertTrue(summary["skip_ensure"])
        self.assertEqual(summary["upserted"], 0)
        self.assertEqual(summary["embedded"], 0)

    def test_skip_ensure_fails_closed_when_search_query_missing(self) -> None:
        from unittest.mock import patch

        from scripts.ingest_chunk_vectors import ingest_chunk_vectors

        with patch("scripts.ingest_chunk_vectors.TigerGraphVectorStore", _MissingQueryStore):
            with self.assertRaises(VectorBackendError) as raised:
                ingest_chunk_vectors(
                    client=object(),
                    chunks=[],
                    embedder=_IngestEmbedder(),
                    cache=None,
                    skip_upsert=True,
                    skip_ensure=True,
                )
        self.assertIn("not installed", str(raised.exception))
        self.assertIn("refusing CREATE OR REPLACE", str(raised.exception))
        self.assertEqual(_RecordingStore.last.calls, ["require_schema", "require_search_query"])

    def test_require_search_query_does_not_run_gsql(self) -> None:
        from retrieval.graph.vector_store import TigerGraphVectorStore

        class Client:
            settings = _Settings()
            gsql_calls: list[str] = []

            def installed_queries(self):
                return ["search_chunk_embedding"]

            def gsql(self, command: str) -> str:
                self.gsql_calls.append(command)
                return command

        client = Client()
        store = TigerGraphVectorStore(client, dimension=384)
        payload = store.require_search_query()
        self.assertEqual(payload["query"], "search_chunk_embedding")
        self.assertEqual(client.gsql_calls, [])

    def test_ensure_search_query_still_emits_create_or_replace(self) -> None:
        from retrieval.graph.vector import search_query_gsql
        from retrieval.graph.vector_store import TigerGraphVectorStore

        class Client:
            settings = _Settings()
            gsql_calls: list[str] = []

            def installed_queries(self):
                return ["search_chunk_embedding"]

            def gsql(self, command: str) -> str:
                self.gsql_calls.append(command)
                return command

        store = TigerGraphVectorStore(Client(), dimension=384)
        store.ensure_search_query()
        self.assertEqual(len(store.client.gsql_calls), 1)
        self.assertIn("CREATE OR REPLACE QUERY search_chunk_embedding", store.client.gsql_calls[0])
        self.assertIn("INSTALL QUERY search_chunk_embedding", store.client.gsql_calls[0])
        self.assertIn("CREATE OR REPLACE QUERY", search_query_gsql())

    def test_require_schema_fails_without_altering(self) -> None:
        from retrieval.graph.vector_store import TigerGraphVectorStore

        class Client:
            settings = _Settings()
            gsql_calls: list[str] = []

            def gsql(self, command: str) -> str:
                self.gsql_calls.append(command)
                return "Using graph 'OlympicGraph'\nVertex Types:\n"

        store = TigerGraphVectorStore(Client(), dimension=384)
        with self.assertRaises(VectorBackendError) as raised:
            store.require_schema()
        self.assertIn("refusing schema change", str(raised.exception))
        self.assertTrue(all("ALTER VERTEX" not in cmd for cmd in store.client.gsql_calls))


class EvalSkipEnsureFlagTests(unittest.TestCase):
    def test_skip_ensure_is_opt_in_and_implies_skip_upsert_in_eval(self) -> None:
        from scripts.eval_vector_ablation import parse_eval_args

        default = parse_eval_args([])
        self.assertFalse(default.skip_ensure)
        self.assertFalse(default.skip_ingest)
        flags = parse_eval_args(["--skip-ingest", "--skip-ensure", "--generator", "semantic"])
        self.assertTrue(flags.skip_ingest)
        self.assertTrue(flags.skip_ensure)
        self.assertEqual(flags.generator, "semantic")

    def test_v0_v1_v2_methods_unchanged(self) -> None:
        from answering.generator import DeterministicGroundedGenerator
        from answering.packer import ContextPacker
        from scripts.eval_vector_ablation import variant_adapters

        adapters = variant_adapters(
            FakeSparse(),
            FakeVector(),
            HybridBM25VectorRetriever(FakeSparse(), FakeVector()),
            ContextPacker(),
            DeterministicGroundedGenerator(),
            top_k=10,
        )
        self.assertEqual(list(adapters), ["V0", "V1", "V2"])
        self.assertEqual(adapters["V0"].method, "sparse")
        self.assertEqual(adapters["V1"].method, "tigergraph_vector")
        self.assertEqual(adapters["V2"].method, "hybrid")


if __name__ == "__main__":
    unittest.main()
