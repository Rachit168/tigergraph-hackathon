"""Tests for sparse / dense / hybrid text retrieval."""

from __future__ import annotations

import unittest

from ingestion.chunker import chunk_corpus
from ingestion.parse import parse_document
from retrieval.rag.dense import DenseRetriever
from retrieval.rag.hybrid import rrf_fuse
from retrieval.rag.models import RetrievalHit
from retrieval.rag.retriever import TextRetriever
from retrieval.rag.sparse import BM25Index, tokenize

CANOE_TITLE = "Canoeing at the 2012 Summer Olympics – Men's K-2 1000 metres"
CANOE_TEXT = """[Infobox Olympic event]
  event: Men's canoe sprint K-2 1,000 metres
  games: 2012 Summer
  venue: Eton Dorney
  date: 6 to 8 August
  competitors: 24
  nations: 12
  gold: Rudolf DombiRoland Kökény
  prev: 2008
  next: 2016

The men's canoe sprint K-2 1,000 metres competition at the 2012 Olympic Games in London took place between 6 and 8 August at Eton Dorney.
"""

JUDO_TITLE = "Judo at the 2016 Summer Olympics – Women's 57 kg"
JUDO_TEXT = """[Infobox Olympic event]
  event: Women's 57 kg
  games: 2016 Summer
  venue: Carioca Arena 2
  date: 8 August
  competitors: 23
  nations: 23
  gold: Rafaela Silva

The women's 57 kg judo event was held on 8 August 2016.
"""

FILM_TEXT = "Forrest Gump is a 1994 American comedy-drama film starring Tom Hanks."


def _chunks():
    docs = [
        parse_document({"doc_id": "Qcanoe", "title": CANOE_TITLE, "text": CANOE_TEXT}),
        parse_document({"doc_id": "Qjudo", "title": JUDO_TITLE, "text": JUDO_TEXT}),
        parse_document({"doc_id": "Qfilm", "title": "Forrest Gump", "text": FILM_TEXT}),
    ]
    return chunk_corpus(docs), docs


class SparseRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks, self.docs = _chunks()
        self.index = BM25Index(self.chunks)

    def test_lookup_query_finds_judo_document(self) -> None:
        result = self.index.retrieve("How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?", top_k=5)
        self.assertTrue(result.hits)
        self.assertEqual(result.hits[0].document_id, "Qjudo")
        self.assertEqual(result.hits[0].retrieval_method, "bm25")

    def test_top_k(self) -> None:
        result = self.index.retrieve("Olympics", top_k=2)
        self.assertLessEqual(len(result.hits), 2)
        ranks = [hit.rank for hit in result.hits]
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    def test_deterministic_repeat(self) -> None:
        a = self.index.retrieve("Eton Dorney canoe sprint", top_k=5)
        b = self.index.retrieve("Eton Dorney canoe sprint", top_k=5)
        self.assertEqual([hit.chunk_id for hit in a.hits], [hit.chunk_id for hit in b.hits])
        self.assertEqual([hit.score for hit in a.hits], [hit.score for hit in b.hits])

    def test_tokenize(self) -> None:
        self.assertEqual(tokenize("Women's 57 kg"), ["women", "s", "57", "kg"])


class DenseAndHybridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks, _docs = _chunks()
        self.retriever = TextRetriever(self.chunks)

    def test_dense_unavailable(self) -> None:
        dense = DenseRetriever()
        self.assertFalse(dense.available)
        result = dense.retrieve("judo", top_k=5)
        self.assertEqual(result.hits, [])

    def test_hybrid_equals_sparse_when_dense_empty(self) -> None:
        sparse = self.retriever.sparse.retrieve("Women's 57 kg judo", top_k=5)
        hybrid = self.retriever.retrieve("Women's 57 kg judo", top_k=5, method="hybrid")
        self.assertEqual([hit.chunk_id for hit in sparse.hits], [hit.chunk_id for hit in hybrid.hits])

    def test_rrf_fusion_and_ties(self) -> None:
        def hit(chunk_id: str, doc: str, rank: int, score: float = 1.0) -> RetrievalHit:
            return RetrievalHit(
                chunk_id=chunk_id,
                document_id=doc,
                document_title=doc,
                score=score,
                rank=rank,
                text="x",
                section="lead",
                kind="lead",
                event_id=None,
                retrieval_method="t",
            )

        left = [hit("c1", "d1", 1), hit("c2", "d2", 2)]
        right = [hit("c2", "d2", 1), hit("c3", "d3", 2)]
        fused = rrf_fuse([left, right], rrf_k=60, top_k=3, query="q")
        self.assertEqual(fused.hits[0].chunk_id, "c2")
        ids = [item.chunk_id for item in fused.hits]
        self.assertEqual(ids, sorted(ids, key=lambda cid: (-next(h.score for h in fused.hits if h.chunk_id == cid), cid)))

    def test_event_document_separation(self) -> None:
        film_chunks = [chunk for chunk in self.chunks if chunk.document_id == "Qfilm"]
        self.assertTrue(film_chunks)
        self.assertTrue(all(chunk.event_id is None for chunk in film_chunks))
        judo = [chunk for chunk in self.chunks if chunk.document_id == "Qjudo"]
        self.assertTrue(all(chunk.event_id == "Qjudo" for chunk in judo))
