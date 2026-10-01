"""Corpus integration tests for chunking and BM25."""

from __future__ import annotations

import unittest

from evaluation.retrieval_gold import official_gold_ids
from ingestion.chunker import chunk_corpus
from ingestion.loader import load_questions, parse_corpus
from ingestion.paths import DEFAULT_CORPUS_PATH, DEFAULT_PUBLIC_QUESTIONS_PATH
from retrieval.rag.retriever import TextRetriever

CORPUS_AVAILABLE = DEFAULT_CORPUS_PATH.exists()


@unittest.skipUnless(CORPUS_AVAILABLE, "hackathon corpus is not present")
class CorpusChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = parse_corpus(DEFAULT_CORPUS_PATH)
        cls.chunks = chunk_corpus(cls.corpus.documents)

    def test_every_document_has_a_chunk(self) -> None:
        doc_ids = {document.doc_id for document in self.corpus.documents}
        chunk_docs = {chunk.document_id for chunk in self.chunks}
        self.assertEqual(doc_ids, chunk_docs)
        self.assertEqual(len({chunk.chunk_id for chunk in self.chunks}), len(self.chunks))

    def test_known_event_starts_with_infobox(self) -> None:
        canoe = [chunk for chunk in self.chunks if chunk.document_id == "Q303623"]
        self.assertTrue(canoe)
        self.assertEqual(canoe[0].kind, "infobox")
        self.assertEqual(canoe[0].event_id, "Q303623")

    def test_body_only_page_has_chunks_without_event(self) -> None:
        chunks = [chunk for chunk in self.chunks if chunk.document_id == "Q3046361"]
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.event_id is None for chunk in chunks))


@unittest.skipUnless(
    CORPUS_AVAILABLE and DEFAULT_PUBLIC_QUESTIONS_PATH.exists(),
    "corpus or public questions missing",
)
class SparseLookupSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = parse_corpus(DEFAULT_CORPUS_PATH)
        cls.retriever = TextRetriever(chunk_corpus(cls.corpus.documents))
        cls.questions = [q for q in load_questions(DEFAULT_PUBLIC_QUESTIONS_PATH) if q.get("qtype") == "lookup"]

    def test_lookup_gold_in_top_10(self) -> None:
        misses = 0
        for question in self.questions:
            gold = set(official_gold_ids(question))
            result = self.retriever.retrieve(question["question"], top_k=10, method="sparse")
            retrieved = {hit.document_id for hit in result.hits}
            if gold.isdisjoint(retrieved):
                misses += 1
        self.assertLessEqual(misses, 3)
