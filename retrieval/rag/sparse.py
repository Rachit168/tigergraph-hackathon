"""Okapi BM25 over chunk indexed_text. Stdlib only."""

from __future__ import annotations

import math
import re
from collections import defaultdict

from ingestion.models import TextChunk
from retrieval.rag.models import RetrievalHit, RetrievalResult

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").casefold())


class BM25Index:
    def __init__(self, chunks: list[TextChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._df: dict[str, int] = defaultdict(int)
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._dl: list[int] = []
        for index, chunk in enumerate(self.chunks):
            tf: dict[str, int] = defaultdict(int)
            tokens = tokenize(chunk.indexed_text)
            for token in tokens:
                tf[token] += 1
            self._dl.append(len(tokens) or 1)
            for token, count in tf.items():
                self._df[token] += 1
                self._postings[token].append((index, count))
        self.n = len(self.chunks) or 1
        self.avgdl = (sum(self._dl) / self.n) if self.chunks else 1.0
        self._idf = {
            token: math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            for token, df in self._df.items()
        }

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        scores: dict[int, float] = defaultdict(float)
        for token in tokenize(query):
            idf = self._idf.get(token)
            if idf is None:
                continue
            for index, freq in self._postings.get(token, ()):
                denom = freq + self.k1 * (1.0 - self.b + self.b * self._dl[index] / self.avgdl)
                scores[index] += idf * (freq * (self.k1 + 1.0)) / denom
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self.chunks[item[0]].chunk_id),
        )
        hits: list[RetrievalHit] = []
        for rank, (index, score) in enumerate(ranked[: max(top_k, 0)], start=1):
            chunk = self.chunks[index]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    score=score,
                    rank=rank,
                    text=chunk.text,
                    section=chunk.section,
                    kind=chunk.kind,
                    event_id=chunk.event_id,
                    retrieval_method="bm25",
                    source_url=chunk.source_url,
                )
            )
        return RetrievalResult(query=query, method="bm25", hits=hits, params={"k1": self.k1, "b": self.b, "top_k": top_k})
