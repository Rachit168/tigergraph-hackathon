"""Offline text retrieval substrate (Phase 3)."""

from retrieval.rag.dense import DenseRetriever
from retrieval.rag.hybrid import rrf_fuse
from retrieval.rag.models import RetrievalHit, RetrievalResult
from retrieval.rag.query import QueryPlan, plan_query
from retrieval.rag.retriever import TextRetriever
from retrieval.rag.select import evidence_text, select_evidence
from retrieval.rag.sparse import BM25Index

__all__ = [
    "BM25Index",
    "DenseRetriever",
    "QueryPlan",
    "RetrievalHit",
    "RetrievalResult",
    "TextRetriever",
    "evidence_text",
    "plan_query",
    "rrf_fuse",
    "select_evidence",
]
