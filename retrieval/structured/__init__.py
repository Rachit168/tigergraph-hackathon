"""Offline structured solver over parsed Event records (Phase 2)."""

from retrieval.structured.index import StructuredIndex
from retrieval.structured.models import EvidenceItem, QuerySpec, SolverResult
from retrieval.structured.question import QuestionParser
from retrieval.structured.solver import StructuredSolver

__all__ = [
    "EvidenceItem",
    "QuerySpec",
    "QuestionParser",
    "SolverResult",
    "StructuredIndex",
    "StructuredSolver",
]
