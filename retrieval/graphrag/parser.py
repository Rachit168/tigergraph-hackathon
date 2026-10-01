"""Adapter from the shared structured parser to a typed retrieval request.

Question templates, intent selection, and argument extraction intentionally
remain in retrieval.structured.question.QuestionParser.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from retrieval.graphrag.validate import validate_spec
from retrieval.structured.models import QuerySpec
from retrieval.structured.question import QuestionParser

ParseStatus = Literal["parsed", "unsupported", "unresolved"]


@dataclass(frozen=True)
class ParsedRetrievalRequest:
    question: str
    spec: QuerySpec
    status: ParseStatus
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def valid(self) -> bool:
        return self.status == "parsed"

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "status": self.status,
            "reason": self.reason,
            "warnings": list(self.warnings),
            "elapsed_ms": self.elapsed_ms,
            "spec": self.spec.to_dict(),
        }


class GraphRAGQuestionParser:
    """Typed adapter only; all parsing is delegated to QuestionParser."""

    def __init__(self, parser: QuestionParser) -> None:
        self.parser = parser

    def parse(self, question: str, qtype: str | None = None) -> ParsedRetrievalRequest:
        started = time.perf_counter()
        spec = self.parser.parse(question, qtype=qtype)
        reason = validate_spec(spec)
        status: ParseStatus = "parsed"
        if reason == "question_template_unparsed":
            status = "unsupported"
        elif reason is not None:
            status = "unresolved"
        return ParsedRetrievalRequest(
            question=question,
            spec=spec,
            status=status,
            reason=reason,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
