"""Load corpus JSONL into parsed documents."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ingestion.models import ParseCorpusResult, ParsedDocument
from ingestion.parse import parse_document
from ingestion.paths import DEFAULT_CORPUS_PATH


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield record


def parse_corpus(path: str | Path | None = None) -> ParseCorpusResult:
    corpus_path = Path(path) if path is not None else DEFAULT_CORPUS_PATH
    documents = [parse_document(record) for record in iter_jsonl(corpus_path)]
    return ParseCorpusResult(documents=documents)


def load_questions(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))
