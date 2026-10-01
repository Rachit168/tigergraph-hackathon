"""Deterministic Olympic event parser and text chunker."""

from ingestion.chunker import chunk_corpus, chunk_document
from ingestion.loader import load_questions, parse_corpus
from ingestion.models import ChunkingConfig, DateSpan, ParsedDocument, ParsedEvent, Provenance, TextChunk, TitleParse
from ingestion.parse import parse_document, parse_title
from ingestion.paths import DEFAULT_CORPUS_PATH, DEFAULT_PUBLIC_QUESTIONS_PATH

__all__ = [
    "ChunkingConfig",
    "DEFAULT_CORPUS_PATH",
    "DEFAULT_PUBLIC_QUESTIONS_PATH",
    "DateSpan",
    "ParsedDocument",
    "ParsedEvent",
    "Provenance",
    "TextChunk",
    "TitleParse",
    "chunk_corpus",
    "chunk_document",
    "load_questions",
    "parse_corpus",
    "parse_document",
    "parse_title",
]
