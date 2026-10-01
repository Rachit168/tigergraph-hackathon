"""Filesystem helpers for local hackathon resources."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCES_ROOT = REPO_ROOT / "_research" / "hackathon-resources"
DEFAULT_CORPUS_PATH = RESOURCES_ROOT / "corpus" / "corpus.jsonl"
DEFAULT_PUBLIC_QUESTIONS_PATH = RESOURCES_ROOT / "questions" / "eval_public.jsonl"
GSQL_DIR = REPO_ROOT / "gsql"
DEFAULT_GRAPH_EXPORT_DIR = REPO_ROOT / "data" / "tigergraph" / "export"
DEFAULT_GRAPH_VERIFY_PATH = REPO_ROOT / "data" / "tigergraph" / "verify.json"
