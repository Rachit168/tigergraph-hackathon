"""Assemble the console bootstrap payload. Live investigation is not preloaded."""

from __future__ import annotations

import json
from typing import Any

from ingestion.paths import DEFAULT_PUBLIC_QUESTIONS_PATH
from ui.catalog import PIPELINE_LABELS
from ui.runtime import get_runtime
from ui.viewmodels import benchmark_view, system_view

PUBLIC_QUESTION_FIELDS = ("qid", "question", "qtype")


def public_questions() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    path = DEFAULT_PUBLIC_QUESTIONS_PATH
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        qid = str(raw.get("qid") or "")
        if not qid.startswith("pub-"):
            continue
        rows.append(
            {
                "qid": qid,
                "question": str(raw.get("question") or ""),
                "qtype": str(raw.get("qtype") or ""),
            }
        )
    rows.sort(key=lambda item: item["qid"])
    return rows


def bootstrap() -> dict[str, Any]:
    capabilities = get_runtime().capabilities()
    return {
        "mode": "unavailable",
        "pipelines": [{"id": key, "name": label} for key, label in PIPELINE_LABELS.items()],
        "benchmark": benchmark_view(),
        "system": system_view(),
        "capabilities": capabilities,
        "public_questions": public_questions(),
        "example_placeholder": example_placeholder(),
    }


def example_placeholder() -> str:
    rows = public_questions()
    if not rows:
        return "Ask any question about Olympic events in the corpus."
    return rows[0]["question"]
