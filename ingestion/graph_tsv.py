"""Deterministic TSV / JSONL helpers for TigerGraph loading files."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def escape_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    return text.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def unescape_cell(value: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char == "\\" and index + 1 < len(value):
            nxt = value[index + 1]
            mapping = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\"}
            if nxt in mapping:
                out.append(mapping[nxt])
                index += 2
                continue
        out.append(char)
        index += 1
    return "".join(out)


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(fieldnames) + "\n")
        for row in rows:
            handle.write("\t".join(escape_cell(row.get(name, "")) for name in fieldnames) + "\n")
            count += 1
    return count


def read_tsv(path: Path) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    for line in lines[1:]:
        raw = line.split("\t")
        parsed.append({key: unescape_cell(raw[index] if index < len(raw) else "") for index, key in enumerate(header)})
    return parsed


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows
