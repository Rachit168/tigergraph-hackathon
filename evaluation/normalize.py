"""Deterministic gold-answer comparison. Does not hardcode expected outputs."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence


def as_answer_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return [str(value)]


def normalize_token(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def coerce_number(value: str) -> int | float | None:
    stripped = value.strip()
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)
    if re.fullmatch(r"-?\d+\.\d+", stripped):
        return float(stripped)
    return None


def normalize_answer(value: object) -> tuple:
    items = [normalize_token(item) for item in as_answer_list(value)]
    numbers = [coerce_number(item) for item in items]
    if items and all(number is not None for number in numbers):
        return ("num", tuple(numbers))
    return ("str", tuple(items))


def answers_match_exact(predicted: object, gold: object) -> bool:
    return as_answer_list(predicted) == as_answer_list(gold)


def answers_match_normalized(predicted: object, gold: object) -> bool:
    if predicted is None:
        return False
    return normalize_answer(predicted) == normalize_answer(gold)


def truncate(value: object, limit: int = 80) -> str:
    text = " | ".join(as_answer_list(value))
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
