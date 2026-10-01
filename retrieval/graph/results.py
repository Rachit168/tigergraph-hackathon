"""Structured result objects for installed GSQL queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphQueryResult:
    operation: str
    status: str
    reason: str | None = None
    event_ids: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    gold: list[str] = field(default_factory=list)
    nations: list[int] = field(default_factory=list)
    has_nations: list[bool] = field(default_factory=list)
    competitors: list[int] = field(default_factory=list)
    count: int | None = None
    max_competitors: int | None = None
    target_year: int | None = None
    year_source: str | None = None
    named_ids: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    source: str = "local_contract"

    def to_compare_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "status": self.status,
            "reason": self.reason,
            "event_ids": sorted(self.event_ids),
            "count": self.count,
            "max_competitors": self.max_competitors,
            "target_year": self.target_year,
            "year_source": self.year_source,
            "named_ids": sorted(self.named_ids),
        }
