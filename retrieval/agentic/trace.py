"""Serializable agent trace for hackathon evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retrieval.agentic.tools import ToolAction, ToolObservation


@dataclass
class PlanStep:
    iteration: int
    kind: str
    reason: str
    actions: list[ToolAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "kind": self.kind,
            "reason": self.reason,
            "actions": [action.to_dict() for action in self.actions],
        }


@dataclass
class SlotUpdate:
    iteration: int
    slot_id: str
    status: str
    evidence_ids: list[str]
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "slot_id": self.slot_id,
            "status": self.status,
            "evidence_ids": list(self.evidence_ids),
            "note": self.note,
        }


@dataclass
class AgentTrace:
    question: str
    interpreted: dict[str, Any]
    plan_steps: list[PlanStep] = field(default_factory=list)
    tool_calls: list[ToolObservation] = field(default_factory=list)
    slot_updates: list[SlotUpdate] = field(default_factory=list)
    follow_up_decisions: list[str] = field(default_factory=list)
    strategy_changes: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    total_steps: int = 0
    total_tool_calls: int = 0
    retrieval_methods: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    observation_chars: int = 0
    parallel_groups: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "interpreted": dict(self.interpreted),
            "plan_steps": [step.to_dict() for step in self.plan_steps],
            "tool_calls": [item.to_dict() for item in self.tool_calls],
            "slot_updates": [item.to_dict() for item in self.slot_updates],
            "follow_up_decisions": list(self.follow_up_decisions),
            "strategy_changes": list(self.strategy_changes),
            "stop_reason": self.stop_reason,
            "total_steps": self.total_steps,
            "total_tool_calls": self.total_tool_calls,
            "retrieval_methods": list(self.retrieval_methods),
            "timings": dict(self.timings),
            "observation_chars": self.observation_chars,
            "parallel_groups": list(self.parallel_groups),
        }
