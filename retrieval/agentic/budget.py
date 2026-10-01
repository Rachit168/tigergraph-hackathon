"""Hard execution limits for the bounded Agentic GraphRAG loop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentBudget:
    max_iterations: int = 3
    max_tool_calls: int = 6
    max_follow_ups: int = 2
    max_results_per_call: int = 50
    max_observation_chars: int = 4000
    max_workers: int = 4

    def allow_iteration(self, iteration: int) -> bool:
        return iteration < self.max_iterations

    def allow_follow_up(self, follow_ups_used: int) -> bool:
        return follow_ups_used < self.max_follow_ups

    def remaining_tool_slots(self, tool_calls_used: int) -> int:
        return max(0, self.max_tool_calls - tool_calls_used)
