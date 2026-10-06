"""Small bounded planner. Adaptive, not a fixed sequence for every question."""

from __future__ import annotations

from retrieval.agentic.budget import AgentBudget
from retrieval.agentic.state import InvestigationState
from retrieval.agentic.tools import (
    EVENT_NEIGHBORHOOD,
    RETRIEVE_SPEC,
    SUPPORTING_CHUNKS,
    VECTOR_SEARCH,
    VECTOR_TOP_K,
    ToolAction,
)


class BoundedPlanner:
    """Selects the next independent retrieval actions from InvestigationState."""

    def next_actions(self, state: InvestigationState, budget: AgentBudget) -> list[ToolAction]:
        if state.stop_reason is not None:
            return []
        if state.parsed.status != "parsed":
            return []
        remaining = budget.remaining_tool_slots(state.tool_calls_used)
        if remaining <= 0:
            return []
        if state.iteration == 0:
            return self._initial(state)[:remaining]
        follow_ups = self._follow_ups(state)
        unique: list[ToolAction] = []
        for action in follow_ups:
            if action.fingerprint() in state.fingerprints:
                continue
            unique.append(action)
            if len(unique) >= remaining:
                break
        return unique

    def _initial(self, state: InvestigationState) -> list[ToolAction]:
        state.mark_searching("target_event", "answer_field", "complete_count", "max_attribute")
        return [
            ToolAction(
                tool=RETRIEVE_SPEC,
                arguments={"operation": state.spec.operation},
                reason=self._initial_reason(state),
                parallel_group="primary",
            )
        ]

    def _initial_reason(self, state: InvestigationState) -> str:
        mapping = {
            "lookup": "lookup needs an Event title match, not a text snippet ranking",
            "aggregation": "aggregation requires the complete Event set for COUNT, not top-k chunks",
            "superlative": "superlative needs competitor attributes over the sport/games set",
            "temporal": "temporal question needs previous/next Event gold, not the named year page",
            "multi_hop": "venue/date question needs typed Event hops rather than lexical lookup",
        }
        return mapping.get(state.spec.qtype, "parsed spec maps to one structured graph operation")

    def _follow_ups(self, state: InvestigationState) -> list[ToolAction]:
        if state.primary_status is None:
            return []
        terminal = state.terminal_from_primary()
        if terminal in {"not_found", "unresolved"}:
            return []
        actions: list[ToolAction] = []
        if (
            state.require_multihop_neighborhood
            and state.slot("graph_relation") is not None
            and state.slot("graph_relation").status != "resolved"
            and state.primary_status == "supported"
            and len(state.event_ids) == 1
            and not state.has_neighborhood
        ):
            actions.append(
                ToolAction(
                    tool=EVENT_NEIGHBORHOOD,
                    arguments={"event_id": state.event_ids[0]},
                    reason="primary venue/date hit left a relational gap: typed HELD_AT/IN_GAMES hops are still missing",
                    parallel_group="repair",
                )
            )
        if (
            state.primary_status == "supported"
            and state.event_ids
            and not state.has_chunks
            and not state.chunks_attempted
        ):
            actions.append(
                ToolAction(
                    tool=SUPPORTING_CHUNKS,
                    arguments={"event_ids": list(state.event_ids), "max_extra": 0},
                    reason="retrieved Events still lack supporting provenance chunks",
                    parallel_group="repair",
                )
            )
        if (
            state.primary_status == "ambiguous"
            and state.event_ids
            and not state.has_chunks
            and not state.chunks_attempted
        ):
            actions.append(
                ToolAction(
                    tool=SUPPORTING_CHUNKS,
                    arguments={"event_ids": list(state.event_ids), "max_extra": 0},
                    reason="multiple candidates remain; fetch supporting chunks for all and keep the collision",
                    parallel_group="repair",
                )
            )
        if (
            state.primary_status in {"supported", "ambiguous"}
            and state.event_ids
            and not state.has_chunks
            and state.chunks_attempted
            and state.vector_search_available
            and not state.vector_search_used
        ):
            actions.append(
                ToolAction(
                    tool=VECTOR_SEARCH,
                    arguments={"query": state.question, "top_k": VECTOR_TOP_K},
                    reason="graph retrieval left a provenance gap after supporting chunks were unavailable; try one bounded vector fallback",
                    parallel_group="repair",
                )
            )
        return actions
