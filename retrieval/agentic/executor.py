"""Bounded parallel execution of independent GraphRetriever tool actions."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

from retrieval.agentic.budget import AgentBudget
from retrieval.agentic.tools import GraphRetrieverTools, ToolAction, ToolObservation
from retrieval.graphrag.models import GraphEvidence, GraphRetrievalResult
from retrieval.structured.models import QuerySpec


class ToolExecutor:
    def __init__(self, tools: GraphRetrieverTools, budget: AgentBudget) -> None:
        self.tools = tools
        self.budget = budget
        self._seq = 0
        self._lock = threading.Lock()

    def execute(self, actions: list[ToolAction], spec: QuerySpec) -> list[ToolObservation]:
        if not actions:
            return []
        if len(actions) == 1:
            return [self._run(actions[0], spec)]
        observations: list[ToolObservation | None] = [None] * len(actions)
        workers = max(1, min(self.budget.max_workers, len(actions)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._run, action, spec): index for index, action in enumerate(actions)}
            for future in as_completed(futures):
                observations[futures[future]] = future.result()
        return [item for item in observations if item is not None]

    def _run(self, action: ToolAction, spec: QuerySpec) -> ToolObservation:
        with self._lock:
            self._seq += 1
            seq = self._seq
        tool_call_id = f"agent:{action.tool}:{seq:03d}"
        started = time.perf_counter()
        started_ms = started * 1000.0
        try:
            result = self.tools.run(action, spec)
            result = self._bound_observation(result)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return ToolObservation(
                tool_call_id=tool_call_id,
                tool=action.tool,
                arguments=dict(action.arguments),
                reason=action.reason,
                success=True,
                started_ms=started_ms,
                elapsed_ms=elapsed_ms,
                fingerprint=action.fingerprint(),
                result=result,
                evidence_ids=[item.evidence_id for item in result.all_evidence()],
                retrieval_method=result.retrieval_method,
                observation_truncated=any(item.observation_truncated for item in result.all_evidence()),
                parallel_group=action.parallel_group,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return ToolObservation(
                tool_call_id=tool_call_id,
                tool=action.tool,
                arguments=dict(action.arguments),
                reason=action.reason,
                success=False,
                started_ms=started_ms,
                elapsed_ms=elapsed_ms,
                fingerprint=action.fingerprint(),
                error=exc.__class__.__name__,
                parallel_group=action.parallel_group,
            )

    def _bound_observation(self, result: GraphRetrievalResult) -> GraphRetrievalResult:
        bounded = deepcopy(result)
        if bounded.is_complete_set:
            bounded.notes["truncated_set"] = False
        limit = self.budget.max_observation_chars
        remaining = limit
        for item in [*bounded.entities, *bounded.facts, *bounded.edges, *bounded.chunks]:
            remaining = self._truncate_item(item, remaining)
        extra_cap = self.budget.max_results_per_call
        if not bounded.is_complete_set and bounded.operation != "count_over_threshold":
            if len(bounded.chunks) > extra_cap:
                bounded.chunks = bounded.chunks[:extra_cap]
                for item in bounded.chunks:
                    item.observation_truncated = True
        return bounded

    @staticmethod
    def _truncate_item(item: GraphEvidence, remaining: int) -> int:
        text = item.text or ""
        if not text:
            return remaining
        if remaining <= 0:
            item.text = ""
            item.observation_truncated = True
            return 0
        if len(text) <= remaining:
            return remaining - len(text)
        item.text = text[:remaining]
        item.observation_truncated = True
        return 0
