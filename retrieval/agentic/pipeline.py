"""Bounded Agentic GraphRAG pipeline. Adaptive retrieval with hard caps."""

from __future__ import annotations

import time
from dataclasses import dataclass

from answering.generator import Generator, public_generation_status
from answering.models import CitedAnswer, GeneratorRequest, PipelineTimings
from answering.packer import ContextPacker, validate_generator_citations
from retrieval.agentic.budget import AgentBudget
from retrieval.agentic.executor import ToolExecutor
from retrieval.agentic.planner import BoundedPlanner
from retrieval.agentic.state import InvestigationState, StopReason
from retrieval.agentic.synthesize import merge_retrievals
from retrieval.agentic.tools import GraphRetrieverTools
from retrieval.agentic.trace import AgentTrace, PlanStep, SlotUpdate
from retrieval.graphrag.parser import GraphRAGQuestionParser, ParsedRetrievalRequest
from retrieval.graphrag.retriever import GraphRetriever


@dataclass
class AgentResult:
    answer: CitedAnswer
    trace: AgentTrace
    state: InvestigationState

    @property
    def status(self) -> str:
        return str(self.answer.status)


class AgenticGraphRAGPipeline:
    def __init__(
        self,
        parser: GraphRAGQuestionParser,
        retriever: GraphRetriever,
        packer: ContextPacker,
        generator: Generator,
        budget: AgentBudget | None = None,
        planner: BoundedPlanner | None = None,
        require_multihop_neighborhood: bool = True,
        vector_retriever=None,
    ) -> None:
        self.parser = parser
        self.retriever = retriever
        self.packer = packer
        self.generator = generator
        self.budget = budget or AgentBudget()
        self.planner = planner or BoundedPlanner()
        self.require_multihop_neighborhood = require_multihop_neighborhood
        self.vector_retriever = vector_retriever
        self.executor = ToolExecutor(
            GraphRetrieverTools(retriever, vector_retriever=vector_retriever),
            self.budget,
        )

    def run(self, question: str, qtype: str | None = None) -> AgentResult:
        total_started = time.perf_counter()
        parse_started = time.perf_counter()
        parsed = self.parser.parse(question, qtype=qtype)
        parsing_ms = (time.perf_counter() - parse_started) * 1000.0

        state = InvestigationState.from_parsed(
            parsed,
            vector_search_available=self.vector_retriever is not None,
        )
        state.require_multihop_neighborhood = self.require_multihop_neighborhood
        trace = AgentTrace(
            question=question,
            interpreted={
                "status": parsed.status,
                "reason": parsed.reason,
                "spec": parsed.spec.to_dict(),
                "qtype": parsed.spec.qtype,
            },
        )

        retrieval_started = time.perf_counter()
        if parsed.status != "parsed":
            state.stop_reason = "unsupported" if parsed.status == "unsupported" else "unresolved"
        else:
            self._loop(state, trace)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

        packing_started = time.perf_counter()
        merged = merge_retrievals(
            state.retrievals,
            operation=parsed.spec.operation,
            status=self._result_status(state),
            reason=state.primary_reason or parsed.reason,
        )
        context = self.packer.pack_graph(merged)
        packing_ms = (time.perf_counter() - packing_started) * 1000.0

        generation_started = time.perf_counter()
        answer = self._generate(question, parsed, merged, context, state)
        generation_ms = (time.perf_counter() - generation_started) * 1000.0

        answer.timings = PipelineTimings(
            parsing_ms=parsing_ms,
            retrieval_ms=retrieval_ms,
            packing_ms=packing_ms,
            generation_ms=generation_ms,
            total_ms=(time.perf_counter() - total_started) * 1000.0,
        )
        answer.notes["agentic"] = True
        answer.notes["graph_retriever_calls"] = state.tool_calls_used
        answer.notes["followup_retrievals"] = state.follow_ups_used
        answer.notes["stop_reason"] = state.stop_reason
        answer.notes["complete_set_preserved"] = not merged.is_complete_set or (
            len(merged.event_ids) == len(context.notes.get("event_ids") or [])
        )
        trace.stop_reason = state.stop_reason
        trace.total_steps = state.iteration
        trace.total_tool_calls = state.tool_calls_used
        trace.retrieval_methods = sorted(
            {item.retrieval_method for item in merged.all_evidence() if item.retrieval_method}
        )
        trace.timings = answer.timings.to_dict()
        trace.observation_chars = sum(len(item.text or "") + len(item.value or "") for item in context.items)
        trace.parallel_groups = sorted(
            {item.parallel_group for item in trace.tool_calls if item.parallel_group}
        )
        return AgentResult(answer=answer, trace=trace, state=state)

    def _loop(self, state: InvestigationState, trace: AgentTrace) -> None:
        while True:
            if not self.budget.allow_iteration(state.iteration):
                state.stop_reason = "budget_exhausted"
                trace.follow_up_decisions.append("stop:budget_exhausted")
                return
            if self.budget.remaining_tool_slots(state.tool_calls_used) <= 0:
                state.stop_reason = "budget_exhausted"
                trace.follow_up_decisions.append("stop:budget_exhausted")
                return
            actions = self.planner.next_actions(state, self.budget)
            kind = "primary" if state.iteration == 0 else "follow_up"
            if state.iteration > 0 and not self.budget.allow_follow_up(state.follow_ups_used):
                actions = []
            actions = [action for action in actions if action.fingerprint() not in state.fingerprints]
            if not actions:
                state.stop_reason = self._stop_without_actions(state)
                trace.follow_up_decisions.append(f"stop:{state.stop_reason}")
                return
            trace.plan_steps.append(
                PlanStep(
                    iteration=state.iteration,
                    kind=kind,
                    reason=actions[0].reason,
                    actions=list(actions),
                )
            )
            if kind == "follow_up":
                trace.follow_up_decisions.append(actions[0].reason)
                trace.strategy_changes.append(
                    f"iteration {state.iteration}: {actions[0].tool} because {actions[0].reason}"
                )
            observations = self.executor.execute(actions, state.spec)
            progress = False
            for observation in observations:
                trace.tool_calls.append(observation)
                if not observation.success:
                    state.last_error = observation.error
                    state.stop_reason = "tool_failure"
                    return
                if not state.record_fingerprint(observation.fingerprint):
                    state.stop_reason = "no_progress"
                    trace.follow_up_decisions.append("stop:no_progress")
                    return
                if observation.result is not None:
                    state.apply_retrieval(observation.result, observation.tool)
                    progress = True
                    self._trace_slots(state, trace)
                state.tool_calls_used += 1
            if state.iteration > 0:
                state.follow_ups_used += 1
            state.iteration += 1
            if not progress:
                state.stop_reason = "no_progress"
                return
            if state.answer_ready():
                state.stop_reason = "answered"
                trace.follow_up_decisions.append("stop:answered")
                return
            terminal = state.terminal_from_primary()
            if terminal in {"not_found", "unresolved"}:
                state.stop_reason = terminal
                trace.follow_up_decisions.append(f"stop:{terminal}")
                return

    def _stop_without_actions(self, state: InvestigationState) -> StopReason:
        if state.answer_ready():
            return "answered"
        terminal = state.terminal_from_primary()
        if terminal is not None:
            return terminal
        if state.tool_calls_used == 0:
            return "unresolved"
        return "no_progress"

    def _result_status(self, state: InvestigationState) -> str:
        if state.stop_reason == "answered":
            return "supported"
        if state.primary_status:
            return state.primary_status
        if state.stop_reason in {"unsupported", "unresolved", "not_found", "ambiguous"}:
            mapping = {
                "unsupported": "unresolved",
                "unresolved": "unresolved",
                "not_found": "not_found",
                "ambiguous": "ambiguous",
            }
            return mapping[state.stop_reason]
        return "unresolved"

    def _trace_slots(self, state: InvestigationState, trace: AgentTrace) -> None:
        for slot in state.slots.values():
            trace.slot_updates.append(
                SlotUpdate(
                    iteration=state.iteration,
                    slot_id=slot.slot_id,
                    status=slot.status,
                    evidence_ids=list(slot.evidence_ids),
                    note=slot.spec,
                )
            )

    def _generate(
        self,
        question: str,
        parsed: ParsedRetrievalRequest,
        merged,
        context,
        state: InvestigationState,
    ) -> CitedAnswer:
        warnings = list(parsed.warnings)
        notes = {"fixed_policy": False, "agentic_policy": True}
        try:
            generated = self.generator.generate(
                GeneratorRequest(question=question, parsed_request=parsed, context=context)
            )
        except Exception as exc:
            return CitedAnswer(
                question=question,
                answer_text="",
                citations=[],
                retrieval_result=merged,
                evidence_used=[],
                status="generation_error",
                parsed_request=parsed,
                warnings=["generator_failed"],
                notes={**notes, "generator_error": exc.__class__.__name__},
            )
        warnings.extend(generated.warnings)
        notes.update(generated.notes)
        if generated.status == "generation_error":
            return CitedAnswer(
                question=question,
                answer_text="",
                citations=[],
                retrieval_result=merged,
                evidence_used=[],
                status="generation_error",
                parsed_request=parsed,
                warnings=warnings,
                notes=notes,
            )
        if context.answer_suppressed or merged.status != "supported":
            warnings.append("answer_suppressed")
            return CitedAnswer(
                question=question,
                answer_text="",
                citations=[],
                retrieval_result=merged,
                evidence_used=[],
                status=self._suppressed_status(parsed, merged.status, state.stop_reason),
                parsed_request=parsed,
                warnings=warnings,
                notes=notes,
            )
        validation = validate_generator_citations(context, generated)
        if not validation.valid:
            warnings.extend(validation.errors)
            notes["citation_validation"] = "failed"
            return CitedAnswer(
                question=question,
                answer_text="",
                citations=[],
                retrieval_result=merged,
                evidence_used=[],
                status="invalid_citations",
                parsed_request=parsed,
                warnings=warnings,
                notes=notes,
            )
        status = public_generation_status(generated)
        notes["citation_validation"] = "passed"
        notes.update(generated.notes)
        return CitedAnswer(
            question=question,
            answer_text=generated.answer_text,
            citations=list(validation.citations),
            retrieval_result=merged,
            evidence_used=list(validation.evidence_used),
            status=status,
            parsed_request=parsed,
            warnings=warnings,
            notes=notes,
        )

    @staticmethod
    def _suppressed_status(
        parsed: ParsedRetrievalRequest,
        retrieval_status: str,
        stop_reason: StopReason | None,
    ) -> str:
        if parsed.status == "unsupported":
            return "unsupported"
        if parsed.status == "unresolved":
            return "unresolved"
        if retrieval_status in {"ambiguous", "not_found", "unresolved"}:
            return retrieval_status
        if stop_reason in {"ambiguous", "not_found", "unresolved", "unsupported"}:
            return stop_reason
        return "abstained"
