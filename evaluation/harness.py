"""Comparable callable adapters for RAG, GraphRAG, and Agentic placeholder."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from answering.generator import Generator, public_generation_status
from answering.models import Citation, GeneratorRequest, GeneratorResult, PipelineTimings
from answering.packer import ContextPacker, validate_generator_citations
from retrieval.graphrag.pipeline import FixedGraphRAGPipeline
from retrieval.rag.retriever import TextRetriever


@dataclass
class HarnessResult:
    system_name: str
    question: str
    answer: str
    citations: list[Citation]
    latency_ms: float
    retrieval_metadata: dict[str, Any]
    evidence_metadata: list[dict[str, Any]]
    status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timings: PipelineTimings = field(default_factory=PipelineTimings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_name": self.system_name,
            "question": self.question,
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "latency_ms": self.latency_ms,
            "retrieval_metadata": dict(self.retrieval_metadata),
            "evidence_metadata": list(self.evidence_metadata),
            "status": self.status,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "timings": self.timings.to_dict(),
        }


class EvaluationSystem(Protocol):
    system_name: str

    def run(self, question: str, qtype: str | None = None) -> HarnessResult: ...


class GraphRAGAdapter:
    system_name = "graphrag"

    def __init__(self, pipeline: FixedGraphRAGPipeline) -> None:
        self.pipeline = pipeline

    def run(self, question: str, qtype: str | None = None) -> HarnessResult:
        answer = self.pipeline.run(question, qtype=qtype)
        errors = []
        if answer.status in {"invalid_citations", "generation_error"}:
            errors.append(str(answer.status))
        metadata = answer.retrieval_result.to_dict()
        metadata["pipeline_notes"] = dict(answer.notes)
        return HarnessResult(
            system_name=self.system_name,
            question=question,
            answer=answer.answer_text,
            citations=list(answer.citations),
            latency_ms=answer.timings.total_ms,
            retrieval_metadata=metadata,
            evidence_metadata=[
                evidence.to_dict() for evidence in answer.retrieval_result.all_evidence()
            ],
            status=str(answer.status),
            errors=errors,
            warnings=list(answer.warnings),
            timings=answer.timings,
        )


class RAGAdapter:
    system_name = "rag"

    def __init__(
        self,
        retriever: TextRetriever,
        packer: ContextPacker,
        generator: Generator,
        *,
        method: str = "sparse",
        top_k: int = 10,
    ) -> None:
        self.retriever = retriever
        self.packer = packer
        self.generator = generator
        self.method = method
        self.top_k = top_k

    def run(self, question: str, qtype: str | None = None) -> HarnessResult:
        total_started = time.perf_counter()
        retrieval_started = time.perf_counter()
        retrieval = self.retriever.retrieve(
            question,
            method=self.method,
            top_k=self.top_k,
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

        packing_started = time.perf_counter()
        context = self.packer.pack_text(retrieval)
        packing_ms = (time.perf_counter() - packing_started) * 1000.0

        generation_started = time.perf_counter()
        errors: list[str] = []
        warnings: list[str] = []
        try:
            generated = self.generator.generate(
                GeneratorRequest(
                    question=question,
                    parsed_request=None,
                    context=context,
                    qtype=qtype,
                )
            )
        except Exception as exc:
            generated = GeneratorResult(
                answer_text="",
                status="generation_error",
                warnings=("generator_failed",),
            )
            errors.append(f"generator_error:{exc.__class__.__name__}")
        generation_ms = (time.perf_counter() - generation_started) * 1000.0
        warnings.extend(generated.warnings)

        if generated.status == "generation_error":
            answer = ""
            citations = []
            status = "generation_error"
            if "generation_error" not in errors and not any(item.startswith("generator_error:") for item in errors):
                errors.append("generation_error")
        else:
            validation = validate_generator_citations(context, generated)
            answer = generated.answer_text
            citations = list(validation.citations)
            status = public_generation_status(generated)
            if not validation.valid:
                answer = ""
                citations = []
                status = "invalid_citations"
                errors.extend(validation.errors)
                warnings.extend(validation.errors)

        total_ms = (time.perf_counter() - total_started) * 1000.0
        timings = PipelineTimings(
            retrieval_ms=retrieval_ms,
            packing_ms=packing_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        )
        return HarnessResult(
            system_name=self.system_name,
            question=question,
            answer=answer,
            citations=citations,
            latency_ms=total_ms,
            retrieval_metadata={
                "query": retrieval.query,
                "method": retrieval.method,
                "params": dict(retrieval.params),
                "elapsed_ms": retrieval.elapsed_ms,
                "hit_count": len(retrieval.hits),
                "generator_notes": dict(generated.notes),
            },
            evidence_metadata=[item.to_dict() for item in context.items],
            status=status,
            errors=errors,
            warnings=warnings,
            timings=timings,
        )


class AgenticGraphRAGPlaceholder:
    """Harness-compatible boundary; Phase 6 placeholder kept for compatibility."""

    system_name = "agentic_graphrag"

    def run(self, question: str, qtype: str | None = None) -> HarnessResult:
        del qtype
        return HarnessResult(
            system_name=self.system_name,
            question=question,
            answer="",
            citations=[],
            latency_ms=0.0,
            retrieval_metadata={"implemented": False, "retrieval_calls": 0},
            evidence_metadata=[],
            status="not_implemented",
            errors=["agentic_planner_not_implemented"],
            warnings=["Phase 6 placeholder only; no planning or retrieval loop executed"],
        )


class AgenticGraphRAGAdapter:
    """Phase 7 bounded agent adapter. Independent from the GraphRAG one-pass pipeline."""

    system_name = "agentic_graphrag"

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def run(self, question: str, qtype: str | None = None) -> HarnessResult:
        result = self.pipeline.run(question, qtype=qtype)
        answer = result.answer
        errors = []
        if answer.status in {"invalid_citations", "generation_error"}:
            errors.append(str(answer.status))
        metadata = answer.retrieval_result.to_dict()
        metadata["pipeline_notes"] = dict(answer.notes)
        metadata["trace"] = result.trace.to_dict()
        metadata["stop_reason"] = result.trace.stop_reason
        metadata["total_tool_calls"] = result.trace.total_tool_calls
        return HarnessResult(
            system_name=self.system_name,
            question=question,
            answer=answer.answer_text,
            citations=list(answer.citations),
            latency_ms=answer.timings.total_ms,
            retrieval_metadata=metadata,
            evidence_metadata=[
                evidence.to_dict() for evidence in answer.retrieval_result.all_evidence()
            ],
            status=str(answer.status),
            errors=errors,
            warnings=list(answer.warnings),
            timings=answer.timings,
        )


class ThreeWayEvaluationHarness:
    def __init__(self, systems: Iterable[EvaluationSystem]) -> None:
        self.systems = {system.system_name: system for system in systems}

    def run_question(
        self,
        question: str,
        *,
        qtype: str | None = None,
        system_names: Iterable[str] | None = None,
    ) -> list[HarnessResult]:
        names = list(system_names) if system_names is not None else list(self.systems)
        results: list[HarnessResult] = []
        for name in names:
            system = self.systems.get(name)
            if system is None:
                results.append(
                    HarnessResult(
                        system_name=name,
                        question=question,
                        answer="",
                        citations=[],
                        latency_ms=0.0,
                        retrieval_metadata={},
                        evidence_metadata=[],
                        status="error",
                        errors=[f"unknown_system:{name}"],
                    )
                )
                continue
            try:
                results.append(system.run(question, qtype=qtype))
            except Exception as exc:
                results.append(
                    HarnessResult(
                        system_name=name,
                        question=question,
                        answer="",
                        citations=[],
                        latency_ms=0.0,
                        retrieval_metadata={},
                        evidence_metadata=[],
                        status="error",
                        errors=[f"system_error:{exc.__class__.__name__}"],
                    )
                )
        return results

    def run_records(
        self,
        records: Iterable[dict[str, Any]],
        *,
        system_names: Iterable[str] | None = None,
    ) -> list[HarnessResult]:
        results: list[HarnessResult] = []
        for record in records:
            results.extend(
                self.run_question(
                    str(record.get("question") or ""),
                    qtype=record.get("qtype"),
                    system_names=system_names,
                )
            )
        return results
