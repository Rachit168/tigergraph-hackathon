"""Synchronous fixed GraphRAG pipeline. No planning or retrieval loop."""

from __future__ import annotations

import time

from answering.generator import Generator, public_generation_status
from answering.models import (
    CitedAnswer,
    GeneratorRequest,
    GeneratorResult,
    PipelineTimings,
)
from answering.packer import ContextPacker, validate_generator_citations
from retrieval.graphrag.parser import GraphRAGQuestionParser, ParsedRetrievalRequest
from retrieval.graphrag.retriever import GraphRetriever


class FixedGraphRAGPipeline:
    def __init__(
        self,
        parser: GraphRAGQuestionParser,
        retriever: GraphRetriever,
        packer: ContextPacker,
        generator: Generator,
    ) -> None:
        self.parser = parser
        self.retriever = retriever
        self.packer = packer
        self.generator = generator

    def run(self, question: str, qtype: str | None = None) -> CitedAnswer:
        total_started = time.perf_counter()

        parse_started = time.perf_counter()
        parsed = self.parser.parse(question, qtype=qtype)
        parsing_ms = (time.perf_counter() - parse_started) * 1000.0

        retrieval_started = time.perf_counter()
        # The fixed policy always makes exactly one GraphRetriever call.
        retrieval = self.retriever.retrieve(
            parsed.spec,
            include_chunks=True,
            max_extra_chunks=0,
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

        packing_started = time.perf_counter()
        context = self.packer.pack_graph(retrieval)
        packing_ms = (time.perf_counter() - packing_started) * 1000.0

        generation_started = time.perf_counter()
        generation_error: Exception | None = None
        try:
            generated = self.generator.generate(
                GeneratorRequest(
                    question=question,
                    parsed_request=parsed,
                    context=context,
                )
            )
        except Exception as exc:
            generation_error = exc
            generated = GeneratorResult(
                answer_text="",
                status="generation_error",
                warnings=("generator_failed",),
            )
        generation_ms = (time.perf_counter() - generation_started) * 1000.0

        timings = PipelineTimings(
            parsing_ms=parsing_ms,
            retrieval_ms=retrieval_ms,
            packing_ms=packing_ms,
            generation_ms=generation_ms,
            total_ms=(time.perf_counter() - total_started) * 1000.0,
        )
        warnings = [*parsed.warnings, *generated.warnings]
        packed_entity_ids = [
            item.event_id
            for item in context.items
            if item.evidence_type == "entity" and item.event_id is not None
        ]
        notes = {
            "fixed_policy": True,
            "graph_retriever_calls": 1,
            "followup_retrievals": 0,
            "context_evidence_count": len(context.items),
            "complete_set_preserved": not retrieval.is_complete_set
            or (
                len(packed_entity_ids) == len(retrieval.event_ids)
                and set(packed_entity_ids) == set(retrieval.event_ids)
            ),
        }

        if generation_error is not None:
            notes["generator_error"] = generation_error.__class__.__name__
            return CitedAnswer(
                question=question,
                answer_text="",
                citations=[],
                retrieval_result=retrieval,
                evidence_used=[],
                status="generation_error",
                parsed_request=parsed,
                warnings=warnings,
                notes=notes,
                timings=timings,
            )

        if generated.status == "generation_error":
            notes.update(generated.notes)
            return CitedAnswer(
                question=question,
                answer_text="",
                citations=[],
                retrieval_result=retrieval,
                evidence_used=[],
                status="generation_error",
                parsed_request=parsed,
                warnings=warnings,
                notes=notes,
                timings=timings,
            )

        if context.answer_suppressed or retrieval.status != "supported":
            warnings.append("answer_suppressed")
            return CitedAnswer(
                question=question,
                answer_text="",
                citations=[],
                retrieval_result=retrieval,
                evidence_used=[],
                status=self._suppressed_status(parsed, retrieval.status),
                parsed_request=parsed,
                warnings=warnings,
                notes=notes,
                timings=timings,
            )

        validation = validate_generator_citations(context, generated)
        if not validation.valid:
            warnings.extend(validation.errors)
            notes["citation_validation"] = "failed"
            return CitedAnswer(
                question=question,
                answer_text="",
                citations=[],
                retrieval_result=retrieval,
                evidence_used=[],
                status="invalid_citations",
                parsed_request=parsed,
                warnings=warnings,
                notes=notes,
                timings=timings,
            )

        status = public_generation_status(generated)
        notes["citation_validation"] = "passed"
        notes.update(generated.notes)
        return CitedAnswer(
            question=question,
            answer_text=generated.answer_text,
            citations=list(validation.citations),
            retrieval_result=retrieval,
            evidence_used=list(validation.evidence_used),
            status=status,
            parsed_request=parsed,
            warnings=warnings,
            notes=notes,
            timings=timings,
        )

    @staticmethod
    def _suppressed_status(parsed: ParsedRetrievalRequest, retrieval_status: str) -> str:
        if parsed.status == "unsupported":
            return "unsupported"
        if parsed.status == "unresolved":
            return "unresolved"
        if retrieval_status in {"ambiguous", "not_found", "unresolved"}:
            return retrieval_status
        return "abstained"
