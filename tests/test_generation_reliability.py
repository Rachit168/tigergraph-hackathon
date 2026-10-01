"""Shared generation reliability: retries, timeouts, truncation, token accounting."""

from __future__ import annotations

import json
import unittest

from answering.factory import build_generator
from answering.models import GeneratorRequest
from answering.packer import ContextPacker
from answering.provider import (
    CompletionResult,
    FakeCompletionProvider,
    ProviderError,
    is_transient_provider_error,
)
from answering.semantic import SemanticGenerator
from config.llm import DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT_S
from evaluation.harness import AgenticGraphRAGAdapter, GraphRAGAdapter, RAGAdapter
from retrieval.agentic.budget import AgentBudget
from retrieval.agentic.pipeline import AgenticGraphRAGPipeline
from retrieval.graphrag.models import GraphEvidence, GraphRef, GraphRetrievalResult
from retrieval.graphrag.parser import GraphRAGQuestionParser
from retrieval.graphrag.pipeline import FixedGraphRAGPipeline
from retrieval.rag.models import RetrievalHit, RetrievalResult
from retrieval.structured.models import QuerySpec


def _json_payload(status: str, answer: str, citation_ids: list[str]) -> str:
    return json.dumps(
        {
            "status": status,
            "answer": answer,
            "citation_ids": citation_ids,
            "reason": "unit-test",
            "support": [{"evidence_id": evidence_id, "quote": answer} for evidence_id in citation_ids],
        }
    )


def _hit() -> RetrievalHit:
    return RetrievalHit(
        chunk_id="Q26217865::c000",
        document_id="Q26217865",
        document_title="Judo at the 2016 Summer Olympics – Women's 57 kg",
        score=4.2,
        rank=1,
        text="nations: 23",
        section="infobox",
        kind="infobox",
        event_id="Q26217865",
        retrieval_method="bm25",
        source_url="https://example.test/judo",
    )


def _text_result() -> RetrievalResult:
    return RetrievalResult(query="How many nations?", method="sparse", hits=[_hit()], params={"top_k": 10})


def _spec() -> QuerySpec:
    return QuerySpec(
        qtype="lookup",
        operation="lookup_nations",
        raw_question="How many nations?",
        matched_template=True,
        event_title="Judo at the 2016 Summer Olympics – Women's 57 kg",
        requested_field="nations",
    )


def _fact() -> GraphEvidence:
    return GraphEvidence(
        evidence_id="call-1:fact:nations:Q26217865",
        evidence_type="fact",
        retrieval_method="gsql:lookup_event",
        tool_call_id="call-1",
        graph_refs=[GraphRef(vertex_type="Event", vertex_id="Q26217865", attribute="nations")],
        event_id="Q26217865",
        document_id="Q26217865",
        field_name="nations",
        value="23",
        why_retrieved="query returned nations",
    )


def _graph_result() -> GraphRetrievalResult:
    return GraphRetrievalResult(
        operation="lookup_event",
        query_name="lookup_event",
        status="supported",
        cardinality="one",
        tool_call_id="call-1",
        retrieval_method="gsql:lookup_event",
        event_ids=["Q26217865"],
        entities=[],
        facts=[_fact()],
        chunks=[],
        notes={},
    )


class SpyParser:
    def parse(self, question: str, qtype: str | None = None) -> QuerySpec:
        return _spec()


class SpyGraphRetriever:
    def __init__(self, result: GraphRetrievalResult) -> None:
        self.result = result
        self.calls: list[tuple] = []

    def retrieve(self, spec, *, include_chunks: bool = False, max_extra_chunks: int = 0):
        self.calls.append(("retrieve", spec.operation, include_chunks, max_extra_chunks))
        return self.result

    def supporting_chunks(self, event_ids, *, max_extra: int = 0, tool_call_id: str | None = None):
        self.calls.append(("supporting_chunks", list(event_ids), max_extra))
        return self.result

    def event_neighborhood(self, event_id: str):
        raise AssertionError(f"unexpected neighborhood {event_id}")


class FakeTextRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls: list[tuple] = []

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        self.calls.append((query, kwargs))
        return self.result


def _request(context=None) -> GeneratorRequest:
    packed = context if context is not None else ContextPacker().pack_text(_text_result())
    return GeneratorRequest(question="How many nations?", parsed_request=None, context=packed)


def _generator(provider: FakeCompletionProvider) -> SemanticGenerator:
    return SemanticGenerator(provider, retry_backoff_s=(), max_attempts=3)


class GenerationReliabilityTests(unittest.TestCase):
    def test_shared_max_tokens_freeze_is_1024(self) -> None:
        self.assertEqual(DEFAULT_MAX_TOKENS, 1024)
        self.assertEqual(DEFAULT_TIMEOUT_S, 30.0)

    def test_transient_http_429_retries_then_succeeds(self) -> None:
        context = ContextPacker().pack_text(_text_result())
        evidence_id = context.items[0].evidence_id
        success = CompletionResult(
            text=_json_payload("answered", "23", [evidence_id]),
            model="fake",
            prompt_tokens=4,
            completion_tokens=6,
            total_tokens=10,
            finish_reason="stop",
        )
        provider = FakeCompletionProvider(
            script=(ProviderError("http_429", "rate limited"), success),
        )
        generated = _generator(provider).generate(_request(context))
        self.assertEqual(generated.status, "answered")
        self.assertEqual(generated.answer_text, "23")
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(generated.notes["model_calls"], 2)
        self.assertEqual(generated.notes["tokens"], 10)
        self.assertTrue(generated.notes["tokens_unknown"])
        self.assertEqual(generated.notes["model_accounting"], "provider_completion_retry")

    def test_permanent_http_401_does_not_retry(self) -> None:
        provider = FakeCompletionProvider(error=ProviderError("http_401", "denied"))
        generated = _generator(provider).generate(_request())
        self.assertEqual(generated.status, "generation_error")
        self.assertEqual(generated.answer_text, "")
        self.assertIn("provider_failed", generated.warnings)
        self.assertEqual(generated.notes["provider_error"], "http_401")
        self.assertEqual(generated.notes["failure_class"], "provider_error")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(generated.notes["model_calls"], 1)
        self.assertTrue(generated.notes["tokens_unknown"])
        self.assertEqual(generated.notes["tokens"], 0)
        self.assertFalse(is_transient_provider_error("http_401"))

    def test_timeout_retries_then_fails_closed(self) -> None:
        provider = FakeCompletionProvider(error=ProviderError("timeout", "llm_provider_timeout"))
        generated = _generator(provider).generate(_request())
        self.assertEqual(generated.status, "generation_error")
        self.assertEqual(generated.answer_text, "")
        self.assertIn("provider_failed", generated.warnings)
        self.assertEqual(generated.notes["failure_class"], "timeout")
        self.assertEqual(generated.notes["provider_error"], "timeout")
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(generated.notes["model_calls"], 3)
        self.assertTrue(generated.notes["tokens_unknown"])

    def test_malformed_json_abstains_without_retry(self) -> None:
        provider = FakeCompletionProvider("not json {", total_tokens=7)
        generated = _generator(provider).generate(_request())
        self.assertEqual(generated.status, "abstained")
        self.assertEqual(generated.answer_text, "")
        self.assertIn("unparseable_provider_output", generated.warnings)
        self.assertEqual(generated.notes["failure_class"], "malformed_output")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(generated.notes["model_calls"], 1)
        self.assertEqual(generated.notes["tokens"], 7)
        self.assertFalse(generated.notes["tokens_unknown"])

    def test_truncated_json_is_generation_error_without_retry(self) -> None:
        provider = FakeCompletionProvider(
            '{"status": "answered"',
            finish_reason="length",
            prompt_tokens=20,
            completion_tokens=512,
            total_tokens=532,
        )
        generated = _generator(provider).generate(_request())
        self.assertEqual(generated.status, "generation_error")
        self.assertEqual(generated.answer_text, "")
        self.assertIn("truncated_completion", generated.warnings)
        self.assertEqual(generated.notes["failure_class"], "truncated")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(generated.notes["model_calls"], 1)
        self.assertEqual(generated.notes["tokens"], 532)
        self.assertFalse(generated.notes["tokens_unknown"])
        self.assertFalse(is_transient_provider_error("truncated"))

    def test_empty_truncated_provider_error_keeps_reported_usage(self) -> None:
        error = ProviderError(
            "truncated",
            "llm_provider_truncated",
            prompt_tokens=40,
            completion_tokens=512,
            total_tokens=552,
        )
        provider = FakeCompletionProvider(error=error)
        generated = _generator(provider).generate(_request())
        self.assertEqual(generated.status, "generation_error")
        self.assertIn("truncated_completion", generated.warnings)
        self.assertEqual(generated.notes["failure_class"], "truncated")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(generated.notes["tokens"], 552)
        self.assertFalse(generated.notes["tokens_unknown"])

    def test_retry_token_accounting_sums_known_attempts(self) -> None:
        context = ContextPacker().pack_text(_text_result())
        evidence_id = context.items[0].evidence_id
        first = ProviderError("http_502", "bad gateway", prompt_tokens=3, completion_tokens=0, total_tokens=3)
        second = CompletionResult(
            text=_json_payload("answered", "23", [evidence_id]),
            model="fake",
            prompt_tokens=3,
            completion_tokens=8,
            total_tokens=11,
            finish_reason="stop",
        )
        provider = FakeCompletionProvider(script=(first, second))
        generated = _generator(provider).generate(_request(context))
        self.assertEqual(generated.status, "answered")
        self.assertEqual(generated.notes["tokens"], 14)
        self.assertFalse(generated.notes["tokens_unknown"])
        self.assertEqual(generated.notes["model_calls"], 2)
        self.assertEqual(generated.notes["attempts"][0]["tokens"], 3)
        self.assertEqual(generated.notes["attempts"][1]["tokens"], 11)

    def test_unknown_first_attempt_usage_is_not_invented_as_zero_success(self) -> None:
        context = ContextPacker().pack_text(_text_result())
        evidence_id = context.items[0].evidence_id
        success = CompletionResult(
            text=_json_payload("answered", "23", [evidence_id]),
            model="fake",
            prompt_tokens=2,
            completion_tokens=4,
            total_tokens=6,
            finish_reason="stop",
        )
        provider = FakeCompletionProvider(script=(ProviderError("http_503", "unavailable"), success))
        generated = _generator(provider).generate(_request(context))
        self.assertEqual(generated.status, "answered")
        self.assertEqual(generated.notes["tokens"], 6)
        self.assertTrue(generated.notes["tokens_unknown"])
        self.assertIsNone(generated.notes["attempts"][0]["tokens"])
        self.assertFalse(generated.notes["attempts"][0]["usage_known"])

    def test_unexpected_exception_does_not_retry(self) -> None:
        provider = FakeCompletionProvider(error=RuntimeError("boom"))
        generated = _generator(provider).generate(_request())
        self.assertEqual(generated.status, "generation_error")
        self.assertEqual(generated.notes["failure_class"], "unexpected_exception")
        self.assertEqual(len(provider.calls), 1)

    def test_reliability_policy_is_identical_across_three_pipelines(self) -> None:
        packer = ContextPacker()
        text = _text_result()
        graph = _graph_result()
        rag_id = packer.pack_text(text).items[0].evidence_id
        graph_id = "call-1:fact:nations:Q26217865"
        parser = GraphRAGQuestionParser(SpyParser())
        success_tokens = CompletionResult(text="", model="fake", prompt_tokens=1, completion_tokens=1, total_tokens=2)

        def run(kind: str, script: tuple) -> tuple[str, int, str | None]:
            evidence_id = rag_id if kind == "rag" else graph_id
            payload = list(script)
            if isinstance(payload[-1], CompletionResult):
                payload[-1] = CompletionResult(
                    text=_json_payload("answered", "23", [evidence_id]),
                    model="fake",
                    prompt_tokens=2,
                    completion_tokens=2,
                    total_tokens=4,
                    finish_reason="stop",
                )
            provider = FakeCompletionProvider(script=tuple(payload))
            generator = _generator(provider)
            if kind == "rag":
                result = RAGAdapter(FakeTextRetriever(text), packer, generator).run("How many nations?")
                notes = result.retrieval_metadata.get("generator_notes") or {}
            elif kind == "graphrag":
                pipeline = FixedGraphRAGPipeline(parser, SpyGraphRetriever(graph), packer, generator)
                result = GraphRAGAdapter(pipeline).run("How many nations?", qtype="lookup")
                notes = result.retrieval_metadata.get("pipeline_notes") or {}
            else:
                agent = AgenticGraphRAGPipeline(
                    parser,
                    SpyGraphRetriever(graph),
                    packer,
                    generator,
                    budget=AgentBudget(),
                )
                result = AgenticGraphRAGAdapter(agent).run("How many nations?", qtype="lookup")
                notes = result.retrieval_metadata.get("pipeline_notes") or {}
            return result.status, len(provider.calls), notes.get("failure_class")

        transient = (ProviderError("http_429", "rate limited"), success_tokens)
        timeout_script = (ProviderError("timeout", "llm_provider_timeout"),)
        answered = [run(kind, transient) for kind in ("rag", "graphrag", "agentic")]
        timed_out = [run(kind, timeout_script) for kind in ("rag", "graphrag", "agentic")]
        self.assertEqual({item[0] for item in answered}, {"answered"})
        self.assertEqual({item[1] for item in answered}, {2})
        self.assertEqual({item[2] for item in answered}, {None})
        self.assertEqual({item[0] for item in timed_out}, {"generation_error"})
        self.assertEqual({item[1] for item in timed_out}, {3})
        self.assertEqual({item[2] for item in timed_out}, {"timeout"})

    def test_factory_semantic_uses_the_same_generator_class(self) -> None:
        provider = FakeCompletionProvider("{}")
        generator = build_generator("semantic", provider=provider)
        self.assertIsInstance(generator, SemanticGenerator)
        self.assertEqual(generator.max_attempts, 3)
        self.assertEqual(generator.retry_backoff_s, (0.2, 0.5))


if __name__ == "__main__":
    unittest.main()
