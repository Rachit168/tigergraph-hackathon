# Fixed GraphRAG pipeline

Fixed GraphRAG adds a synchronous answer pipeline and a common evaluation boundary on
top of the graph retrieval contract. It does not add an agent.

## Fixed flow

```
Question
  -> GraphRAGQuestionParser
  -> GraphRetriever.retrieve (exactly once)
  -> ContextPacker
  -> Generator
  -> citation validation
  -> CitedAnswer
```

The call to `GraphRetriever.retrieve` enables the existing deterministic
supporting-chunk behavior. The pipeline does not inspect retrieval output to
choose a second operation, does not call `event_neighborhood`, and does not
perform a follow-up retrieval.

## Question parsing

`retrieval.structured.question.QuestionParser.parse(...) -> QuerySpec` remains
the only source of question templates, operation selection, and argument
extraction.

`retrieval.graphrag.parser.GraphRAGQuestionParser` is an adapter only:

1. delegate to the existing `QuestionParser`;
2. preserve its exact `QuerySpec`;
3. apply `retrieval.graphrag.validate.validate_spec`;
4. return `ParsedRetrievalRequest`.

The adapter contains no regular expressions, intent maps, query maps, or slot
extraction. Installed-query mapping remains in
`retrieval.graph.params.params_for_spec`.

## Evidence and context

`ContextPacker.pack_graph` orders entity, fact, edge, and chunk evidence and
deduplicates only identical `evidence_id` values. Every graph provenance
field is retained, including null fields. Complete-set retrieval has no
top-k packing limit; every returned Event entity and the complete `event_ids`
list remain present.

Text RAG uses the same packed-context contract. Existing `RetrievalHit` records
receive stable context IDs of the form `rag:{method}:{chunk_id}` without
changing BM25.

## Generation and citations

`Generator` is a provider-neutral synchronous protocol. The live three-way
harness supplies the shared `SemanticGenerator` to this pipeline (or the
deterministic generator when no provider is configured). Fixed GraphRAG packs
typed graph facts and supporting chunks; citation validation remains fail-closed.

A citation identifies a packed `evidence_id` and exposes its source URL, source
date, document, chunk, and Event IDs where available. Citation validation is
fail-closed:

- an answer with no citations is rejected;
- a citation absent from packed context is rejected;
- rejected output has no accepted answer, citations, or evidence-used list.

`CitedAnswer` preserves the original `GraphRetrievalResult`, the exact cited
evidence subset, warnings, notes, and parsing/retrieval/packing/generation/total
timings.

## Ambiguity and absence

An ambiguous, unresolved, unsupported, or missing retrieval result is passed
to the generator but cannot become a definitive answer. The pipeline enforces
graph retrieval `answer_suppressed` semantics after generation, keeps all candidate
entities in the retrieval result, and discards unsupported generated output.

## Three-way evaluation boundary

`evaluation.harness` provides independently callable adapters with one common
result schema:

- `RAGAdapter` uses the existing `TextRetriever`;
- `GraphRAGAdapter` uses `FixedGraphRAGPipeline`;
- `AgenticGraphRAGAdapter` wraps the bounded live Agentic pipeline separately.

`ThreeWayEvaluationHarness` applies the same question and optional `qtype` to
each selected adapter. It does not inspect public gold answers during pipeline
execution.

## Intentionally deferred

This pipeline does not include agentic planning, tool-selection loops,
follow-up retrieval, iterative refinement, MCP, vector retrieval, or
communities. The separate Agentic pipeline owns those behaviors.
