# Architecture (as implemented)

This is the production system in this public snapshot. It does not implement
the earlier planning document that described Entity/Alias/Community vertices
or vector-seeded GraphRAG. Those layers are **not** the production path.

## Shared pipeline

```
USER QUESTION
  → retrieval policy
        RAG:        BM25 (no QuestionParser)
        GraphRAG:   QuestionParser → one typed GSQL call
        Agentic:    QuestionParser → BoundedPlanner
  → ContextPacker
  → shared SemanticGenerator
  → fail-closed citation / grounding validation
  → CitedAnswer
```

The three policies share:

- corpus (parsed Event table + chunks)
- `QuestionParser` / `QuerySpec` (**graph policies only**; RAG does not parse)
- `ContextPacker`
- `SemanticGenerator` (or deterministic generator)
- evaluator and token accounting
- fail-closed citation validator

They differ only in retrieval / control.

Gold answers never enter the pipelines. They are applied only in
`evaluation/` after a `HarnessResult` exists.

## Parser

`retrieval/structured/question.py` is the only intent/slot extractor.

Operations (unchanged since the graph contract):

| Operation | Typical GSQL query |
|---|---|
| `lookup_nations` | `lookup_event` |
| `count_over_threshold` | `count_over_threshold` |
| `argmax_competitors` | `argmax_competitors` |
| `previous_event_gold` / `next_event_gold` | `previous_event_gold` |
| `events_at_venue_date` | `events_at_venue_date` |

Normalization covers whitespace, case, framing prefixes, and closed
synonym/shape classes. Overlapping family matches fail closed. There is no LLM
parser, no qid-specific rules, and no embeddings on this path.

`GraphRAGQuestionParser` only adapts `QuestionParser` and `validate_spec`.

## RAG (production)

```
question → TextRetriever (BM25 / sparse) → pack_text → SemanticGenerator → citations
```

- No TigerGraph, no `QuestionParser` in the retrieval path.
- `DenseRetriever` is a stub that returns no hits. Production RAG is BM25.
- Top-k packing has no complete-set semantics. Aggregation counts that are not
  extractive spans typically abstain.

## Fixed GraphRAG

```
question → parser → GraphRetriever.retrieve(spec, include_chunks=True)
        → pack_graph → SemanticGenerator → citations
```

Exactly one structured retrieval. No planner, no `event_neighborhood`, no
follow-up. Ambiguous / not_found / unresolved / unsupported retrieval suppresses
the answer after generation.

## Bounded Agentic GraphRAG

```
question → parser → BoundedPlanner
        → retrieve_spec (include_chunks=False)
        → optional event_neighborhood / supporting_chunks
        → merge → pack_graph → SemanticGenerator → citations
```

Orchestration is deterministic: slot ledger + unused-tool classes + `AgentBudget`
(3 / 6 / 2). Repeated `(tool, arguments)` fingerprints are rejected. The graph
is never mutated. This is not an LLM planner and not a multi-agent swarm.

On the measured public set, extra hops did not improve accuracy versus Fixed
GraphRAG (97% vs 98%).

Tools: `retrieve_spec`, `supporting_chunks`, `event_neighborhood`.

## TigerGraph role

Installed queries in `gsql/00_schema.gsql` … `04_retrieval.gsql` implement the
five operations plus chunk/neighborhood fetch. The graph stores Event vertices
with typed attributes (sport, year, season, venue, date, competitors, nations,
gold, prev/next). That is what makes complete-set aggregation, venue/date
collision detection, and temporal prev-year lookup possible. Chunks are
supporting evidence, not the aggregation engine.

## Why vector retrieval exists, but is not production RAG

`gsql/05_vector.gsql` and the Python vector helpers are an **experiment-only**
capability. They were verified live in the development environment. They are
**not** installed by the standard production ingest path
(`python -m scripts.ingest_tigergraph`), which applies `gsql/00_schema.gsql`
through `04_retrieval.gsql` only.

Phase 13 measured three RAG-only variants; none of these is production routing:

- V0 = production BM25
- V1 = TigerGraph vector search over Chunk embeddings
- V2 = BM25 + vector hybrid

Vector search can rank Chunk embeddings on `OlympicGraph` when the experiment
schema/query is installed separately. It is **not** the production RAG
retriever. GraphRAG and Agentic still use fixed typed GSQL. Hybrid/vector did
not replace BM25 as the default `TextRetriever` method.

GRIP / MCP is **not** part of this public production runtime. The three
pipelines talk to TigerGraph through pyTigerGraph / installed GSQL only.

## Evidence and grounding

`ContextPacker` preserves provenance fields. Graph complete-set results are not
top-k truncated. `SemanticGenerator` must cite packed `evidence_id`s with quotes
that are exact spans of packed text. Unsupported, orphan, or uncited answers are
cleared. Ambiguity and absence fail closed rather than guessing.

## Evaluation surfaces

| Surface | Path | Notes |
|---|---|---|
| Canonical public 100 | `scripts/eval_three_way.py` | Official public questions; gold used only in scoring |
| Phase 13 vector ablation | `scripts/eval_vector_ablation.py` | Experiment-only; does not change production routing |
| Parser robustness | `tests/test_parser_robustness.py` | Closed-class paraphrases of public templates; no live graph |

## Submission-safe trace export

`evaluation.trace_export.export_submission_record` serializes **system-observable**
behavior for later packaging (public demos or a private submission bundle).

It records the answer, tokens, latency, citations, retrieval methods, and — for
Agentic — the full plan/tool/observation trace. Public three-way **scoring**
still uses compact traces and gold only inside `evaluation.benchmark`.

The exporter intentionally excludes expected answers, gold document IDs,
correctness, and other evaluator-only labels. A later hidden-50 runner should
call this generic exporter; it must not copy scored benchmark rows as-is.

## Explicitly not production

- Entity / Alias / Community vertices
- Vector index / kNN as GraphRAG or as default RAG
- Dense RAG (stub only in production `TextRetriever`)
- LLM planner / swarm / MCP / GRIP runtime
- Graph mutation by the agent
- Question-id special cases
- Changing gold to raise scores
