# Bounded Agentic GraphRAG

Agentic GraphRAG adds a bounded adaptive orchestrator on top of
`GraphRetriever` and the Fixed GraphRAG answering/citation layer. It does not change
the fixed GraphRAG pipeline.

## Flow

```
Question
  -> GraphRAGQuestionParser (delegates to QuestionParser)
  -> BoundedPlanner
  -> parallel GraphRetriever tools
  -> InvestigationState / slot ledger
  -> at most 2 reactive follow-ups, including at most one vector fallback
  -> stop
  -> ContextPacker + Generator + fail-closed citations
```

## How this differs from GraphRAG

Fixed GraphRAG always performs one `retrieve(spec, include_chunks=True)` call.
Agentic GraphRAG:

- chooses tools from the current slots and evidence
- runs a primary structured query without a fixed extra hop
- follows up only when a gap remains
- uses `event_neighborhood` for unique multi-hop hits because the venue/date
  answer still lacks typed relationship evidence
- may use one bounded TigerGraph `vector_search` follow-up when provenance
  remains missing and the vector capability is configured
- keeps complete aggregation sets and ambiguous venue collisions

Lookup, aggregation, superlative, and temporal questions stop after the
primary structured query when the required slots are already filled.

## Safety

`AgentBudget` caps iterations, tool calls, follow-ups, per-call extras, and
observation size. Repeated `(tool, arguments)` fingerprints are rejected.
The graph is never mutated. Vector fallback is optional and fail-closed; if
embedding configuration or the vector backend is unavailable, the Agentic
pipeline continues without that tool.

Stop reasons: `answered`, `ambiguous`, `unresolved`, `not_found`,
`unsupported`, `budget_exhausted`, `no_progress`, `tool_failure`.

## Intentionally outside this pipeline

Planner/LLM loops, MCP, schema changes, communities, swarm agents, and any
change to `FixedGraphRAGPipeline` or Basic RAG/BM25.
