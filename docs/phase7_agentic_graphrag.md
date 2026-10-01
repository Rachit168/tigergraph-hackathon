# Phase 7: Bounded Agentic GraphRAG

Phase 7 adds a bounded adaptive orchestrator on top of the Phase 5
`GraphRetriever` and the Phase 6 answering/citation layer. It does not change
the fixed GraphRAG pipeline.

## Flow

```
Question
  -> GraphRAGQuestionParser (delegates to QuestionParser)
  -> BoundedPlanner
  -> parallel GraphRetriever tools
  -> InvestigationState / slot ledger
  -> at most 2 reactive follow-ups
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
- keeps complete aggregation sets and ambiguous venue collisions

Lookup, aggregation, superlative, and temporal questions stop after the
primary structured query when the required slots are already filled.

## Safety

`AgentBudget` caps iterations, tool calls, follow-ups, per-call extras, and
observation size. Repeated `(tool, arguments)` fingerprints are rejected.
The graph is never mutated.

Stop reasons: `answered`, `ambiguous`, `unresolved`, `not_found`,
`unsupported`, `budget_exhausted`, `no_progress`, `tool_failure`.

## Intentionally not implemented

Planner/LLM loops, MCP, schema changes, communities, swarm agents, and any
change to `FixedGraphRAGPipeline` or BM25.
