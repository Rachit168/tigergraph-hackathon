# Public benchmark

This is a sanitized summary of the final public semantic benchmark. It
describes the measured behavior of the three production pipelines on 100
visible public questions. It does not publish per-question answers, expected
answers, gold document IDs, raw traces, or evaluator-only fields.

All three pipelines used the same `SemanticGenerator` configuration: model
`dynamic/olympic-llm`, temperature `0`, and maximum completion tokens `1024`.
Evaluation is separate from retrieval and generation; evaluation-only
reference data is never passed to a pipeline.

## Headline measurements

| System | Questions | Semantic accuracy | Average latency | p95 latency | Tokens | Generation errors |
|---|---:|---:|---:|---:|---:|---:|
| Basic RAG (BM25) | 100 | 65% | 10.67 s | 30.68 s | 295,344 | 4 |
| Fixed GraphRAG | 100 | 98% | 7.59 s | 17.78 s | 146,180 | 1 |
| Agentic GraphRAG | 100 | 99% | 7.11 s | 17.19 s | 123,016 | 0 |

These are descriptive results for this corpus, question set, model
configuration, and run. They are not guarantees or a universal ranking.

## Question-family breakdown

| Family | Questions | Basic RAG | Fixed GraphRAG | Agentic GraphRAG |
|---|---:|---:|---:|---:|
| Direct factual / lookup | 19 | 100% | 100% | 100% |
| Complete-set aggregation | 21 | 0% | 95.2% | 100% |
| Relationship / multi-hop | 28 | 78.6% | 96.4% | 96.4% |
| Comparative / superlative | 10 | 80% | 100% | 100% |
| Temporal | 22 | 72.7% | 100% | 100% |

## Agentic behavior

- 72 investigations used one tool call.
- 28 investigations used follow-up retrieval.
- 27 investigations used neighborhood retrieval.
- Average retrieval/tool actions: 1.55.
- Average Agentic steps: 1.28.
- Average follow-ups: 0.28.
- 99 investigations were answered and 1 remained ambiguous.
- No budget-exhausted or no-progress stops occurred.
- The production Agentic runtime supports an optional bounded TigerGraph
  Vector fallback. The canonical benchmark harness did not construct or pass
  a vector retriever, so the fallback was neither wired nor exercised; the
  99% result is not attributable to vector retrieval.

The Agentic pipeline is evidence-state-driven; the canonical benchmark did
not wire vector retrieval into its harness.

## Pipeline and backend methodology

- **Basic RAG:** sparse BM25 chunk retrieval, evidence packing, and shared
  semantic generation. It does not use graph retrieval.
- **Fixed GraphRAG:** one deterministic typed TigerGraph retrieval followed by
  evidence packing and the same semantic generator.
- **Agentic GraphRAG:** bounded evidence-state-driven investigation using
  deterministic graph tools, with an optional single bounded TigerGraph vector
  fallback when the evidence state warrants it.
- **TigerGraph:** used as the graph backend and vector backend. The vector
  capability uses BAAI/bge-small-en-v1.5 embeddings with dimension 384 and a
  COSINE/HNSW index.

## Vector measurements and limitation

The separate public vector A/B experiment measured:

| Variant | Result |
|---|---:|
| BM25 baseline | 66% |
| TigerGraph Vector alone | 53% |
| BM25 + TigerGraph Vector hybrid/RRF | 65% |

The vector smoke test also verified five real TigerGraph Chunk hits with
`method=tigergraph_vector`, backend `tigergraph_vector`, embedding model
`BAAI/bge-small-en-v1.5`, and dimension 384. These measurements support
selective Agentic use; they do not support replacing BM25 globally. The
verified infrastructure snapshot reports 28,905 embedded Chunk vertices;
readiness checks do not perform an expensive full live count.

## Reproduction and public/private boundary

With the public corpus available, a configured read-only TigerGraph, and the
same semantic provider settings:

```text
python -m scripts.eval_three_way --generator semantic
```

The command evaluates public questions only. A fresh run writes local output
under `_research/`; it does not replace this summary automatically. Raw scored
artifacts under `data/final_public_benchmark/` are intentionally ignored and
must remain untracked because they contain evaluator-only fields. The public
repository contains this sanitized summary instead.

Hidden evaluation is handled as a separate evaluator-side process. Hidden
questions and reference data are not stored in this public repository, are not
used by the live console, and are not described or reconstructed here.

## Limitations

- Results are specific to this corpus, 100-question public set, generator
  configuration, and measured run.
- Basic RAG has no complete-set graph operator, so aggregation behavior differs
  by design.
- The parser supports a bounded question-language contract rather than open
  domain natural-language understanding.
- Canonical benchmark reproduction requires the external organizer-provided
  public corpus (see [reproducibility](reproducibility.md)), a configured
  TigerGraph graph with the required GSQL queries, and semantic-provider
  configuration.
- Embedding configuration and a populated vector index are optional and are
  required only to enable the live Agentic vector fallback or run the vector
  experiment; they are not requirements for the canonical benchmark.
