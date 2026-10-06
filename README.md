# TigerGraph Agentic GraphRAG

A three-way comparison of retrieval policies over the same Olympic event corpus,
the same evidence packer, and the same `SemanticGenerator`:

1. **Basic RAG** — BM25 over chunks with the shared semantic generator. No graph retrieval.
2. **Fixed GraphRAG** — one typed GSQL retrieval per question.
3. **Bounded Agentic GraphRAG** — deterministic planner/tool selection with hard caps
   (3 iterations / 6 tool calls / 2 follow-ups). Not an unconstrained LLM planner.

TigerGraph is used as both the graph backend and the vector backend. Vector
search is real and production-wired as one optional, bounded Agentic follow-up;
it is not the default Basic RAG retriever and does not replace BM25 globally.

## What TigerGraph is doing

TigerGraph is not a citation store for text chunks. Installed GSQL answers
complete-set aggregations, typed event lookup, venue/date resolution, temporal
previous/next relations, and fail-closed ambiguity. Those operations model
count and relation questions that BM25 does not represent as typed graph
operations.

Vector search (`gsql/05_vector.gsql`) ranks 384-dimensional Chunk embeddings
with COSINE/HNSW. The Agentic pipeline may use it only when its evidence state
leaves a provenance gap. Fixed GraphRAG remains deterministic typed retrieval,
and Basic RAG remains BM25.

## Architecture

```mermaid
flowchart TD
  Q[User question] --> R[RAG BM25]
  Q --> P[Deterministic QuestionParser]
  P --> G[Fixed GraphRAG<br/>typed GSQL]
  P --> A[Bounded Agentic GraphRAG<br/>evidence-state decisions + rationale]
  G --> TG[(TigerGraph OlympicGraph)]
  A --> TG
  R --> E[Shared evidence pack]
  G --> E
  A --> E
  E --> S[Shared SemanticGenerator]
  S --> V[Fail-closed citation / grounding validator]
  V --> ANS[Answer or abstain / ambiguous]
  A -.->|optional bounded vector follow-up| TG
```

| Pipeline | Flow |
|---|---|
| RAG | question → BM25 → pack → shared generator → citation check |
| GraphRAG | question → parser → one typed GSQL call (`include_chunks=True`) → pack → shared generator → citation check |
| Agentic | question → parser → evidence-state-driven planner → `retrieve_spec` → optional graph/document/vector follow-up → pack → shared generator → citation check |

Gold answers are used only in evaluation. They never enter retrieval or generation.

## Setup

```
python -m pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` locally. Never commit it.

| Need | Variables |
|---|---|
| Live TigerGraph | `TG_HOST`, `TG_GRAPHNAME`, and one of `TG_API_TOKEN` / `TG_JWT_TOKEN` / `TG_SECRET` / `TG_USERNAME`+`TG_PASSWORD` |
| Semantic generation | `LLM_PROVIDER=openai_compatible`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` |
| Optional Agentic vector fallback | `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY` |

Corpus JSONL is **not** in Git. Place it at
`_research/hackathon-resources/corpus/corpus.jsonl`.
The public evaluation questions are tracked in this repository at
`_research/hackathon-resources/questions/eval_public.jsonl` and are the default
benchmark input. Private evaluation material is handled separately and is not
included in this public repository.

Without TigerGraph or the LLM provider, unit tests still run; live graph tests
skip, and semantic evaluation cannot reproduce the reported numbers.

## Tests

```
python -m unittest discover -s tests
```

Corpus-backed and live TigerGraph tests skip when those dependencies are missing.
See [docs/reproducibility.md](docs/reproducibility.md).

## Evaluation

Canonical public 100-question three-way (needs corpus + live TigerGraph + semantic provider):

```
python -m scripts.eval_three_way --generator semantic
```

Parser robustness (no live graph required; uses public wording only):

```
python -m unittest tests.test_parser_robustness
```

The live judge console can construct the optional Agentic vector fallback
when the graph and embedding setup are ready. The canonical 100-question
benchmark harness did not construct or pass a vector retriever, so the
fallback was neither wired nor exercised; the 99% Agentic result is not
attributed to vector retrieval.

The published public benchmark reports a descriptive 100-question semantic run:

| Population | RAG (BM25) | Fixed GraphRAG | Agentic GraphRAG |
|---|---:|---:|---:|
| Published public benchmark (100q) | 65% | 98% | 99% |

Full public metrics and methodology: [docs/public_benchmark.md](docs/public_benchmark.md).

Optional vector A/B experiment (does not change production routing):

```
python -m scripts.eval_vector_ablation --generator semantic --skip-ingest
```

## Parser

Five operations only: nation lookup, count-over-threshold, argmax competitors,
previous/next event gold, events at venue/date.

The parser uses **closed synonym/shape classes** (case, nations/countries,
over/more than, most/highest number, temporal and venue phrasing). Not an LLM
parser.

## Investigation Console

Judge-facing live console over the production pipelines (BM25 RAG, Fixed GraphRAG, Agentic GraphRAG).

```
python -m scripts.serve_ui
```

Then open `http://127.0.0.1:8765/`.

The main workspace is: question → pipeline (or Compare All) → answer → evidence → investigation trace → evidence diff → graph context → why it continued or stopped → metrics.

Graph pipelines need a configured, reachable TigerGraph instance (`TG_*`), and RAG needs the corpus at `_research/hackathon-resources/corpus/corpus.jsonl`. When semantic settings (`LLM_*`) are present, all three pipelines use the shared `SemanticGenerator`; without them, the console uses its deterministic generator, which does not reproduce the published semantic benchmark. Missing graph or corpus dependencies are reported as **UNAVAILABLE**. Health checks are read-only. Agentic traces explain why a follow-up ran or why the agent stopped, using only observed planner/tool data.

| Goal | Where |
|---|---|
| Live investigation | **Investigate** — type any question, choose a pipeline, Run |
| Three-way comparison | **Compare** — same question through RAG, GraphRAG, Agentic |
| Published scores | **Benchmark** — published public 100-question three-way |
| Runtime / graph health | **System** — read-only checks; vector readiness and routing are explicit |
| Replay | **History** — current browser session only |
| Export | **Reports** — sanitized schema_version=1 JSON |

Export uses `evaluation.trace_export`. Evaluator scoring fields are not shown.

Alternative: `python -m ui --port 8765`.

## Limitations

- The parser is a bounded question-language contract, not open-domain NLU.
- Graph evaluation needs a live `OlympicGraph` with the repo GSQL installed.
- Reported semantic scores need the configured completion provider.
- The corpus is external; this snapshot includes the public question file, not `corpus.jsonl`.
- Benchmarks test this corpus and these templates, not arbitrary natural language.
- Basic RAG is BM25. TigerGraph vector retrieval is selective Agentic fallback, not a global BM25 replacement.
- GRIP / MCP is not part of the public production runtime.
- Agentic follow-ups were used selectively under a hard budget; the public
  benchmark recorded 28 follow-up investigations.

## Docs

- [Architecture (as implemented)](docs/architecture.md)
- [Metrics for judges](docs/metrics.md)
- [Demo questions](docs/demo.md)
- [Reproduction](docs/reproducibility.md)
- Public benchmark: [summary and methodology](docs/public_benchmark.md)

## License

This repository's original source is released under the [MIT License](LICENSE).
Python packages in `requirements.txt` remain under their own licenses.
