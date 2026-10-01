# Metrics (judge sheet)

Keep these populations **separate**. Do not merge them into one leaderboard.

Generator for all reported live scores: shared `SemanticGenerator`.
Graph status: live `OlympicGraph`.

## 1. Published public three-way benchmark

100 official public questions. This is the published public production score.
Replace `ui/catalog.py` `BENCHMARK` when a later official run is published.

| System | n | Correctness | Exact | Completeness | Citation | Grounding | Avg ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAG (BM25) | 100 | 67.0% | 67.0% | — | 95.0% | 95.0% | 9136 | 21491 |
| Fixed GraphRAG | 100 | 98.0% | 98.0% | 100% | 99.0% | 99.0% | 6290 | 13926 |
| Agentic GraphRAG | 100 | 97.0% | 97.0% | 100% | 99.0% | 99.0% | 6396 | 17077 |

Public completeness is complete-set aggregation only (graph). RAG has no
complete-set operator (cells are null).

### Public correctness by family

| Family | n | RAG | GraphRAG | Agentic |
|---|---:|---:|---:|---:|
| lookup | 19 | 100.0 | 100.0 | 100.0 |
| aggregation | 21 | 0.0 | 95.2 | 100.0 |
| superlative | 10 | 90.0 | 100.0 | 100.0 |
| temporal | 22 | 77.3 | 100.0 | 100.0 |
| multi_hop | 28 | 78.6 | 96.4 | 89.3 |

Public RAG's gap is almost entirely aggregation (extractive ceiling) plus some
temporal/multi-hop misses. Agentic did not beat Fixed GraphRAG overall (97 vs 98).

Agentic public tool stats: avg 1.55 tool calls; 72 questions one call; 28 with
follow-ups; 27 neighborhood uses.

Reproduce (does not manufacture a new published number unless you intentionally
replace the published public benchmark source in `ui/catalog.py`):

```
python -m scripts.eval_three_way --generator semantic
```

## 2. Vector ablation (experiment-only, not the published public benchmark)

Same public questions, RAG-only variants. GraphRAG and Agentic routing were
not changed. **Do not treat this table as the published public benchmark.** The
published public scores remain 67 / 98 / 97 in section 1.

These figures are **documented research results**. This snapshot does
not include committed JSON/CSV run artifacts, so a clone cannot replay the
table without re-running the experiment (optional; not required to judge the
production three-way).

| Variant | Mechanism | Public correctness (semantic) |
|---|---|---:|
| V0 | BM25 (production RAG control) | 66% |
| V1 | TigerGraph vector search | 53% |
| V2 | BM25 + vector hybrid | 65% |

V0 is a rerun of BM25 under the ablation harness (66% vs the published public
RAG score of 67%). Treat V0 as the experiment control, not a new public RAG score.

Recall@k in that experiment is **any gold document in the packed top-k**, not
full-set aggregation recall.

Optional entry point (does not change production routing):

```
python -m scripts.eval_vector_ablation --generator semantic --skip-ingest
```

## 3. Engineering experiments (not a leaderboard)

Additional engineering gates (for example temporal V0 vs V2 packing) are
research measurements. They are not headline accuracy, not the published public
benchmark, and must not be averaged with section 1.

Parser robustness unit tests check closed synonym/shape classes on public
templates. They are a contract suite, not a second leaderboard.
