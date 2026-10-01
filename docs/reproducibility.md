# Reproduction

How a fresh clone becomes a running three-way system. No secrets belong in Git.

This public snapshot only documents **public** data and the three production
pipelines. It does not instruct a clone to load private evaluation files.

## 1. Install

Python 3.12+ recommended (developed against 3.14). From the repo root:

```
python -m pip install -r requirements.txt
```

`requirements.txt` lists:

- **Production essentials:** `pyTigerGraph`, `python-dotenv`
- **Optional vector experiments only:** `numpy`, `fastembed`

The three-way public benchmark (BM25 RAG, fixed GraphRAG, BoundedPlanner Agentic)
does **not** require FastEmbed or numpy. Those packages support the separate
TigerGraph vector experiment path. Vector experiments also read the embedding
client settings in `.env` (see `.env.example`).

## 2. Configure environment

```
copy .env.example .env
```

Edit `.env`. Never commit it (`.gitignore` already excludes `.env`).

TigerGraph (required for graph tests and graph evaluation):

- `TG_HOST` — Savanna / Cloud REST host
- `TG_GRAPHNAME=OlympicGraph`
- Auth: `TG_API_TOKEN` or `TG_JWT_TOKEN` or `TG_SECRET` or `TG_USERNAME` + `TG_PASSWORD`

Semantic generation (required to reproduce reported scores):

- `LLM_PROVIDER=openai_compatible`
- `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`
- Optional: `LLM_TIMEOUT_S` (default 30s), `LLM_MAX_TOKENS` (freeze **1024**), auth header for a gateway
- `LLM_MAX_TOKENS=1024` is shared by RAG, GraphRAG, and Agentic. Historical live runs used 512; that cap truncated some JSON answers that copy verbatim evidence quotes. Do not set a per-pipeline override.

Leave `LLM_PROVIDER=none` to keep the deterministic generator. That path will
not match the published semantic percentages.

Optional vector experiments:

- `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`

## 3. Corpus (not in Git)

Place the hackathon corpus at:

```
_research/hackathon-resources/corpus/corpus.jsonl
```

The official public question file is in this snapshot at:

```
_research/hackathon-resources/questions/eval_public.jsonl
```

It is present in the public worktree and is the default input for
`scripts.eval_three_way`. It has not been committed yet.

Without the corpus JSONL, corpus-backed tests skip. Graph evaluation still
needs the same documents loaded into TigerGraph.

## 4. Connect TigerGraph and install GSQL

One-time against an empty or reset `OlympicGraph` (see `gsql/`):

```
python -m scripts.ingest_tigergraph
python -m scripts.verify_tigergraph
```

`python -m scripts.verify_tigergraph` is **read-only**: it checks connectivity, schema, and installed queries, then writes JSON to `data/tigergraph/verify.json`. It does not reset, bind, upsert, or install GSQL. Use `ingest_tigergraph` for those steps. `--skip-live` checks the local export contract without connecting.

`gsql/00_schema.gsql` … `04_retrieval.gsql` are the production schema, load, and
queries. `gsql/05_vector.gsql` is the optional vector experiment install.
Evaluation scripts are **read-only**: they do not reset or reinstall.

Confirm:

```
python -c "from retrieval.graph.client import TigerGraphClient; c=TigerGraphClient(); print(c.connect(), c.verify_queries().get('ok'), c.verify_retrieval_queries().get('ok'))"
```

## 5. Tests

```
python -m unittest discover -s tests
```

Corpus-backed and live TigerGraph tests skip when those dependencies are missing.

### Which tests need the corpus

Skip if `corpus.jsonl` is absent: `test_parser_corpus`, `test_structured_corpus`,
`test_text_retrieval_corpus`, corpus cases in `test_graph_equivalence`,
and public cases in `test_parser_robustness`.

### Which tests need live TigerGraph

Skip unless settings are configured: `test_tigergraph_live`,
`test_graph_retrieval_live`, `test_phase6_live`, `test_phase7_live`.
Those live tests are read-only against the graph.

`tests/test_tigergraph_vector.py` is a **unit-test** module. It does not need a
live graph or installed vector queries. The public unit-test run includes it
and passes without TigerGraph credentials.

### Which tests need the LLM provider

Unit tests mock or avoid the provider (`test_semantic_answering`). Live
percentage reproduction is **evaluation**, not the unit suite.

## 6. Public evaluation (canonical)

Needs: corpus, live graph with queries, semantic provider.

```
python -m scripts.eval_three_way --generator semantic
```

Default output: `_research/phase8_public_three_way/` (local, gitignored via
`_research/phase*`).

`--generator deterministic` runs without an LLM and will not match the 67/98/97
semantic baseline.

The published canonical number remains the Phase 10 semantic freeze. A new run
is a reproduction check, not an automatic replacement of that table.

## 7. Vector ablation (optional, not the production baseline)

```
python -m scripts.eval_vector_ablation --generator semantic --skip-ingest
python -m scripts.eval_temporal_vector_gate --generator semantic
```

These scripts score public questions only. They do not change RAG/GraphRAG/Agentic
production routing.

## 8. Parser robustness

No live graph required:

```
python -m unittest tests.test_parser_robustness
```

## 9. What a public clone will not contain

- `.env` and credentials
- `corpus.jsonl`
- `_research/phase*` result dumps
- Private evaluation datasets

## 10. Ingest / export helpers

```
python -m scripts.parse_corpus
python -m scripts.export_graph
python -m scripts.verify_tigergraph
```

Do not point these at production passwords in shell history dumps that you
commit. Evaluation never needs write probes.
