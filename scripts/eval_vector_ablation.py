"""Phase 13 TigerGraph Vector RAG A/B: V0 BM25, V1 vector, V2 hybrid.

Does not modify GraphRAG or the production Agentic planner.
Scores official public questions only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from answering.factory import build_generator
from answering.packer import ContextPacker
from config.embeddings import embedding_cache_path, load_embedding_settings
from config.llm import load_llm_settings
from config.settings import load_settings
from evaluation.benchmark import git_revision, public_run_id, score_harness_result, utc_timestamp
from evaluation.harness import RAGAdapter
from evaluation.three_way import CSV_COLUMNS, load_public_questions
from experiments.vector_metrics import (
    GOLD_QTYPES,
    STATUS_BUCKETS,
    VARIANT_LABELS,
    VARIANT_ORDER,
    RECALL_AT_K_DEFINITION,
    attach_retrieval_metrics,
    fairness_report,
    summarize_vector_ablation,
)
from ingestion.chunker import ChunkingConfig, chunk_corpus
from ingestion.embeddings import EmbeddingCache, build_embedder
from ingestion.loader import parse_corpus
from ingestion.paths import DEFAULT_CORPUS_PATH, DEFAULT_PUBLIC_QUESTIONS_PATH, REPO_ROOT
from retrieval.graph.client import TigerGraphClient
from retrieval.graph.vector import VectorBackendError
from retrieval.graph.vector_store import TigerGraphVectorStore
from retrieval.rag.hybrid_vector import HybridBM25VectorRetriever
from retrieval.rag.retriever import TextRetriever
from retrieval.rag.vector import TigerGraphVectorRetriever
from scripts.ingest_chunk_vectors import ingest_chunk_vectors

DEFAULT_OUTPUT_DIR = REPO_ROOT / "_research" / "phase13_vector_ablation"
CSV_FIELDS = ("variant", "variant_label") + CSV_COLUMNS + (
    "recall_at_5",
    "recall_at_10",
    "candidate_recall",
    "packed_recall",
    "retrieval_ms",
    "generation_ms",
    "backend",
)


def parse_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 13 TigerGraph Vector RAG ablation")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--questions", type=Path, default=DEFAULT_PUBLIC_QUESTIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--generator", default="semantic", choices=("deterministic", "semantic"))
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument(
        "--skip-ensure",
        action="store_true",
        help=(
            "Read-only live backend check: do not CREATE/ALTER schema, "
            "CREATE OR REPLACE/INSTALL queries, or upsert vectors"
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args(argv)


def variant_adapters(sparse, vector, hybrid, packer, generator, *, top_k: int) -> dict[str, RAGAdapter]:
    return {
        "V0": RAGAdapter(sparse, packer, generator, method="sparse", top_k=top_k),
        "V1": RAGAdapter(vector, packer, generator, method="tigergraph_vector", top_k=top_k),
        "V2": RAGAdapter(hybrid, packer, generator, method="hybrid", top_k=top_k),
    }


def main() -> int:
    args = parse_eval_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_public_questions(args.questions)
    if args.limit:
        records = records[: args.limit]
    settings = load_settings()
    llm = load_llm_settings()
    embedding_settings = load_embedding_settings(llm=llm)
    corpus = parse_corpus(args.corpus)
    chunks = chunk_corpus(corpus.documents, ChunkingConfig())
    packer = ContextPacker()
    generator = build_generator(args.generator)
    sparse = TextRetriever(chunks)
    config = {
        "benchmark": "phase13_vector_ablation",
        "questions_name": args.questions.name,
        "corpus_name": args.corpus.name,
        "generator": generator.__class__.__name__,
        "embedding": embedding_settings.redacted(),
        "graph": settings.redacted(),
        "llm": llm.redacted(),
        "git_revision": git_revision(),
        "generated_at": utc_timestamp(),
        "private_eval_used": False,
        "production_agentic_changed": False,
        "production_graphrag_changed": False,
        "top_k": args.top_k,
        "vector_mechanism": "TigerGraph VECTOR ATTRIBUTE + installed vectorSearch() query",
        "production_scorer": "evaluation.benchmark.score_harness_result",
    }

    client = TigerGraphClient()
    if not client.connect():
        return _fail_closed(output_dir, config, f"TigerGraph connect failed: {client.error}", records)
    try:
        version = str(client.conn.getVer()) if hasattr(client.conn, "getVer") else "unknown"
    except Exception:
        version = "unknown"
    config["tigergraph_version"] = version
    config["tigergraph_environment"] = client.environment

    try:
        embedder = build_embedder(embedding_settings)
        ingest = ingest_chunk_vectors(
            client=client,
            chunks=chunks,
            embedder=embedder,
            cache=None if args.skip_ensure else EmbeddingCache(embedding_cache_path(embedder.model)),
            skip_upsert=bool(args.skip_ingest or args.skip_ensure),
            skip_ensure=bool(args.skip_ensure),
        )
        vector = TigerGraphVectorRetriever(
            TigerGraphVectorStore(client, dimension=embedder.dimension),
            chunks,
            embedder,
        )
        hybrid = HybridBM25VectorRetriever(sparse, vector)
    except (VectorBackendError, Exception) as exc:
        if isinstance(exc, VectorBackendError) or "vector" in str(exc).casefold():
            return _fail_closed(output_dir, config, str(exc), records)
        raise

    config["ingest"] = {
        key: ingest.get(key)
        for key in ("n_chunks", "cache_hits", "embedded", "upserted", "model", "dimension", "vector_attribute", "search_query")
    }
    variants = variant_adapters(sparse, vector, hybrid, packer, generator, top_k=args.top_k)
    all_rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    try:
        for variant, adapter in variants.items():
            rows = []
            for index, record in enumerate(records, start=1):
                result = adapter.run(str(record.get("question") or ""), qtype=record.get("qtype"))
                _assert_fairness(variant, result)
                row = score_harness_result(record, result)
                row["variant"] = variant
                row["variant_label"] = VARIANT_LABELS[variant]
                row["graph_operations"] = False
                row = attach_retrieval_metrics(row, list(record.get("gold_doc_ids") or []), result)
                rows.append(row)
                print(f"{variant} {index}/{len(records)} {record.get('qid')}", flush=True)
            payload = {
                "run_id": public_run_id(
                    {
                        "benchmark": "phase13_vector_ablation",
                        "variant": variant,
                        "n_questions": len(records),
                        "qids": [record.get("qid") for record in records],
                        "generator": generator.__class__.__name__,
                        "embedding_model": embedding_settings.model,
                    }
                ),
                "config": config,
                "n_questions": len(records),
                "n_rows": len(rows),
                "rows": rows,
            }
            payloads[variant] = payload
            all_rows.extend(rows)
            (output_dir / f"{variant.lower()}_results.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
    except VectorBackendError as exc:
        return _fail_closed(output_dir, config, str(exc), records)

    summary = summarize_vector_ablation(all_rows)
    summary["fairness"] = fairness_report(all_rows)
    comparison = {"config": config, "summary": summary, "n_questions": len(records)}
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    _write_comparison_csv(summary.get("overall") or {}, output_dir / "comparison.csv")
    _write_recall_csv(all_rows, output_dir / "retrieval_recall.csv")
    report = render_report(comparison)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def _assert_fairness(variant: str, result) -> None:
    params = (result.retrieval_metadata or {}).get("params") or {}
    method = str((result.retrieval_metadata or {}).get("method") or "")
    if variant == "V0":
        if params.get("backend") != "bm25" and "bm25" not in method:
            raise VectorBackendError("V0 did not run BM25")
        if "vector" in str(params.get("backend") or ""):
            raise VectorBackendError("V0 used vector retrieval")
    if variant == "V1":
        if params.get("backend") != "tigergraph_vector":
            raise VectorBackendError("V1 did not run TigerGraph vector search")
        if params.get("bm25") or "bm25" in method:
            raise VectorBackendError("V1 silently fell back to BM25")
        if params.get("graph_operations"):
            raise VectorBackendError("V1 used graph retrieval operations")
    if variant == "V2":
        backends = params.get("backends") or []
        if "bm25" not in backends or "tigergraph_vector" not in backends:
            raise VectorBackendError("V2 did not run both BM25 and TigerGraph vector")
        if params.get("graph_operations"):
            raise VectorBackendError("V2 used graph retrieval operations")


def _fail_closed(output_dir: Path, config: dict[str, Any], error: str, records: list[dict[str, Any]]) -> int:
    payload = {
        "ok": False,
        "error": error,
        "config": config,
        "n_questions": len(records),
        "note": "TigerGraph vector search failed. BM25 was not substituted for V1/V2.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report = (
        "# Phase 13 TigerGraph Vector RAG ablation\n\n"
        "FAIL CLOSED. TigerGraph vector search was unavailable or failed.\n\n"
        f"- Error: {error}\n"
        "- V1 was not replaced with BM25.\n"
        "- V2 was not converted into a vector-only or BM25-only run.\n"
        "- Only official public questions were scored.\n"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 1


def _write_comparison_csv(overall: dict[str, Any], path: Path) -> None:
    fields = [
        "variant",
        "label",
        "n",
        "correctness",
        "correctness_exact",
        "completeness",
        "grounding",
        "citation_validity",
        "recall_at_5",
        "recall_at_10",
        "candidate_recall",
        "packed_recall",
        "tokens_total",
        "latency_p50_ms",
        "latency_p95_ms",
        "retrieval_p50_ms",
        "generation_p50_ms",
        "generation_failures",
        "abstentions",
        "ambiguous",
        "not_found",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for vid in VARIANT_ORDER:
            stats = overall.get(vid)
            if not stats:
                continue
            writer.writerow(
                {
                    "variant": vid,
                    "label": VARIANT_LABELS.get(vid, vid),
                    **{key: stats.get(key) for key in fields if key not in {"variant", "label"}},
                }
            )


def _write_recall_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "variant",
        "qid",
        "qtype",
        "recall_at_5",
        "recall_at_10",
        "candidate_recall",
        "packed_recall",
        "correctness",
        "backend",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def render_report(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    overall = summary.get("overall") or {}
    deltas = summary.get("deltas_vs_V0") or {}
    families = summary.get("by_family") or {}
    fairness = summary.get("fairness") or {}
    config = payload.get("config") or {}
    lines = [
        "# Phase 13 TigerGraph Vector RAG ablation",
        "",
        "V0 = current BM25 RAG. V1 = TigerGraph Vector only. V2 = BM25 + TigerGraph Vector RRF hybrid.",
        "GraphRAG and the production BoundedPlanner were not modified.",
        "",
        f"- Recall@k definition: {RECALL_AT_K_DEFINITION}.",
        "- Gold qtypes are lookup, temporal, multi_hop, superlative, aggregation. "
        "There is no synthetic venue_disambiguation family. "
        "Abstention and ambiguity are answer-status buckets, not gold qtypes.",
        "",
        f"- TigerGraph version: `{config.get('tigergraph_version')}` ({config.get('tigergraph_environment')})",
        f"- Vector mechanism: {config.get('vector_mechanism')}",
        f"- Embedding model: `{(config.get('embedding') or {}).get('model')}` dim={(config.get('embedding') or {}).get('dimension')}",
        f"- Generator: `{config.get('generator')}`",
        f"- Questions: {payload.get('n_questions')}",
        "- Only official public questions were scored.",
        "",
        "## Overall",
        "",
        "| variant | corr % | exact % | ground % | cite % | R@5 | R@10 | cand recall | packed recall | tokens | p50 ms | p95 ms | ret p50 | gen fail | abstain | amb | not_found |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for vid in VARIANT_ORDER:
        stats = overall.get(vid) or {}
        lines.append(
            f"| {vid} | {_fmt(stats.get('correctness'))} | {_fmt(stats.get('correctness_exact'))} | "
            f"{_fmt(stats.get('grounding'))} | {_fmt(stats.get('citation_validity'))} | "
            f"{_fmt(stats.get('recall_at_5'))} | {_fmt(stats.get('recall_at_10'))} | "
            f"{_fmt(stats.get('candidate_recall'))} | {_fmt(stats.get('packed_recall'))} | "
            f"{_fmt(stats.get('tokens_total'))} | {_fmt(stats.get('latency_p50_ms'))} | "
            f"{_fmt(stats.get('latency_p95_ms'))} | {_fmt(stats.get('retrieval_p50_ms'))} | "
            f"{stats.get('generation_failures')} | {stats.get('abstentions')} | "
            f"{stats.get('ambiguous')} | {stats.get('not_found')} |"
        )
    lines.extend(["", "## Deltas vs V0", "", "| variant | corr pp | R@5 pp | R@10 pp | cand recall pp | tokens | p50 ms | p95 ms |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for vid in ("V1", "V2"):
        stats = deltas.get(vid) or {}
        lines.append(
            f"| {vid} | {_fmt(stats.get('correctness_pp'))} | {_fmt(stats.get('recall_at_5_pp'))} | "
            f"{_fmt(stats.get('recall_at_10_pp'))} | {_fmt(stats.get('candidate_recall_pp'))} | "
            f"{_fmt(stats.get('tokens_total'))} | {_fmt(stats.get('latency_p50_ms'))} | "
            f"{_fmt(stats.get('latency_p95_ms'))} |"
        )
    lines.extend(["", "## Category correctness", ""])
    for family in GOLD_QTYPES + STATUS_BUCKETS:
        block = families.get(family) or {}
        if not block:
            continue
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| variant | n | corr % | R@10 | tokens | p50 ms | status |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for vid in VARIANT_ORDER:
            stats = block.get(vid)
            if not stats:
                continue
            lines.append(
                f"| {vid} | {stats.get('n')} | {_fmt(stats.get('correctness'))} | "
                f"{_fmt(stats.get('recall_at_10'))} | {_fmt(stats.get('tokens_total'))} | "
                f"{_fmt(stats.get('latency_p50_ms'))} | `{json.dumps(stats.get('status_counts') or {}, sort_keys=True)}` |"
            )
        lines.append("")
    v0 = overall.get("V0") or {}
    v1 = overall.get("V1") or {}
    v2 = overall.get("V2") or {}
    d1 = deltas.get("V1") or {}
    d2 = deltas.get("V2") or {}
    benefit, regress = _family_moves(families)
    recall_gain = (d1.get("recall_at_10_pp") or 0) > 0.5 or (d2.get("recall_at_10_pp") or 0) > 0.5
    corr_gain = (d1.get("correctness_pp") or 0) > 0.5 or (d2.get("correctness_pp") or 0) > 0.5
    hybrid_best = (
        v2.get("correctness") is not None
        and v2.get("correctness") >= max(v0.get("correctness") or 0, v1.get("correctness") or 0)
        and v2.get("recall_at_10") is not None
        and v2.get("recall_at_10") >= max(v0.get("recall_at_10") or 0, v1.get("recall_at_10") or 0)
    )
    replace = None
    if (d2.get("correctness_pp") or 0) > 0.5 and (d2.get("recall_at_10_pp") or 0) >= 0 and not regress:
        replace = "V2"
    elif (d1.get("correctness_pp") or 0) > 0.5 and (d1.get("recall_at_10_pp") or 0) >= 0 and not regress:
        replace = "V1"
    lines.extend(
        [
            "## Fairness",
            "",
            f"- V1/V2 graph ops: {fairness.get('v1_has_graph_ops')}/{fairness.get('v2_has_graph_ops')}",
            f"- V0 used vector: {fairness.get('v0_used_vector')}",
            f"- V1 used BM25: {fairness.get('v1_used_bm25')}",
            f"- V2 missing BM25/vector: {fairness.get('v2_missing_bm25')}/{fairness.get('v2_missing_vector')}",
            f"- Private eval used: {fairness.get('private_eval_used')}",
            "",
            "## Required answers",
            "",
            f"1. Does TigerGraph Vector improve retrieval recall over BM25? **{'Yes' if recall_gain else 'No'}.** "
            f"V0 R@10={_fmt(v0.get('recall_at_10'))}; V1={_fmt(v1.get('recall_at_10'))}; V2={_fmt(v2.get('recall_at_10'))}.",
            f"2. Does Vector improve end-to-end correctness? **{'Yes' if corr_gain else 'No'}.** "
            f"V0={_fmt(v0.get('correctness'))}; V1={_fmt(v1.get('correctness'))}; V2={_fmt(v2.get('correctness'))}.",
            f"3. Does hybrid improve over either individual retriever? **{'Yes' if hybrid_best else 'No'}.**",
            f"4. What is the token cost difference? V1-V0={_fmt(d1.get('tokens_total'))}; V2-V0={_fmt(d2.get('tokens_total'))}.",
            f"5. What is the latency difference? V1 p50={_fmt(d1.get('latency_p50_ms'))} ms; V2 p50={_fmt(d2.get('latency_p50_ms'))} ms.",
            f"6. Which question families benefit? **{', '.join(benefit) if benefit else 'none'}.**",
            f"7. Which question families regress? **{', '.join(regress) if regress else 'none'}.**",
            "8. Does Vector materially reduce any of the current RAG failure classes? See status counts above; "
            f"V0 abstentions={v0.get('abstentions')}, V1={v1.get('abstentions')}, V2={v2.get('abstentions')}.",
            f"9. Is the additional complexity justified by measured results? **{'Yes' if replace else 'No'}.**",
            f"10. Which RAG retrieval method should become the final Round 1 RAG baseline? **{replace or 'V0 / BM25'}.**",
            "",
            "## Recommendation",
            "",
            (
                f"Promote {replace} to the Round 1 RAG baseline only after an explicit follow-up. "
                "GraphRAG and Agentic stay unchanged."
                if replace
                else "Keep V0 BM25 as the Round 1 RAG baseline. Measured gains do not clearly justify replacing it."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _family_moves(families: dict[str, Any]) -> tuple[list[str], list[str]]:
    benefit: list[str] = []
    regress: list[str] = []
    for family, block in families.items():
        v0 = (block.get("V0") or {}).get("correctness")
        best = None
        for vid in ("V1", "V2"):
            value = (block.get(vid) or {}).get("correctness")
            if value is None:
                continue
            best = value if best is None else max(best, value)
        if v0 is None or best is None:
            continue
        if float(best) - float(v0) >= 0.1:
            benefit.append(f"{family} {v0}->{best}")
        if float(best) - float(v0) <= -0.1:
            regress.append(f"{family} {v0}->{best}")
    return benefit, regress


if __name__ == "__main__":
    raise SystemExit(main())
