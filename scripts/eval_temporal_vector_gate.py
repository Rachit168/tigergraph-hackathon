"""Read-only temporal V0 BM25 vs V2 hybrid gate. Does not change production routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from answering.factory import build_generator
from answering.packer import ContextPacker
from config.embeddings import load_embedding_settings
from config.llm import load_llm_settings
from config.settings import load_settings
from evaluation.benchmark import git_revision, public_run_id, score_harness_result, utc_timestamp
from evaluation.harness import RAGAdapter
from evaluation.three_way import load_public_questions
from experiments.temporal_vector_gate import (
    RecordingRetriever,
    packed_chunk_ids,
    packed_document_ids,
    parse_named_year,
    previous_doc_entered_packed,
    previous_olympiad_gold,
    hits_payload,
)
from experiments.vector_metrics import RECALL_AT_K_DEFINITION, attach_retrieval_metrics
from ingestion.chunker import ChunkingConfig, chunk_corpus
from ingestion.embeddings import build_embedder
from ingestion.loader import parse_corpus
from ingestion.paths import DEFAULT_CORPUS_PATH, DEFAULT_PUBLIC_QUESTIONS_PATH, REPO_ROOT
from retrieval.graph.client import TigerGraphClient
from retrieval.graph.vector import VectorBackendError
from retrieval.graph.vector_store import TigerGraphVectorStore
from retrieval.rag.hybrid_vector import HybridBM25VectorRetriever
from retrieval.rag.retriever import TextRetriever
from retrieval.rag.vector import TigerGraphVectorRetriever
from scripts.ingest_chunk_vectors import ingest_chunk_vectors

DEFAULT_OUTPUT_DIR = REPO_ROOT / "_research" / "phase13_temporal_vector_gate"
ENGINEERING_SET_PATH = DEFAULT_OUTPUT_DIR / "engineering_temporal_paraphrases.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only temporal V0 vs V2 vector gate")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--questions", type=Path, default=DEFAULT_PUBLIC_QUESTIONS_PATH)
    parser.add_argument("--engineering-set", type=Path, default=ENGINEERING_SET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generator", default="semantic", choices=("deterministic", "semantic"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--set",
        dest="which",
        default="both",
        choices=("official", "robustness", "both"),
    )
    return parser.parse_args(argv)


def load_engineering_set(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    records.sort(key=lambda row: str(row.get("qid") or ""))
    return records


def capture_row(
    *,
    record: dict[str, Any],
    variant: str,
    adapter: RAGAdapter,
    recorder: RecordingRetriever,
    vector_recorder: RecordingRetriever | None,
    previous_meta: dict[str, Any],
) -> dict[str, Any]:
    result = adapter.run(str(record.get("question") or ""), qtype=record.get("qtype"))
    row = score_harness_result(record, result)
    row["variant"] = variant
    row["graph_operations"] = False
    row = attach_retrieval_metrics(row, list(record.get("gold_doc_ids") or []), result)
    packed_docs = packed_document_ids(result.evidence_metadata)
    packed_chunks = packed_chunk_ids(result.evidence_metadata)
    last = recorder.last_result
    params = dict((last.params if last is not None else {}) or {})
    candidate_chunks = [str(value) for value in (params.get("candidate_chunk_ids") or [])]
    candidate_docs = [str(value) for value in (params.get("candidate_document_ids") or [])]
    vector_hits = hits_payload(vector_recorder.last_result if vector_recorder else None, include_scores=True)
    previous_id = previous_meta.get("previous_olympiad_doc_id")
    row.update(
        {
            "set_name": record.get("set") or "official_public_temporal",
            "official_benchmark": bool(record.get("official_benchmark", True)),
            "source_qid": record.get("source_qid"),
            "paraphrase_pattern": record.get("paraphrase_pattern"),
            "named_year": previous_meta.get("named_year"),
            "previous_olympiad_doc_id": previous_id,
            "named_year_doc_id": previous_meta.get("named_year_doc_id"),
            "previous_olympiad_identifiable": previous_meta.get("identifiable"),
            "gold_years": previous_meta.get("gold_years"),
            "candidate_chunk_ids": candidate_chunks or None,
            "candidate_document_ids": candidate_docs or None,
            "packed_chunk_ids": packed_chunks or None,
            "packed_document_ids": packed_docs,
            "packed_evidence_count": len(result.evidence_metadata or []),
            "previous_olympiad_in_candidates": (
                previous_id in set(candidate_docs) if previous_id and candidate_docs else None
            ),
            "previous_olympiad_in_packed": previous_doc_entered_packed(previous_id, packed_docs),
            "vector_hit_scores": vector_hits or None,
            "private_eval_used": False,
        }
    )
    return row


def summarize_pair(official_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_var: dict[str, list[dict[str, Any]]] = {"V0": [], "V2": []}
    for row in official_rows:
        by_var.setdefault(str(row.get("variant")), []).append(row)
    out: dict[str, Any] = {}
    for variant, items in by_var.items():
        n = len(items)
        correct = sum(1 for row in items if row.get("correctness"))
        packed_prev = [row for row in items if row.get("previous_olympiad_identifiable")]
        out[variant] = {
            "n": n,
            "correct": correct,
            "correctness": round(100.0 * correct / n, 1) if n else None,
            "abstentions": sum(1 for row in items if row.get("answer_status") == "abstained"),
            "ambiguous": sum(1 for row in items if row.get("answer_status") == "ambiguous"),
            "generation_error": sum(1 for row in items if row.get("answer_status") == "generation_error"),
            "mean_packed_recall": round(
                sum(float(row.get("packed_recall") or 0) for row in items) / n, 4
            )
            if n
            else None,
            "previous_olympiad_in_packed": sum(
                1 for row in packed_prev if row.get("previous_olympiad_in_packed")
            ),
            "previous_olympiad_identifiable": len(packed_prev),
            "tokens_total": int(sum(int(row.get("tokens") or 0) for row in items)),
        }
    return out


def render_report(payload: dict[str, Any]) -> str:
    official = payload.get("official_summary") or {}
    robust = payload.get("robustness_summary") or {}
    v0 = official.get("V0") or {}
    v2 = official.get("V2") or {}
    hypothesis = payload.get("hypothesis") or {}
    lines = [
        "# Phase 13C temporal vector gate",
        "",
        "Engineering experiment only. Production BM25 / GraphRAG / Agentic were not changed.",
        "TigerGraph was used read-only (skip-ensure: no schema/query install, no upsert).",
        "Only official public questions were scored.",
        "",
        f"- Recall@k definition: {RECALL_AT_K_DEFINITION}.",
        f"- Official public temporal n={v0.get('n')}",
        f"- V0 correctness={v0.get('correctness')} packed_prev={v0.get('previous_olympiad_in_packed')}/{v0.get('previous_olympiad_identifiable')}",
        f"- V2 correctness={v2.get('correctness')} packed_prev={v2.get('previous_olympiad_in_packed')}/{v2.get('previous_olympiad_identifiable')}",
        "",
        "## Hypothesis decision",
        "",
        f"- Decision: **{hypothesis.get('decision')}**",
        f"- Reason: {hypothesis.get('reason')}",
        "",
        "## Robustness set",
        "",
        f"- n V0={((robust.get('V0') or {}).get('n'))} correct={((robust.get('V0') or {}).get('correctness'))}",
        f"- n V2={((robust.get('V2') or {}).get('n'))} correct={((robust.get('V2') or {}).get('correctness'))}",
        "",
    ]
    return "\n".join(lines)


def decide_hypothesis(official: dict[str, Any], robust: dict[str, Any]) -> dict[str, str]:
    v0 = official.get("V0") or {}
    v2 = official.get("V2") or {}
    r0 = robust.get("V0") or {}
    r2 = robust.get("V2") or {}
    official_gain = (v2.get("correctness") or 0) - (v0.get("correctness") or 0)
    pack0 = v0.get("previous_olympiad_in_packed") or 0
    pack2 = v2.get("previous_olympiad_in_packed") or 0
    ident = v0.get("previous_olympiad_identifiable") or 0
    robust_n = r0.get("n") or 0
    robust_gain = ((r2.get("correctness") or 0) - (r0.get("correctness") or 0)) if robust_n else None
    pack_gain = ident > 0 and pack2 > pack0
    official_better = official_gain >= 5
    if official_better and pack_gain:
        if robust_n and (robust_gain or 0) < 0:
            return {
                "decision": "keep BM25 only",
                "reason": "official set improved but engineering paraphrases regressed; treat as overfit to public wording",
            }
        if robust_n and (robust_gain or 0) >= 0:
            return {
                "decision": "selectively augment with vector",
                "reason": (
                    "V2 improved official temporal correctness and previous-Olympiad packing, "
                    "and paraphrases did not regress. Production routing is still not changed."
                ),
            }
        return {
            "decision": "keep BM25 only",
            "reason": "official temporal improved, but no robustness set was scored; do not change production",
        }
    if pack_gain and not official_better:
        return {
            "decision": "reject the hypothesis as a production change",
            "reason": "packing the previous Olympiad document improved but end-to-end correctness did not",
        }
    return {
        "decision": "keep BM25 only",
        "reason": "official temporal V2 did not beat V0 enough to justify a gated vector path",
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    public = [row for row in load_public_questions(args.questions) if row.get("qtype") == "temporal"]
    public.sort(key=lambda row: str(row.get("qid") or ""))
    for row in public:
        row["set"] = "official_public_temporal"
        row["official_benchmark"] = True
    engineering = load_engineering_set(args.engineering_set)
    if args.limit:
        public = public[: args.limit]
        engineering = engineering[: args.limit]
    selected: list[dict[str, Any]] = []
    if args.which in {"official", "both"}:
        selected.extend(public)
    if args.which in {"robustness", "both"}:
        selected.extend(engineering)
    settings = load_settings()
    llm = load_llm_settings()
    embedding_settings = load_embedding_settings(llm=llm)
    corpus = parse_corpus(args.corpus)
    documents_by_id = corpus.document_by_id()
    chunks = chunk_corpus(corpus.documents, ChunkingConfig())
    packer = ContextPacker()
    generator = build_generator(args.generator)
    config = {
        "benchmark": "phase13_temporal_vector_gate",
        "private_eval_used": False,
        "production_agentic_changed": False,
        "production_graphrag_changed": False,
        "production_bm25_changed": False,
        "skip_ensure": True,
        "skip_upsert": True,
        "generator": generator.__class__.__name__,
        "embedding": embedding_settings.redacted(),
        "graph": settings.redacted(),
        "llm": llm.redacted(),
        "git_revision": git_revision(),
        "generated_at": utc_timestamp(),
        "recall_at_k_definition": RECALL_AT_K_DEFINITION,
        "top_k": args.top_k,
        "n_official": len(public) if args.which in {"official", "both"} else 0,
        "n_engineering": len(engineering) if args.which in {"robustness", "both"} else 0,
    }
    client = TigerGraphClient()
    if not client.connect():
        (output_dir / "report.md").write_text(
            f"FAIL CLOSED. TigerGraph connect failed: {client.error}\n", encoding="utf-8"
        )
        return 1
    try:
        embedder = build_embedder(embedding_settings)
        ingest_chunk_vectors(
            client=client,
            chunks=chunks,
            embedder=embedder,
            cache=None,
            skip_upsert=True,
            skip_ensure=True,
        )
        sparse = RecordingRetriever(TextRetriever(chunks))
        vector_inner = TigerGraphVectorRetriever(
            TigerGraphVectorStore(client, dimension=embedder.dimension),
            chunks,
            embedder,
        )
        vector = RecordingRetriever(vector_inner)
        hybrid = RecordingRetriever(HybridBM25VectorRetriever(sparse, vector))
        adapters = {
            "V0": (RAGAdapter(sparse, packer, generator, method="sparse", top_k=args.top_k), sparse, None),
            "V2": (RAGAdapter(hybrid, packer, generator, method="hybrid", top_k=args.top_k), hybrid, vector),
        }
    except (VectorBackendError, Exception) as exc:
        (output_dir / "report.md").write_text(
            f"FAIL CLOSED. {exc}\n", encoding="utf-8"
        )
        return 1

    all_rows: list[dict[str, Any]] = []
    for record in selected:
        named_year = parse_named_year(str(record.get("question") or ""))
        previous_meta = previous_olympiad_gold(
            list(record.get("gold_doc_ids") or []),
            named_year,
            documents_by_id,
        )
        for variant, (adapter, recorder, vector_recorder) in adapters.items():
            row = capture_row(
                record=record,
                variant=variant,
                adapter=adapter,
                recorder=recorder,
                vector_recorder=vector_recorder,
                previous_meta=previous_meta,
            )
            all_rows.append(row)
            print(f"{variant} {record.get('qid')} {row.get('answer_status')} correct={row.get('correctness')}", flush=True)

    official_rows = [row for row in all_rows if row.get("official_benchmark")]
    robust_rows = [row for row in all_rows if not row.get("official_benchmark")]
    official_summary = summarize_pair(official_rows)
    robustness_summary = summarize_pair(robust_rows)
    hypothesis = decide_hypothesis(official_summary, robustness_summary)
    payload = {
        "run_id": public_run_id(
            {
                "benchmark": "phase13_temporal_vector_gate",
                "qids": [record.get("qid") for record in selected],
                "generator": generator.__class__.__name__,
            }
        ),
        "config": config,
        "official_summary": official_summary,
        "robustness_summary": robustness_summary,
        "hypothesis": hypothesis,
        "n_rows": len(all_rows),
        "rows": all_rows,
    }
    (output_dir / "temporal_gate_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    report = render_report(payload)
    (output_dir / "temporal_gate_report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
