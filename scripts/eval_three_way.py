"""Run the Phase 8 public three-way benchmark.

Usage:
    python -m scripts.eval_three_way

Read-only. Does not reset, ingest, bind, or install graph queries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from answering.factory import build_generator
from answering.packer import ContextPacker
from config.settings import load_settings
from evaluation.harness import AgenticGraphRAGAdapter, GraphRAGAdapter, RAGAdapter
from evaluation.retrieval_gold import build_parser
from evaluation.three_way import (
    DEFAULT_OUTPUT_DIR,
    evaluate_three_way,
    load_public_questions,
    make_three_way_harness,
    write_benchmark_artifacts,
)
from ingestion.chunker import ChunkingConfig, chunk_corpus
from ingestion.loader import parse_corpus
from ingestion.paths import DEFAULT_CORPUS_PATH, DEFAULT_PUBLIC_QUESTIONS_PATH
from retrieval.agentic.pipeline import AgenticGraphRAGPipeline
from retrieval.graph.client import TigerGraphClient
from retrieval.graphrag.parser import GraphRAGQuestionParser
from retrieval.graphrag.pipeline import FixedGraphRAGPipeline
from retrieval.graphrag.retriever import GraphRetriever
from retrieval.rag.retriever import TextRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 8 public three-way benchmark")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--questions", type=Path, default=DEFAULT_PUBLIC_QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rag-method", default="sparse")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="optional cap for smoke runs; 0 means all")
    parser.add_argument(
        "--generator",
        default="deterministic",
        choices=("deterministic", "semantic"),
        help="shared answering implementation used by RAG, GraphRAG, and Agentic",
    )
    args = parser.parse_args()

    records = load_public_questions(args.questions)
    if args.limit:
        records = records[: args.limit]

    settings = load_settings()
    if not settings.configured:
        raise SystemExit("TigerGraph is not configured; live GraphRAG/Agentic benchmark cannot run")

    client = TigerGraphClient()
    if not client.connect():
        raise SystemExit(client.error or "TigerGraph is not reachable")
    if not client.verify_queries().get("ok") or not client.verify_retrieval_queries().get("ok"):
        raise SystemExit("required installed queries are unavailable")

    corpus = parse_corpus(args.corpus)
    chunks = chunk_corpus(corpus.documents, ChunkingConfig())
    _index, question_parser = build_parser(corpus)
    graph_parser = GraphRAGQuestionParser(question_parser)
    graph_retriever = GraphRetriever.from_client(client)
    packer = ContextPacker()
    generator = build_generator(args.generator)

    harness = make_three_way_harness(
        rag=RAGAdapter(
            TextRetriever(chunks),
            packer,
            generator,
            method=args.rag_method,
            top_k=args.top_k,
        ),
        graphrag=GraphRAGAdapter(
            FixedGraphRAGPipeline(graph_parser, graph_retriever, packer, generator)
        ),
        agentic=AgenticGraphRAGAdapter(
            AgenticGraphRAGPipeline(graph_parser, graph_retriever, packer, generator)
        ),
    )
    payload = evaluate_three_way(
        records,
        harness,
        config={
            "questions_name": args.questions.name,
            "corpus_name": args.corpus.name,
            "rag_method": args.rag_method,
            "top_k": args.top_k,
            "generator": generator.__class__.__name__,
            "graph": settings.redacted(),
            "read_only": True,
        },
    )
    paths = write_benchmark_artifacts(payload, args.output)
    report = paths["report"].read_text(encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nWrote {paths['json']}")
    print(f"Wrote {paths['csv']}")
    print(f"Wrote {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
