"""Write deterministic TigerGraph export files from the Phase 1/3 corpus.

Usage:
    python -m scripts.export_graph
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingestion.paths import DEFAULT_CORPUS_PATH, DEFAULT_GRAPH_EXPORT_DIR
from retrieval.graph.verify import run_export


def main() -> int:
    parser = argparse.ArgumentParser(description="Export OlympicGraph vertices and edges")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_GRAPH_EXPORT_DIR)
    args = parser.parse_args()
    result = run_export(args.corpus, args.out)
    manifest = result["export"]
    print(json.dumps({"counts": manifest["counts"], "elapsed_ms": manifest["elapsed_ms"], "directory": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
