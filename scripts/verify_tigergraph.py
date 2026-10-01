"""Read-only check of OlympicGraph connectivity, schema, and installed queries.

Usage:
    python -m scripts.verify_tigergraph
    python -m scripts.verify_tigergraph --skip-live

Does not create schema, install queries, upsert vertices, or ingest vectors.
Use ``python -m scripts.ingest_tigergraph`` for those mutating steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingestion.paths import DEFAULT_GRAPH_VERIFY_PATH
from retrieval.graph.verify import run_verify, write_verify_payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only verify OlympicGraph export contract and live TigerGraph"
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="local export/contract only; do not connect to TigerGraph",
    )
    parser.add_argument("--verify-out", type=Path, default=DEFAULT_GRAPH_VERIFY_PATH)
    args = parser.parse_args()
    if args.skip_live:
        from retrieval.graph.verify import run_export, _local_verification

        exported = run_export()
        result = {
            "settings": {"configured": False},
            "export": exported["export"],
            "connection": {"ok": False, "error": "skip-live", "environment": "local"},
            "schema": {"attempted": False, "ok": False},
            "load": {"attempted": False, "ok": False},
            "queries": {"attempted": False, "ok": False, "names": []},
            "read_only": True,
            "blockers": ["skip-live: TigerGraph load not attempted"],
            "local_contract": _local_verification(exported),
        }
    else:
        result = run_verify()
    write_verify_payload(result, args.verify_out)
    public = {
        "connection": result.get("connection"),
        "schema_ok": (result.get("schema") or {}).get("ok"),
        "queries_ok": (result.get("queries") or {}).get("ok"),
        "retrieval_queries_ok": (result.get("retrieval_queries") or {}).get("ok"),
        "read_only": bool(result.get("read_only")),
        "blockers": result.get("blockers"),
        "export_counts": (result.get("export") or {}).get("counts"),
    }
    print(json.dumps(public, indent=2, default=str))
    print(f"Wrote {args.verify_out}")
    if args.skip_live:
        return 0 if (result.get("export") or {}).get("counts") else 2
    schema_ok = bool((result.get("schema") or {}).get("ok"))
    queries_ok = bool((result.get("queries") or {}).get("ok"))
    retrieval_ok = bool((result.get("retrieval_queries") or {}).get("ok"))
    if result.get("blockers") or not (schema_ok and queries_ok and retrieval_ok):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
