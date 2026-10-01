"""Create schema, upsert export files, and install GSQL queries.

Usage:
    python -m scripts.ingest_tigergraph
    python -m scripts.ingest_tigergraph --probe-write
    python -m scripts.ingest_tigergraph --bind
    python -m scripts.ingest_tigergraph --reset
    python -m scripts.ingest_tigergraph --export-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingestion.paths import DEFAULT_GRAPH_VERIFY_PATH
from retrieval.graph.verify import run_bind, run_export, run_ingest, run_probe_write, write_verify_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest OlympicGraph into TigerGraph")
    parser.add_argument("--reset", action="store_true", help="drop and recreate the graph before upsert")
    parser.add_argument("--bind", action="store_true", help="attach existing GLOBAL types to OlympicGraph without loading data")
    parser.add_argument("--probe-write", action="store_true", help="upsert, read, and delete one synthetic Document vertex")
    parser.add_argument("--export-only", action="store_true", help="write local files without connecting")
    parser.add_argument("--verify-out", type=Path, default=DEFAULT_GRAPH_VERIFY_PATH)
    args = parser.parse_args()
    exclusive = [flag for flag, enabled in (("reset", args.reset), ("bind", args.bind), ("probe-write", args.probe_write), ("export-only", args.export_only)) if enabled]
    if len(exclusive) > 1:
        parser.error("these flags cannot be combined: " + ", ".join("--" + name for name in exclusive))
    if args.export_only:
        result = {"export": run_export()["export"], "connection": {"ok": False, "error": "export-only", "environment": "local"}}
        result["blockers"] = ["export-only: TigerGraph load not attempted"]
    elif args.bind:
        result = run_bind()
    elif args.probe_write:
        result = run_probe_write()
    else:
        result = run_ingest(reset=args.reset)
    write_verify_payload(result, args.verify_out)
    public = {
        "connection": result.get("connection"),
        "schema_ok": (result.get("schema") or {}).get("ok"),
        "bind": result.get("bind"),
        "probe": result.get("probe"),
        "load_ok": (result.get("load") or {}).get("ok"),
        "queries": (result.get("queries") or {}).get("names") or (result.get("queries") or {}).get("installed"),
        "counts": result.get("counts"),
        "blockers": result.get("blockers"),
        "export_counts": (result.get("export") or {}).get("counts"),
        "local_mismatches": ((result.get("local_contract") or {}).get("equivalence") or {}).get("local_mismatches"),
        "tigergraph_mismatches": ((result.get("equivalence") or {}).get("tigergraph_mismatches")),
    }
    print(json.dumps(public, indent=2, default=str))
    print(f"Wrote {args.verify_out}")
    if args.bind:
        if result.get("blockers") or not ((result.get("bind") or {}).get("ok")):
            return 2
        return 0
    if args.probe_write:
        if result.get("blockers") or not ((result.get("probe") or {}).get("ok")):
            return 2
        return 0
    if args.export_only:
        return 0 if (result.get("export") or {}).get("counts") else 2
    load_ok = bool((result.get("load") or {}).get("ok"))
    schema_ok = bool((result.get("schema") or {}).get("ok"))
    queries_ok = bool((result.get("queries") or {}).get("ok"))
    verify_ok = bool((result.get("verification") or {}).get("ok"))
    if result.get("blockers") or not (load_ok and schema_ok and queries_ok and verify_ok):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
