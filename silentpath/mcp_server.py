"""D1 MCP tools with plain functions that can be tested without MCP transport."""
from __future__ import annotations

from typing import Any

from silentpath.report import findings_only, render_divergence_table
from silentpath.store import RecordStore


def which_path(store_dir: str) -> dict[str, Any]:
    records = RecordStore(store_dir).load()
    return {
        "records": len(records),
        "paths": [{"workload": r.workload_id, "requested": r.path.requested,
                   "observed": r.path.observed, "confidence": r.path.confidence.value,
                   "artifacts": r.artifacts,
                   "device_seconds": r.cost.device_seconds or r.cost.wall_seconds,
                   "samples": r.cost.samples} for r in records],
        "silent_fallbacks": [{"workload": r.workload_id, "requested": r.path.requested,
                              "observed": r.path.observed} for r in records
                             if r.path.is_silent_fallback],
        "unresolved": [{"workload": r.workload_id, "requested": r.path.requested,
                        "observed": r.path.observed,
                        "confidence": r.path.confidence.value} for r in records
                       if r.path.is_unresolved],
    }


def summarise_store(store_dir: str) -> dict[str, Any]:
    records = RecordStore(store_dir).load()
    rows = render_divergence_table(records)
    ratios = [row["cost_ratio"] for row in rows if row["cost_ratio_admissible"]]
    return {
        "records": len(records), "comparisons": len(rows),
        "findings": len(findings_only(rows)),
        "worst_cost_ratio": max(ratios) if ratios else None,
        "inadmissible_speed_rows": sum(not row["cost_ratio_admissible"] for row in rows),
        "silent_fallbacks": sum(record.path.is_silent_fallback for record in records),
        "unresolved_paths": sum(record.path.is_unresolved for record in records),
    }


def build_server():  # pragma: no cover - optional transport wiring
    from mcp.server.fastmcp import FastMCP
    server = FastMCP("silentpath")
    server.tool(name="which_path")(which_path)
    server.tool(name="summarise_store")(summarise_store)
    return server


if __name__ == "__main__":  # pragma: no cover
    build_server().run()
