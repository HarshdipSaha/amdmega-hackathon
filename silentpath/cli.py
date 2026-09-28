"""Canonical command line interface for running and reporting matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from silentpath.budget import BudgetExceeded, GpuBudget
from silentpath.matrix import expand, load_matrix
from silentpath.report import findings_only, render_divergence_table
from silentpath.runner import billable_seconds, run_matrix
from silentpath.store import RecordStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="silentpath")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="execute a configuration matrix")
    run.add_argument("--matrix", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--cap-hours", required=True, type=float)
    run.add_argument("--estimate-seconds", type=float, default=600.0)
    run.add_argument("--producer", choices=("fake", "sdpa"), default="sdpa")
    run.add_argument("--fake", action="store_true", help="explicitly select the fake producer")
    report = sub.add_parser("report", help="render comparisons from a record store")
    report.add_argument("--out", required=True, type=Path)
    report.add_argument("--findings-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = RecordStore(args.out)
    if args.command == "run":
        cells = expand(load_matrix(args.matrix.read_text(encoding="utf-8")))
        producer_kind = "fake" if args.fake else args.producer
        if producer_kind == "fake":
            from silentpath.producers.fake import FakeProducer
            producer = FakeProducer()
        else:
            from silentpath.producers.sdpa import SdpaProducer
            producer = SdpaProducer(artifact_dir=args.out / "artifacts")
        budget = GpuBudget(cap_hours=args.cap_hours)
        budget.seed(sum(billable_seconds(record) for record in store.load()))
        try:
            count = run_matrix(cells, producer, store, budget, args.estimate_seconds)
        except BudgetExceeded as exc:
            print(f"[silentpath] stopped: {exc}")
            return 2
        print(f"[silentpath] executed {count} new cell(s); {len(store.load())} total")
        return 0
    rows = render_divergence_table(store.load())
    if args.findings_only:
        rows = findings_only(rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
