"""Run the MC3 contract exactly like the harness: one --index, then one fresh process per question.

    python eval_mc3/run_eval.py --questions <questions.json> --corpus <dir> --out results/<tag>.jsonl [--skip-index]

Appends one JSON line per question (survives a quota cut-off) and prints a summary JSON as the last stdout line.
Strict score = answer AND exact citation set right, 20 points each.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_mc3.score import score_one  # noqa: E402
from eval_mc3.vram import VramSampler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--app", default=str(ROOT / "mc3" / "app.py"))
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--timeout", type=float, default=35.0)
    a = ap.parse_args()
    qs = json.loads(Path(a.questions).read_text(encoding="utf-8"))["queries"]
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outdir = Path(os.environ.get("SB_OUTPUT_DIR", str(out_path.parent / "app_output")))
    env = {**os.environ, "SB_OUTPUT_DIR": str(outdir), "SB_DIAG_LOG": str(out_path.with_suffix(".diag.jsonl"))}
    vram = VramSampler()
    vram.start()
    index_s, index_stats = None, None
    if not a.skip_index:
        t = time.monotonic()
        r = subprocess.run([sys.executable, a.app, "--index", a.corpus], env=env, capture_output=True, text=True, timeout=660)
        index_s = round(time.monotonic() - t, 2)
        print(f"index: rc={r.returncode} {index_s}s {r.stderr.strip()[-400:]}", file=sys.stderr)
        for line in reversed(r.stderr.strip().splitlines()):
            try:
                index_stats = json.loads(line).get("stats")
                break
            except (ValueError, AttributeError):
                continue
    by = defaultdict(lambda: [0, 0])
    kinds: Counter = Counter()
    lat, violations, total_t = [], 0, time.monotonic()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, q in enumerate(qs, start=1):
            qid = f"query_{i:02d}"
            t = time.monotonic()
            try:
                subprocess.run([sys.executable, a.app, "--corpus", a.corpus, "--query-id", qid, "--query", q["query"]],
                               env=env, timeout=a.timeout, capture_output=True, text=True)
            except subprocess.TimeoutExpired:
                pass
            dt = time.monotonic() - t
            try:
                pred = json.loads((outdir / f"{qid}_output.json").read_text(encoding="utf-8"))
            except Exception:
                pred = None
            s = score_one(pred, q, a.corpus)
            violations += int(dt >= 30 or s["kind"] == "malformed")
            cat = q.get("category", "all").split(",")[0].strip()
            by[cat][0] += int(s["strict"])
            by[cat][1] += 1
            kinds[s["kind"]] += 1
            lat.append(dt)
            f.write(json.dumps({"qid": qid, "query": q["query"], "expected": q.get("expected_answer"),
                                "expected_citations": q.get("expected_citations"), "pred": pred, **s,
                                "seconds": round(dt, 3)}, ensure_ascii=False) + "\n")
            f.flush()
    vram.stop()
    lat.sort()
    strict = sum(v[0] for v in by.values())
    print(json.dumps({"strict": strict, "total": len(qs), "points": 20 * strict, "max_points": 20 * len(qs),
                      "by_category": dict(by), "kinds": dict(kinds), "index_s": index_s, "index_stats": index_stats,
                      "run_s": round(time.monotonic() - total_t, 2),
                      "p50_s": round(lat[len(lat) // 2], 3) if lat else 0, "max_s": round(lat[-1], 3) if lat else 0,
                      "peak_vram_gib": vram.peak_gib, "violations": violations}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
