"""Evaluate every corpus of a split, one after another, against the running worker; aggregate one summary.

    python eval_mc3/run_suite.py --split eval_mc3/data/dev --tag dev-q3vl4b-e1 [--work /tmp/sb-suite]
    python eval_mc3/run_suite.py --split eval_mc3/kit --tag kit-q3vl4b-e1

A split is a directory of c<seed>/ folders (questions.json + corpus/ + hazards.json) or the starter kit
(sample-questions.json + mc3-corpus/). Each corpus is COPIED to --work before its hazards are applied
(chmod 000, empty dirs), so the synced sources stay intact. Results: results/<tag>.jsonl (+ .diag.jsonl) and
results/<tag>.summary (last line = aggregate JSON).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT_HAZARDS = {"chmod000": ["vendor/internal_audit.txt"], "empty_dirs": ["archive"]}


def corpora(split: Path) -> list[tuple[str, Path, Path, dict]]:
    if (split / "sample-questions.json").exists():
        return [("kit", split / "sample-questions.json", split / "mc3-corpus", KIT_HAZARDS)]
    out = []
    for d in sorted(p for p in split.iterdir() if p.is_dir() and (p / "questions.json").exists()):
        hz = json.loads((d / "hazards.json").read_text()) if (d / "hazards.json").exists() else {}
        out.append((d.name, d / "questions.json", d / "corpus", hz))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "sb-suite")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    a.results.mkdir(parents=True, exist_ok=True)
    out = a.results / f"{a.tag}.jsonl"
    for p in (out, out.with_suffix(".diag.jsonl")):
        p.unlink(missing_ok=True)
    is_root = hasattr(os, "geteuid") and os.geteuid() == 0 and os.environ.get("SB_KEEP_CHMOD000") != "1"
    if is_root:
        print("note: running as root; mode-000 hazard files are DELETED instead (their handling is covered by the "
              "unit tests and the CI grader-isolation job)", file=sys.stderr)
    parts, total = [], Counter()
    for name, questions, corpus, hz in corpora(a.split):
        work = a.work / name
        if work.exists():
            for p in work.rglob("*"):
                try:
                    p.chmod(0o755 if p.is_dir() else 0o644)
                except OSError:
                    pass
            shutil.rmtree(work)
        shutil.copytree(corpus, work)
        for d in hz.get("empty_dirs", []):
            (work / d).mkdir(parents=True, exist_ok=True)
        for f in hz.get("chmod000", []):
            if (work / f).exists():
                if is_root:
                    (work / f).unlink()           # root reads mode 000, so make the file absent instead (CI covers chmod)
                else:
                    os.chmod(work / f, 0)
        r = subprocess.run([sys.executable, str(ROOT / "eval_mc3" / "run_eval.py"), "--questions", str(questions),
                            "--corpus", str(work), "--out", str(out)], capture_output=True, text=True)
        try:
            s = json.loads(r.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            s = {"error": (r.stderr or r.stdout)[-800:]}
        s["corpus"] = name
        parts.append(s)
        print(json.dumps(s), flush=True)
        for k in ("strict", "total", "violations"):
            total[k] += s.get(k, 0) or 0
    kinds = Counter()
    for s in parts:
        kinds.update(s.get("kinds", {}))
    agg = {"tag": a.tag, "strict": total["strict"], "total": total["total"], "points": 20 * total["strict"],
           "kinds": dict(kinds), "violations": total["violations"],
           "max_s": max((s.get("max_s") or 0 for s in parts), default=0),
           "max_index_s": max((s.get("index_s") or 0 for s in parts), default=0),
           "peak_vram_gib": max((s.get("peak_vram_gib") or 0 for s in parts), default=0),
           "corpora": [{k: s.get(k) for k in ("corpus", "strict", "total", "index_s", "index_stats", "max_s", "error")} for s in parts]}
    (a.results / f"{a.tag}.summary").write_text("\n".join(json.dumps(s) for s in parts) + "\n" + json.dumps(agg) + "\n")
    print(json.dumps(agg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
