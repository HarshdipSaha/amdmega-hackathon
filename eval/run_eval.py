"""Invoke app.py exactly like the harness (fresh process per image); score by normalized exact match.
Appends one JSON line per image (survives quota cut-off) and prints a summary JSON as the last stdout line."""
import argparse, json, os, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roadread.normalize import matches  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--app", default=str(Path(__file__).resolve().parent.parent / "app" / "app.py"))
a = ap.parse_args()
root = Path(a.manifest).parent
out_path = Path(a.out)
outdir = Path(os.environ.get("ROADREAD_OUTPUT_DIR", str(out_path.parent / "app_output")))
diag_path = out_path.with_suffix(".diag.jsonl")
env = {**os.environ, "ROADREAD_OUTPUT_DIR": str(outdir), "ROADREAD_DIAG_LOG": str(diag_path)}
rows = [json.loads(l) for l in Path(a.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]

def last_diag() -> dict:
    try:
        return json.loads(diag_path.read_text(encoding="utf-8").splitlines()[-1])
    except Exception:
        return {}

by = defaultdict(lambda: [0, 0]); lat, over, violations = [], [], 0
with open(out_path, "a", encoding="utf-8") as f:
    for row in rows:
        img = root / row["image"]
        t = time.monotonic()
        try:
            subprocess.run([sys.executable, a.app, "--input-image", str(img)], env=env, timeout=35)
        except subprocess.TimeoutExpired:
            pass
        dt = time.monotonic() - t
        try:
            pred = json.loads((outdir / f"{img.stem}_output.json").read_text(encoding="utf-8"))["text"]
            if not isinstance(pred, str):
                raise TypeError("text is not a string")
        except Exception:
            pred = None                                   # missing/malformed output is a failure, not a crash
        d = last_diag()
        inference = (d.get("diag") or {}).get("elapsed_s")
        overhead = round(dt - inference, 3) if isinstance(inference, (int, float)) else None
        ok = pred is not None and matches(pred, row["gold"])
        violations += int(dt >= 30 or pred is None)
        sl = row.get("slice", "all"); by[sl][0] += int(ok); by[sl][1] += 1
        lat.append(dt)
        if overhead is not None:
            over.append(overhead)
        f.write(json.dumps({**row, "pred": pred, "ok": ok, "seconds": round(dt, 3), "overhead_s": overhead,
                            "diag": d.get("diag")}, ensure_ascii=False) + "\n")
        f.flush()
lat.sort()
print(json.dumps({"correct": sum(v[0] for v in by.values()), "total": len(rows), "by_slice": dict(by),
                  "p50_s": round(lat[len(lat) // 2], 3) if lat else 0, "max_s": round(lat[-1], 3) if lat else 0,
                  "max_overhead_s": max(over) if over else None, "violations": violations}, ensure_ascii=False))
