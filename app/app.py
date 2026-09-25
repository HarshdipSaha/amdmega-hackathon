"""Evaluator entry point. Thin client: stdlib + roadread.protocol only. Never imports torch/transformers/PIL."""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # /app (roadread/ is copied beside app.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root in development
from roadread import protocol  # noqa: E402

BUDGET_S = float(os.environ.get("ROADREAD_BUDGET_S", "27"))       # internal deadline under the 30 s hard limit

def write_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)

def main() -> int:
    t0 = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", required=True)
    a = ap.parse_args()
    img = Path(a.input_image)
    out = Path(os.environ.get("ROADREAD_OUTPUT_DIR", "/app/output")) / f"{img.stem}_output.json"
    if out.exists():
        out.unlink()                                               # never leave a stale answer
    result, diag = {"text": ""}, {}
    try:
        resp = protocol.request(str(img.resolve()), deadline_s=max(1.0, BUDGET_S - (time.monotonic() - t0)))
        result["text"] = str(resp.get("text", ""))
        if isinstance(resp.get("confidence"), (int, float)):
            result["confidence"] = max(0.0, min(1.0, float(resp["confidence"])))
        diag = resp.get("diag") or {}
    except Exception as e:                                         # worker absent/slow: fast valid failure
        diag = {"error": f"{type(e).__name__}: {e}"}
        print(f"roadread: worker error: {e}", file=sys.stderr)
    write_atomic(out, result)
    log = os.environ.get("ROADREAD_DIAG_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"image": img.name, "client_wall_s": round(time.monotonic() - t0, 3), "diag": diag},
                               ensure_ascii=False) + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
