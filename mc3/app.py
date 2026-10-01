#!/usr/bin/env python3
"""MC3 harness entry point (spec §2). Thin client: standard library + sourcebound.protocol only.

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

Every query writes /app/output/<query-id>_output.json with "answer" and "citations", whatever happens.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]          # /app in the image; repo root in development
from sourcebound import protocol  # noqa: E402

OUTPUT_DIR = Path(os.environ.get("SB_OUTPUT_DIR", "/app/output"))
QUERY_BUDGET_S = float(os.environ.get("SB_QUERY_BUDGET_S", "25"))
READY_WAIT_S = float(os.environ.get("SB_READY_WAIT_S", "560"))
REFUSAL = {"answer": "", "citations": [], "confidence": 0.0}


def write_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def clean(resp: dict, corpus: Path) -> dict:
    """Enforce the output schema. An answer without citations, or citations without an answer, is a refusal."""
    answer = resp.get("answer")
    answer = answer.strip() if isinstance(answer, str) else ""
    cites, root = [], corpus.as_posix().rstrip("/") + "/"
    for c in resp.get("citations") or []:
        if isinstance(c, str) and c.strip():
            c = c.strip().replace("\\", "/")
            c = c[len(root):] if c.startswith(root) else c
            while c.startswith("./"):
                c = c[2:]
            if c not in cites:
                cites.append(c)
    if not answer or not cites:
        return dict(REFUSAL)
    conf = resp.get("confidence")
    conf = max(0.0, min(1.0, float(conf))) if isinstance(conf, (int, float)) else 0.5
    return {"answer": answer, "citations": cites, "confidence": conf}


def alive() -> bool:
    try:
        return bool(protocol.request({"op": "status"}, timeout=2.0).get("ok"))
    except Exception:
        return False


def spawn_supervisor() -> None:
    if os.environ.get("SB_NO_SPAWN") == "1":          # tests: never leave a background worker behind
        return
    kw = {"start_new_session": True} if os.name == "posix" else {}
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(HERE), str(HERE.parent), os.environ.get("PYTHONPATH", "")])}
    subprocess.Popen([sys.executable, "-m", "sourcebound.supervisor"], cwd=str(HERE), env=env,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)


def wait_ready(seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if alive():
            return True
        time.sleep(1.0)
    return False


def do_index(corpus: Path) -> int:
    if not alive():
        print("sourcebound: worker not running; starting supervisor", file=sys.stderr)
        spawn_supervisor()
        if not wait_ready(READY_WAIT_S):
            print("sourcebound: worker never became ready", file=sys.stderr)
            return 1
    try:
        resp = protocol.request({"op": "index", "corpus": str(corpus.resolve())}, timeout=READY_WAIT_S)
    except Exception as e:  # noqa: BLE001 - the worker is up; queries will answer or refuse on their own
        resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps({"ok": resp.get("ok"), "stats": resp.get("stats"), "skipped": resp.get("skipped"),
                      "error": resp.get("error")}, ensure_ascii=False), file=sys.stderr)
    return 0                                       # non-zero only when the worker never became ready


def do_query(corpus: Path, qid: str, query: str, t0: float) -> int:
    out = OUTPUT_DIR / f"{qid}_output.json"
    try:
        out.unlink()
    except FileNotFoundError:
        pass
    result = dict(REFUSAL)
    try:
        budget = QUERY_BUDGET_S - (time.monotonic() - t0)
        resp = protocol.request({"op": "query", "corpus": str(corpus), "query_id": qid, "query": query,
                                 "deadline_s": max(1.0, budget - 1.0)}, timeout=max(1.0, budget))
        if resp.get("ok"):
            result = clean(resp, corpus)
            log = os.environ.get("SB_DIAG_LOG")
            if log:                                    # development only: keep the pipeline's trace per question
                with open(log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"qid": qid, "query": query, "result": result, "diag": resp.get("diag"),
                                        "client_s": round(time.monotonic() - t0, 3)}, ensure_ascii=False) + "\n")
        else:
            print(f"sourcebound: worker error: {resp.get('error')}", file=sys.stderr)
    except (ConnectionRefusedError, FileNotFoundError) as e:
        print(f"sourcebound: worker unreachable ({e}); starting it for later questions", file=sys.stderr)
        try:
            spawn_supervisor()
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001 - timeouts and bad replies become a refusal, never a crash
        print(f"sourcebound: {type(e).__name__}: {e}", file=sys.stderr)
    write_atomic(out, result)
    return 0


def main() -> int:
    t0 = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    a = ap.parse_args()
    if a.index is not None:
        return do_index(a.index)
    if a.corpus is None or a.query is None or not a.query_id:
        ap.error("a query needs --corpus, --query-id and --query")
    return do_query(a.corpus, a.query_id, a.query, t0)


if __name__ == "__main__":
    sys.exit(main())
