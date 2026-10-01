"""The highest seam: the harness contract. `mc3/app.py` runs as separate processes against a real worker
(FakeEngine), exactly as the grader execs it, and the JSON files are scored with the grader's rules."""
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from eval_mc3.score import score_one
from sourcebound import protocol
from sourcebound.engine import FakeEngine
from sourcebound.worker import Worker, serve
from tests.fixtures import kit_fake as kf

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "mc3" / "app.py"


def free_addr(tmp_path) -> str:
    if hasattr(socket, "AF_UNIX"):
        return f"unix:{tmp_path / 'sb.sock'}"
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"tcp:127.0.0.1:{port}"


@pytest.fixture
def running_worker(tmp_path):
    addr = free_addr(tmp_path)
    eng = FakeEngine(replies=kf.REPLIES, transcripts=kf.TRANSCRIPTS, warrant=kf.warrant)
    w = Worker(eng, tmp_path / "index", parse_workers=2)
    stop = threading.Event()
    t = threading.Thread(target=serve, args=(w, addr, None, stop), daemon=True)
    t.start()
    for _ in range(50):
        try:
            protocol.request({"op": "status"}, timeout=1, addr=addr)
            break
        except OSError:
            time.sleep(0.1)
    yield addr, w, eng
    stop.set()
    t.join(5)


def app(args, addr, out_dir, timeout=120):
    env = {**os.environ, "SB_ADDR": addr, "SB_OUTPUT_DIR": str(out_dir), "PYTHONPATH": str(ROOT),
           "SB_READY_WAIT_S": "5", "SB_NO_SPAWN": "1"}
    t = time.monotonic()
    r = subprocess.run([sys.executable, str(APP), *args], env=env, capture_output=True, text=True, timeout=timeout)
    return r, time.monotonic() - t


def test_sample_kit_scores_200_through_the_contract(running_worker, kit_corpus, kit_questions, tmp_path):
    addr, w, _ = running_worker
    out = tmp_path / "out"
    r, _ = app(["--index", str(kit_corpus)], addr, out)
    assert r.returncode == 0, r.stderr
    assert "telemetry_capture.dat" in r.stderr                      # skip ledger is reported
    results = []
    for i, q in enumerate(kit_questions, start=1):
        qid = f"query_{i:02d}"
        r, dt = app(["--corpus", str(kit_corpus), "--query-id", qid, "--query", q["query"]], addr, out)
        assert r.returncode == 0, r.stderr
        pred = json.loads((out / f"{qid}_output.json").read_text(encoding="utf-8"))
        assert set(pred) == {"answer", "citations", "confidence"}
        results.append((q["n"], pred, score_one(pred, q, str(kit_corpus))))
    failed = [(n, p, s["kind"]) for n, p, s in results if not s["strict"]]
    assert failed == []


def test_unreadable_file_does_not_cost_other_files(running_worker, kit_corpus, tmp_path):
    addr, w, _ = running_worker
    real_open = open

    def deny(path, mode="r", *a, **k):
        if os.path.basename(path) == "asset_label.jpg":
            raise PermissionError(13, "denied", path)
        return real_open(path, mode, *a, **k)
    w.opener = deny
    r, _ = app(["--index", str(kit_corpus)], addr, tmp_path / "out")
    assert r.returncode == 0
    assert "support/asset_label.jpg" not in w.index.files and "support/rma_parts.xlsx" in w.index.files


def test_worker_absent_writes_valid_refusal_fast(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "query_07_output.json").write_text('{"answer": "STALE", "citations": ["x"]}')
    env_addr = free_addr(tmp_path)
    r, dt = app(["--corpus", str(tmp_path), "--query-id", "query_07", "--query", "anything?"], env_addr, out)
    assert r.returncode == 0 and dt < 15
    assert json.loads((out / "query_07_output.json").read_text()) == {"answer": "", "citations": [], "confidence": 0.0}


def test_query_id_is_used_verbatim(running_worker, kit_corpus, tmp_path):
    addr, _, _ = running_worker
    app(["--index", str(kit_corpus)], addr, tmp_path / "out")
    app(["--corpus", str(kit_corpus), "--query-id", "Q-weird_9", "--query", "What error code is logged when the thermal throttle engages?"],
        addr, tmp_path / "out")
    assert json.loads((tmp_path / "out" / "Q-weird_9_output.json").read_text())["answer"] == "E7731"


def test_bad_request_does_not_kill_worker(running_worker):
    addr, _, _ = running_worker
    with protocol.connect(addr) as s:
        s.sendall(b"not json\n")
        assert protocol.recv_line(s)["ok"] is False
    assert protocol.request({"op": "status"}, timeout=2, addr=addr)["ok"] is True


def test_client_imports_no_heavy_modules():
    code = ("import sys, runpy; sys.argv=['app.py','--help']\n"
            "try:\n    runpy.run_path(r'%s', run_name='not_main')\nexcept SystemExit:\n    pass\n"
            "print(sorted(m for m in ('torch','transformers','PIL','numpy','fitz','openpyxl') if m in sys.modules))") % APP
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert out.stdout.strip() == "[]", out.stderr


def test_only_the_first_index_is_charged_to_the_startup_budget(tmp_path, monkeypatch):
    import sourcebound.worker as wk
    seen = []

    def fake_build(corpus, eng, d, deadline, **k):
        seen.append(deadline - time.monotonic())
        return type("I", (), {"stats": {}, "ledger": []})()
    monkeypatch.setattr(wk, "build_index", fake_build)
    w = Worker(FakeEngine(), tmp_path / "idx", started=time.time() - 1000)     # the container started long ago
    w.handle({"op": "index", "corpus": str(tmp_path)})
    w.handle({"op": "index", "corpus": str(tmp_path)})
    assert seen[0] == pytest.approx(wk.MIN_INDEX_S, abs=2)                    # first: what is left of startup
    assert seen[1] == pytest.approx(wk.INDEX_BUDGET_S, abs=2)                 # later (eval suites): a full budget
