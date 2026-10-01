"""The generator must produce self-consistent corpora, and the pipeline (with a FakeEngine replaying the oracle
evidence) must score every generated question through the gates. This exercises supersession, qualifiers, units,
second sheets, near-miss rows, both chain types and the necessity rule on fresh names and values."""
import json
import os
import time

import pytest

from eval_mc3.generate_corpus import generate
from eval_mc3.score import score_one
from sourcebound import pipeline
from sourcebound.engine import FakeEngine
from sourcebound.index import build_index


@pytest.fixture(scope="module", params=[11, 12])
def gen(request, tmp_path_factory):
    out = tmp_path_factory.mktemp(f"g{request.param}")
    generate(out, request.param)
    return out


def oracle_replies(qs):
    reps = []
    for q in qs:
        if q["expected_answer"]:
            reps.append((q["query"], {"status": "answered", "answer": q["expected_answer"], "answer_type": "extracted",
                                      "evidence": q["oracle"], "lookup": []}))
        else:
            reps.append((q["query"], {"status": "not_found", "answer": "", "evidence": []}))
    return reps


def test_generated_corpus_is_self_consistent(gen):
    qs = json.loads((gen / "questions.json").read_text())["queries"]
    assert len(qs) == 16
    for q in qs:
        for c in q["expected_citations"]:
            assert (gen / "corpus" / c).exists(), c
    assert (gen / "corpus" / "archive").is_dir()


def test_suite_runner_end_to_end_with_oracle_worker(gen, tmp_path):
    """run_suite -> run_eval -> app.py processes -> worker (FakeEngine replaying the oracle) -> aggregate summary."""
    import socket
    import subprocess
    import sys
    import threading
    from pathlib import Path

    from sourcebound.worker import Worker, serve

    qs = json.loads((gen / "questions.json").read_text())["queries"]
    eng = FakeEngine(replies=oracle_replies(qs), transcripts=json.loads((gen / "transcripts.json").read_text()))
    if hasattr(socket, "AF_UNIX"):
        addr = f"unix:{tmp_path / 's.sock'}"
    else:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        addr = f"tcp:127.0.0.1:{s.getsockname()[1]}"
        s.close()
    stop = threading.Event()
    t = threading.Thread(target=serve, args=(Worker(eng, tmp_path / "idx", parse_workers=2), addr, None, stop), daemon=True)
    t.start()
    root = Path(__file__).resolve().parent.parent
    split = tmp_path / "split"
    split.mkdir()
    import shutil
    shutil.copytree(gen, split / "c1")
    env = {**os.environ, "SB_ADDR": addr, "SB_NO_SPAWN": "1", "PYTHONPATH": str(root),
           "SB_OUTPUT_DIR": str(tmp_path / "out")}
    r = subprocess.run([sys.executable, str(root / "eval_mc3" / "run_suite.py"), "--split", str(split), "--tag", "t",
                        "--work", str(tmp_path / "work"), "--results", str(tmp_path / "res")],
                       env=env, capture_output=True, text=True, timeout=600)
    stop.set()
    agg = json.loads(r.stdout.strip().splitlines()[-1])
    assert (agg["strict"], agg["total"], agg["violations"]) == (16, 16, 0), r.stdout + r.stderr
    assert (tmp_path / "res" / "t.diag.jsonl").exists()


def test_oracle_evidence_passes_the_gates(gen, tmp_path):
    qs = json.loads((gen / "questions.json").read_text())["queries"]
    transcripts = json.loads((gen / "transcripts.json").read_text())
    (gen / "corpus" / "vendor" / "internal_audit.txt").unlink(missing_ok=True)   # stands in for chmod 000 on Windows
    idx = build_index(gen / "corpus", FakeEngine(transcripts=transcripts), tmp_path / "idx", time.monotonic() + 300,
                      workers=2, log=lambda *a: None)
    eng = FakeEngine(replies=oracle_replies(qs))
    bad = []
    for q in qs:
        out = pipeline.answer(idx, eng, q["query"], 20)
        s = score_one(out, q)
        if not s["strict"]:
            bad.append((q["n"], q["category"], out["answer"], out["citations"], s["kind"], out["diag"].get("reason")))
    assert bad == []


def test_row_key_rule_cites_chain_without_link_quotes(gen, tmp_path):
    """Value-only replies (what the 4B reader produces): chains must still cite log + table, singles stay single."""
    qs = json.loads((gen / "questions.json").read_text())["queries"]
    for q in qs:
        q["oracle"] = [e for e in q["oracle"] if e["role"] == "value"]
    transcripts = json.loads((gen / "transcripts.json").read_text())
    (gen / "corpus" / "vendor" / "internal_audit.txt").unlink(missing_ok=True)
    idx = build_index(gen / "corpus", FakeEngine(transcripts=transcripts), tmp_path / "idx2", time.monotonic() + 300,
                      workers=2, log=lambda *a: None)
    eng = FakeEngine(replies=oracle_replies(qs))
    bad = [(q["n"], pipeline.answer(idx, eng, q["query"], 20)["citations"]) for q in qs
           if not score_one(pipeline.answer(idx, eng, q["query"], 20), q)["strict"]]
    assert bad == []
