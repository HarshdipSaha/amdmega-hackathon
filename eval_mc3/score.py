"""Grader-faithful scoring: official answer normalization + exact citation sets (spec Testing Decisions)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sourcebound.normalize import official  # noqa: E402

CORPUS_PREFIXES = ("/app/corpus/",)


def norm_cite(c: str, corpus: str = "") -> str:
    c = c.replace("\\", "/")
    for p in (*CORPUS_PREFIXES, corpus.replace("\\", "/").rstrip("/") + "/" if corpus else None):
        if p and c.startswith(p):
            c = c[len(p):]
    while c.startswith("./"):
        c = c[2:]
    return c


def score_one(pred: dict | None, q: dict, corpus: str = "") -> dict:
    """pred = parsed output JSON (None if missing/malformed); q = a sample-questions.json entry."""
    if not isinstance(pred, dict) or not isinstance(pred.get("answer"), str) or not isinstance(pred.get("citations"), list):
        return {"answer_ok": False, "cite_ok": False, "strict": False, "kind": "malformed"}
    want_a, want_c = q.get("expected_answer", ""), {norm_cite(c) for c in q.get("expected_citations", [])}
    got_c = {norm_cite(c, corpus) for c in pred["citations"] if isinstance(c, str)}
    if want_a == "":
        a_ok = pred["answer"] == ""
    else:
        a_ok = official(pred["answer"]) == official(want_a)      # aliases deliberately ignored (spec G7)
    c_ok = got_c == want_c
    if a_ok and c_ok:
        kind = "ok"
    elif want_a == "" and pred["answer"]:
        kind = "false_answer"
    elif want_a and not pred["answer"]:
        kind = "false_refusal"
    elif not a_ok:
        kind = "wrong_answer"
    elif got_c > want_c:
        kind = "over_citation"
    elif got_c < want_c:
        kind = "under_citation"
    else:
        kind = "wrong_citation"
    return {"answer_ok": a_ok, "cite_ok": c_ok, "strict": a_ok and c_ok, "kind": kind}
