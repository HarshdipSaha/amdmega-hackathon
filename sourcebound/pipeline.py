"""One question, end to end, under a deadline: retrieve -> bridge -> read -> gates -> warrant (spec §5-7)."""
from __future__ import annotations

import os
import time

from .gates import judge
from .reader import build_pack, parse_reply, reader_prompt, warrant_prompt
from .retrieve import bridge, rank_files, search

MIN_CALL_S = float(os.environ.get("SB_MIN_CALL_S", "6"))
WARRANT = os.environ.get("SB_WARRANT", "1") == "1"
MAX_NEW = int(os.environ.get("SB_MAX_NEW_TOKENS", "200"))
BRIDGE = os.environ.get("SB_BRIDGE", "1") == "1"


def refusal(diag: dict, reason: str) -> dict:
    diag["reason"] = reason
    return {"answer": "", "citations": [], "confidence": 0.0, "diag": diag}


def answer(index, engine, question: str, deadline_s: float = 24.0) -> dict:
    t0 = time.monotonic()
    end = t0 + deadline_s
    left = lambda: end - time.monotonic()  # noqa: E731
    diag: dict = {"attempts": []}
    fused = search(index, engine, question)
    ranked = rank_files(index, fused)
    bridges = bridge(index, question, ranked) if BRIDGE else []
    diag["ranked"] = [r.rel for r in ranked[:8]]
    diag["bridges"] = [(b.ident, index.segments[b.seg].file, b.from_file) for b in bridges]
    exclude: set[str] = set()
    feedback, looked_up = "", False
    for attempt in range(3):
        if left() < MIN_CALL_S:
            diag["attempts"].append("skipped: deadline")
            break
        pack = build_pack(index, ranked, bridges, exclude)
        raw = engine.generate("read", reader_prompt(pack, question, feedback), pack.images, MAX_NEW, question=question)
        reply = parse_reply(raw)
        diag["attempts"].append({"fids": pack.fids, "raw": raw[:600]})
        if reply and reply["status"] == "need_lookup" and reply["lookup"] and not looked_up:
            looked_up = True
            bridges = bridges + bridge(index, question, [], extra_ids=reply["lookup"])
            continue
        v = judge(index, pack, question, reply)
        diag["attempts"][-1]["verdict"] = v.reason
        if v.answer:
            if WARRANT and left() > 2.0:
                quotes = [e.quote for e in v.values[:2] + v.links[:2]]
                ok = engine.generate("warrant", warrant_prompt(question, v.answer, quotes), (), 3, question=question)
                diag["warrant"] = ok.strip()[:20]
                if ok.strip().upper().startswith("NO"):
                    feedback = (f"The answer {v.answer!r} does not state the requested attribute of the requested "
                                "entity. Look again; if no document states it, reply not_found.")
                    continue
            conf = 0.9 if len(v.citations) == 1 else 0.8
            diag["seconds"] = round(time.monotonic() - t0, 3)
            return {"answer": v.answer, "citations": v.citations, "confidence": conf, "diag": diag}
        if v.reason in ("not_found", "need_lookup"):
            return refusal(diag, v.reason)
        exclude |= set(v.exclude)
        feedback = v.retry
        if not feedback:
            break
    diag["seconds"] = round(time.monotonic() - t0, 3)
    return refusal(diag, "gates failed")
