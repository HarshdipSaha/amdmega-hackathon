"""One read; optional cheap cross-view check; at most one escalation; choose a complete candidate (spec §6)."""
import os, re, time
from dataclasses import dataclass, field
from .decode import bounded, center_crop, reread_view
from .normalize import normalize
from .prompts import INSTRUCTION, PROMPT_VERSION
from .rules import postprocess

_SPECIAL = re.compile(r"<\|[^|>]*\|>")
_REPLY = re.compile(r"KIND:\s*(plate|sign)\s*\n\s*TEXT:\s*(.*)", re.I | re.S)

def parse_reply(raw: str):
    raw = _SPECIAL.sub("", raw).replace("```", "").strip()
    m = _REPLY.search(raw)
    if not m:
        return "sign", " ".join(raw.split()), False
    return m.group(1).lower(), " ".join(m.group(2).split()), True

@dataclass
class Result:
    text: str
    confidence: float
    diag: dict = field(default_factory=dict)

class Pipeline:
    def __init__(self, engine, max_pixels=1_000_000, reread_pixels=2_500_000, max_new_tokens=128,
                 reread_upper_s=12.0, cross_view=None):
        self.e, self.mp, self.rp, self.mnt, self.upper = engine, max_pixels, reread_pixels, max_new_tokens, reread_upper_s
        self.xv = cross_view if cross_view is not None else os.environ.get("ROADREAD_XVIEW") == "1"

    def _read(self, view, tag):
        r = self.e.read(view, INSTRUCTION, self.mnt)
        kind, text, well_formed = parse_reply(r.text)
        rule = postprocess(text, kind)
        return {"tag": tag, "kind": kind, "text": rule.text, "raw": r.text[:200], "view": list(view.size),
                "suspect": rule.suspect or r.truncated or not well_formed or not rule.text, "seconds": round(r.seconds, 3)}

    def run(self, image, deadline_s: float) -> Result:
        t0, spent = time.monotonic(), 0.0

        def left():                      # fake engines report seconds without sleeping; real ones elapse
            return deadline_s - max(time.monotonic() - t0, spent)

        first = self._read(bounded(image, self.mp), "first"); spent += first["seconds"]
        cands, uncertain = [first], first["suspect"]
        if self.xv and not uncertain and left() > 2 * self.upper:
            x = self._read(bounded(center_crop(image, 0.9), self.mp), "xview"); spent += x["seconds"]
            cands.append(x)
            uncertain = normalize(x["text"]) != normalize(first["text"])
        escalated = None
        if uncertain and left() > self.upper:
            escalated = self._read(reread_view(image, self.mp, self.rp), "reread"); spent += escalated["seconds"]
            cands.append(escalated)
        order = ([escalated] if escalated else []) + [c for c in cands if c is not escalated]
        best = next((c for c in order if not c["suspect"]), first)
        diag = {"prompt": PROMPT_VERSION, "engine": getattr(self.e, "name", "?"), "uncertain": uncertain,
                "reads": cands, "chosen": best["tag"], "elapsed_s": round(time.monotonic() - t0, 3)}
        if hasattr(self.e, "peak_gib"):
            diag["vram_peak_gib"] = self.e.peak_gib()
        return Result(best["text"], 0.9 if not best["suspect"] else 0.3, diag)
