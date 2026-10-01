"""Engine interface and the deterministic FakeEngine used by every GPU-free test (spec §1, Testing seam 2).

An engine provides:
    transcribe(paths) -> list[str]                    text printed in each image
    embed(texts, kind="document"|"query") -> ndarray  L2-normalized float32 rows
    generate(task, prompt, images=(), max_new_tokens=200, question="") -> str
        task "read" returns the reader JSON; task "warrant" returns YES or NO.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

NOT_FOUND = json.dumps({"status": "not_found", "answer": "", "evidence": []})


class FakeEngine:
    name = "fake"

    def __init__(self, replies=None, transcripts=None, warrant="YES", dim: int = 256):
        # replies: [(question substring, reply)], reply = str | dict | list of those (consumed one per call)
        self.replies = [(k, list(v) if isinstance(v, list) else v) for k, v in (replies or [])]
        self.transcripts = transcripts or {}
        self.warrant = warrant
        self.dim = dim
        self.calls: list[tuple[str, str, int]] = []

    @classmethod
    def from_env(cls) -> "FakeEngine":
        def load(var):
            p = os.environ.get(var)
            return json.loads(Path(p).read_text(encoding="utf-8")) if p else None
        no = [x.lower() for x in os.environ.get("SB_FAKE_WARRANT_NO", "").split("|") if x]   # questions the warrant rejects
        return cls(replies=[tuple(x) for x in (load("SB_FAKE_REPLIES") or [])], transcripts=load("SB_FAKE_TRANSCRIPTS"),
                   warrant=(lambda q, prompt: "NO" if any(k in q.lower() for k in no) else "YES"))

    def transcribe(self, paths):
        return [self.transcripts.get(Path(p).name, "") for p in paths]

    def embed(self, texts, kind="document"):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[r, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)

    def generate(self, task, prompt, images=(), max_new_tokens=200, question=""):
        self.calls.append((task, question, len(images)))
        if task == "warrant":
            return self.warrant(question, prompt) if callable(self.warrant) else self.warrant
        for key, reply in self.replies:
            if key.lower() in question.lower():
                if isinstance(reply, list):
                    reply = reply.pop(0) if len(reply) > 1 else reply[0]
                return reply if isinstance(reply, str) else json.dumps(reply)
        return NOT_FOUND

    def warm_up(self):
        return None
