"""Okapi BM25 over pre-tokenized documents. Small and dependency-free; rebuilt from segments on load."""
from __future__ import annotations

import math
from collections import Counter, defaultdict


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / self.n) if self.n else 1.0
        self.post: dict[str, dict[int, int]] = defaultdict(dict)
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                self.post[t][i] = c
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def search(self, q: list[str], k: int = 50) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(q):
            p = self.post.get(t)
            if not p:
                continue
            idf = self.idf[t]
            for i, tf in p.items():
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
