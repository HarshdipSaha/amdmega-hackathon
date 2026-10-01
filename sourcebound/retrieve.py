"""Hybrid retrieval, file ranking and the identifier bridge hop (spec §5)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .normalize import official
from .terms import id_key, identifiers, terms

RRF_K = 60


@dataclass
class Ranked:
    rel: str
    score: float
    segs: list[int] = field(default_factory=list)


@dataclass
class Bridge:
    seg: int
    ident: str
    from_file: str


def rrf(*lists: list[tuple[int, float]], k: int = RRF_K) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for lst in lists:
        for rank, (i, _) in enumerate(lst):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


def search(index, engine, question: str, k: int = 50) -> list[tuple[int, float]]:
    lex = index.bm25.search(terms(question, query=True), k)
    dense: list[tuple[int, float]] = []
    if os.environ.get("SB_DENSE", "1") == "1" and index.vectors is not None and len(index.vectors):
        q = np.asarray(engine.embed([question], kind="query")[0], dtype=np.float32)
        sims = index.vectors @ q
        top = np.argsort(-sims)[:k]
        dense = [(int(i), float(sims[i])) for i in top]
    return rrf(lex, dense)


def rank_files(index, fused: list[tuple[int, float]], demote: float = 0.5) -> list[Ranked]:
    per: dict[str, list[tuple[int, float]]] = {}
    for i, s in fused:
        per.setdefault(index.segments[i].file, []).append((i, s))
    out = []
    for rel, hits in per.items():
        hits.sort(key=lambda x: -x[1])
        score = hits[0][1] + (0.1 * hits[1][1] if len(hits) > 1 else 0.0)
        rec = index.files.get(rel)
        if rec and rec.status != "CURRENT" and rec.superseded_by in index.files:
            score *= demote
        out.append(Ranked(rel, score, [i for i, _ in hits]))
    return sorted(out, key=lambda r: (-r.score, r.rel))


def bridge(index, question: str, ranked: list[Ranked], top_files: int = 6, segs_per_file: int = 3,
           max_files_per_id: int = 3, max_links: int = 6, extra_ids: list[str] | None = None) -> list[Bridge]:
    """Follow identifiers that appear in the top evidence but not in the question into other files."""
    qkey = official(question)
    cands: list[tuple[str, str]] = [(t, "") for t in (extra_ids or [])]
    for r in ranked[:top_files]:
        for i in r.segs[:segs_per_file]:
            cands += [(t, r.rel) for t in identifiers(index.segments[i].text)]
    out: list[Bridge] = []
    done: set[str] = set()
    for tok, src in cands:
        key = id_key(tok)
        if key in done or (src and key in qkey):
            continue
        done.add(key)
        hits = sorted(index.ids.get(key, ()))
        files = {index.segments[i].file for i in hits}
        if not hits or len(files) > max_files_per_id:
            continue
        for i in hits:
            f = index.segments[i].file
            if f == src:
                continue
            if any(b.seg == i for b in out):
                continue
            out.append(Bridge(i, tok, src))
            if len(out) >= max_links:
                return out
    return out
