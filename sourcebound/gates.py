"""Deterministic gates: grounding, answer-in-evidence, supersession, necessity (spec §7).

The model proposes an answer and quotes; these rules decide the answer and the citation set.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .normalize import align_number, complete_qualifier, contains_answer, official, plain, shape
from .terms import id_key, identifiers

ALLOW_DERIVED = os.environ.get("SB_ALLOW_DERIVED", "0") == "1"


@dataclass
class Evidence:
    rel: str
    quote: str
    role: str


@dataclass
class Verdict:
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    reason: str = ""
    retry: str = ""                        # feedback for one more reader attempt
    exclude: list[str] = field(default_factory=list)
    values: list[Evidence] = field(default_factory=list)
    links: list[Evidence] = field(default_factory=list)


def grounded(quote: str, text: str, min_ratio: float = 0.9) -> bool:
    q, t = plain(quote), plain(text)
    if not q or not t:
        return False
    if q in t:
        return True
    parts = [p.strip() for p in re.split(r"\.\.\.|\|", q) if len(p.strip()) >= 4]
    if len(parts) > 1:
        return all(grounded(p, text, min_ratio) for p in parts)
    if len(q) < 12:
        return False
    m = SequenceMatcher(None, t, q, autojunk=False).find_longest_match(0, len(t), 0, len(q))
    start = max(0, m.a - m.b)
    window = t[start:start + len(q) + 8]
    return SequenceMatcher(None, window, q, autojunk=False).ratio() >= min_ratio


def segment_context(index, ev: Evidence) -> str:
    """Text of the segments of ev.rel that contain ev.quote (the row, page or line window it came from)."""
    q = plain(ev.quote)
    return "\n".join(index.segments[i].text for i in index.by_file.get(ev.rel, []) if q in plain(index.segments[i].text))


def row_key(ev: Evidence, index) -> str:
    """First identifier-like cell value of the table row (segment kind 'row') that contains the quote."""
    q = plain(ev.quote)
    for i in index.by_file.get(ev.rel, []):
        seg = index.segments[i]
        if seg.kind == "row" and q in plain(seg.text):
            cells = [c for c in seg.text.split(" | ") if ": " in c]
            for c in cells:
                ids = identifiers(c.split(": ", 1)[1])
                if ids:
                    return ids[0]
            return ""
    return ""


def resolve(ref: str, pack, index) -> str | None:
    ref = (ref or "").strip().strip("[]")
    if ref in pack.fids:
        return pack.fids[ref]
    cand = ref.replace("\\", "/")
    root = index.root.replace("\\", "/").rstrip("/") + "/"
    for prefix in (root, "/app/corpus/", "./"):
        if cand.startswith(prefix):
            cand = cand[len(prefix):]
    return cand if cand in index.files else None


def judge(index, pack, question: str, reply: dict | None) -> Verdict:
    if reply is None:
        return Verdict(reason="unparseable reply", retry="Reply with one valid JSON object only.")
    if reply["status"] != "answered" or not reply["answer"]:
        return Verdict(reason=reply["status"] or "not_found")
    ev = []
    for e in reply["evidence"]:
        rel = resolve(e["file"], pack, index)
        if rel and grounded(e["quote"], index.file_text(rel)):
            ev.append(Evidence(rel, e["quote"], e["role"]))
    raw = reply["answer"]
    values = [e for e in ev if e.role == "value"] or [e for e in ev if contains_answer(shape(raw), e.quote)]
    links = [e for e in ev if e.role == "link" and e not in values]
    if not values:
        return Verdict(reason="no grounded value quote",
                       retry="Quote the exact text that states the answer, copied character for character, with its file id.")
    answer = shape(raw)
    answer = align_number(complete_qualifier(answer, values[0].quote), values[0].quote)
    derived = reply["answer_type"] == "derived" and ALLOW_DERIVED
    if not derived:
        # The answer must be in the quote, or in the segment (row, page, window) that contains the quote.
        # Never the whole file: a near-miss row elsewhere in the same table would then pass.
        hits = [v for v in values if contains_answer(answer, v.quote)] or \
               [v for v in values if contains_answer(answer, segment_context(index, v))]
        if not hits:
            return Verdict(reason="answer not found in its evidence",
                           retry=f"The answer {answer!r} does not appear in the quoted text. Answer with the value exactly as printed.")
        values = hits + [v for v in values if v not in hits]
    primary = values[0]
    rec = index.files[primary.rel]
    if rec.status != "CURRENT" and rec.superseded_by in index.files:
        return Verdict(reason=f"value came from a {rec.status} document", exclude=[primary.rel],
                       retry=f"{primary.rel} is {rec.status}; use {rec.superseded_by} instead.")
    cites = [primary.rel]
    if derived:
        cites += [v.rel for v in values[1:] if v.rel not in cites][:1]
    qkey = official(question)
    value_ctx = primary.quote + "\n" + segment_context(index, primary)
    ctx_keys = {id_key(t) for t in identifiers(value_ctx)}
    for l in links:
        if l.rel in cites:
            continue
        shared = [t for t in identifiers(l.quote) if id_key(t) in ctx_keys and id_key(t) not in qkey]
        if shared:
            cites.append(l.rel)
    # Row key rule: a table row keyed by an identifier that the question and the answer both lack was found
    # through another file that states that identifier, so that file is necessary (spec §7.5).
    key = row_key(primary, index)
    if key and id_key(key) not in qkey and id_key(key) not in official(answer):
        files = {index.segments[i].file for i in index.ids.get(id_key(key), ())}
        if len(files) <= 3:
            for rel in sorted(files - {primary.rel}):
                if rel not in cites and rel in set(pack.fids.values()):
                    cites.append(rel)
    # a value segment that was reachable only through an identifier bridge implies its source file is a link
    for b in getattr(pack, "bridges", []):
        if (b.seg in pack.bridged_only and index.segments[b.seg].file == primary.rel and b.from_file
                and b.from_file not in cites and id_key(b.ident) in ctx_keys and id_key(b.ident) not in qkey
                and plain(primary.quote) in plain(index.segments[b.seg].text)):
            cites.append(b.from_file)
    return Verdict(answer=answer, citations=cites, reason="ok", values=values, links=links)
