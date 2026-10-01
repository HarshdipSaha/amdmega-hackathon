"""Evidence pack, reader prompt and reply parsing (spec §6)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .retrieve import Bridge, Ranked

WHOLE_FILE_CHARS = 6000
PACK_CHARS = 36000


@dataclass
class Pack:
    text: str
    images: list[str] = field(default_factory=list)
    fids: dict[str, str] = field(default_factory=dict)          # "F1" -> rel
    bridged_only: set[int] = field(default_factory=set)         # segments that entered only through a bridge
    bridges: list[Bridge] = field(default_factory=list)


def build_pack(index, ranked: list[Ranked], bridges: list[Bridge], exclude: set[str] = frozenset(),
               max_files: int = 8, max_images: int = 3, image_rank: int = 4, budget: int = PACK_CHARS) -> Pack:
    order: list[str] = []
    direct: set[int] = set()
    for r in ranked:
        if r.rel in exclude:
            continue
        if len(order) < max_files:
            order.append(r.rel)
        direct.update(r.segs)
    linked: dict[str, list[Bridge]] = {}
    for b in bridges:
        f = index.segments[b.seg].file
        if f in exclude:
            continue
        linked.setdefault(f, []).append(b)
        if f not in order:
            order.append(f)
    pack = Pack(text="", bridges=[b for bs in linked.values() for b in bs])
    blocks, used = [], 0
    for n, rel in enumerate(order, start=1):
        fid = f"F{n}"
        rec = index.files[rel]
        head = f"[{fid}] path={rel} type={rec.ftype} status={rec.status}"
        if rec.superseded_by:
            head += f" superseded_by={rec.superseded_by}"
        if rec.ftype == "image" and n <= image_rank and len(pack.images) < max_images:
            pack.images.append(rec.extra.get("image_path", rec.path))
            head += f" (image #{len(pack.images)} attached; transcript below)"
        for b in linked.get(rel, []):
            src = next((k for k, v in pack.fids.items() if v == b.from_file), b.from_file or "the question")
            head += f"\n(linked by identifier {b.ident} found in {src})"
        whole = index.file_text(rel)
        if len(whole) <= WHOLE_FILE_CHARS:
            body = whole
        else:
            picks = next((r.segs for r in ranked if r.rel == rel), [])[:4] + [b.seg for b in linked.get(rel, [])]
            keep = sorted({j for i in picks for j in (i - 1, i, i + 1)
                           if 0 <= j < len(index.segments) and index.segments[j].file == rel})
            body = "\n...\n".join(index.segments[j].text for j in keep)
        block = f"{head}\n{body.strip()}\n"
        if used + len(block) > budget and blocks:
            break
        pack.fids[fid] = rel
        blocks.append(block)
        used += len(block)
    for b in pack.bridges:
        if b.seg not in direct:
            pack.bridged_only.add(b.seg)
    pack.text = "\n".join(blocks)
    return pack


INSTRUCTIONS = """You answer a question using ONLY the documents below. They describe fictional products, so anything you know from elsewhere is wrong.

Rules:
- The answer is the value only: a number, code, part number, version, quarter or name. No sentence and no unit.
- Copy the value exactly as printed, including qualifiers that identify it (for example the fiscal year after a quarter, or the prefix of a revision).
- The entity and the attribute must match the question exactly (product, model, component, column). A value for a different product or a different attribute is not an answer.
- A document whose status is WITHDRAWN or SUPERSEDED is not authoritative; use the current document.
- In source code the assigned value is the answer; comments can mention old values.
- When an identifier (ticket, error code, part number) found in one document leads to the value in another document, quote both: the value quote with role "value" and the identifier quote with role "link".
- Check how you located the value. If it sits in a table row or passage keyed by an identifier (ticket, part number, code) that the question does NOT contain, that identifier came from another document in the list: find the document that states the same identifier in the situation the question describes (an incident, a replacement, a log entry) and quote that sentence too, with role "link". Never add a link quote for an identifier the question itself contains.
- Check how you located the value. If it sits in a table row or passage keyed by an identifier (ticket, part number, code) that the question does NOT contain, that identifier came from another document in the list: find the document that states the same identifier in the situation the question describes (an incident, a replacement, a log entry) and quote that sentence too, with role "link". Never add a link quote for an identifier the question itself contains.
- Quotes must be copied character for character from the documents.
- If the documents do not state the answer, reply with status "not_found".
- If a document names an identifier whose details are missing from these documents, reply with status "need_lookup" and list it in "lookup".

Reply with one JSON object and nothing else:
{"status": "answered", "answer": "...", "answer_type": "extracted", "evidence": [{"file": "F1", "quote": "...", "role": "value"}], "lookup": []}
status is one of answered, not_found, need_lookup; answer_type is extracted or derived; role is value or link."""


def reader_prompt(pack: Pack, question: str, feedback: str = "") -> str:
    fb = f"\nA previous attempt was rejected: {feedback}\n" if feedback else ""
    return f"{INSTRUCTIONS}\n\nDOCUMENTS\n{pack.text}\n{fb}\nQUESTION: {question}\nJSON:"


WARRANT = """Question: {question}
Proposed answer: {answer}
Evidence:
{evidence}

Does the evidence state exactly the attribute the question asks for, for exactly the entity the question asks about, with the proposed answer as its value? Reply YES or NO."""


def warrant_prompt(question: str, answer: str, quotes: list[str]) -> str:
    return WARRANT.format(question=question, answer=answer, evidence="\n".join(f"- {q}" for q in quotes))


def _first_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def parse_reply(text: str) -> dict | None:
    """First balanced JSON object in the reply, coerced to the reader schema; None if absent or invalid."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"<\|[^|]*\|>", "", text)
    obj = _first_object(text)
    if obj is None:
        return None
    try:
        d = json.loads(obj)
    except ValueError:
        return None
    if not isinstance(d, dict):
        return None
    status = str(d.get("status", "")).strip().lower()
    if status not in ("answered", "not_found", "need_lookup"):
        status = "answered" if str(d.get("answer", "")).strip() else "not_found"
    ev = []
    for e in d.get("evidence") or []:
        if isinstance(e, dict) and str(e.get("quote", "")).strip():
            role = str(e.get("role", "value")).lower()
            ev.append({"file": str(e.get("file", "")).strip(), "quote": str(e["quote"]),
                       "role": role if role in ("value", "link") else "value"})
    lookup = [str(x) for x in (d.get("lookup") or []) if str(x).strip()][:4]
    atype = str(d.get("answer_type", "extracted")).lower()
    return {"status": status, "answer": str(d.get("answer", "") or "").strip(), "evidence": ev, "lookup": lookup,
            "answer_type": atype if atype in ("extracted", "derived") else "extracted"}
