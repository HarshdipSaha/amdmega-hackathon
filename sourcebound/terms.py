"""Identifier-aware tokenization for lexical search, and identifier extraction for the bridge hop (spec §5)."""
from __future__ import annotations

import re

from .normalize import official

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_.#/][A-Za-z0-9]+)*")
_SPLIT = re.compile(r"[-_.#/]")
STOP = frozenset("""a an and are as at be by does do for from has have how in is it its of on or that the this to was
were what when where which who why with whose into than then there these those under over per via""".split())


def terms(text: str, query: bool = False) -> list[str]:
    """Whole compound tokens plus their parts and a de-punctuated form: ORR-1847 -> orr-1847, orr, 1847, orr1847."""
    out: list[str] = []
    for m in _TOKEN.finditer(text or ""):
        t = m.group(0).lower()
        if query and t in STOP:
            continue
        out.append(t)
        parts = [p for p in _SPLIT.split(t) if p]
        if len(parts) > 1:
            out.extend(p for p in parts if not (query and p in STOP))
            out.append("".join(parts))
    return out


_ID = re.compile(r"(?<![A-Za-z0-9_\-])[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*")


def identifiers(text: str) -> list[str]:
    """Code-like tokens that start with a letter and contain a digit: ORR-1847, E7731, ORR-FAN-2214-B, TQ-40."""
    seen, out = set(), []
    for m in _ID.finditer(text or ""):
        tok = m.group(0).strip("-_")
        if len(tok) < 3 or not re.search(r"\d", tok):
            continue
        k = id_key(tok)
        if k not in seen:
            seen.add(k)
            out.append(tok)
    return out


def id_key(tok: str) -> str:
    return official(tok)
