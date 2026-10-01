"""Official answer normalization and deterministic answer shaping (spec §7, Further Notes A).

Nothing here may know about a particular corpus: no answer tables, no sample-specific rules.
"""
from __future__ import annotations

import re
import unicodedata

_DROP = re.compile(r"[\s\-.·_]")
_PUNCT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                        "—": "-", "−": "-", " ": " ", "…": "..."})


def official(s: str) -> str:
    """The grader's normalization: uppercase, drop all whitespace and the characters - . · _"""
    return _DROP.sub("", (s or "").upper())


def plain(s: str) -> str:
    """Loose form used to match quotes against documents: NFKC, ASCII quotes/dashes, lowercase, single spaces."""
    s = unicodedata.normalize("NFKC", s or "").translate(_PUNCT)
    return " ".join(s.lower().split())


_LABEL = re.compile(r"^(?:the\s+)?(?:answer|value|pin|version|firmware(?:\s+version)?|release|part(?:\s+number|\s+no\.?)?"
                    r"|error\s+code|code)(?:\s*[:#=]\s*|\s+is\s+|\s+)(?=\S)", re.I)
_UNITS = {"c", "°c", "°f", "f", "k", "°", "deg", "degc", "degree", "degrees", "degrees c", "s", "sec", "secs", "second",
          "seconds", "ms", "min", "mins", "minute", "minutes", "h", "hr", "hrs", "hour", "hours", "d", "day", "days",
          "week", "weeks", "w", "kw", "mw", "v", "mv", "a", "ma", "hz", "khz", "mhz", "ghz", "b", "kb", "mb", "gb",
          "tb", "kib", "mib", "gib", "tib", "gb/s", "tb/s", "mb/s", "gbps", "mbps", "%", "mm", "cm", "m", "in", "kg",
          "g", "lb", "lbs", "usd", "eur", "units", "unit", "pcs", "rpm", "nm", "ns", "us", "µs"}
_NUM_UNIT = re.compile(r"^([-+]?\d[\d,]*(?:\.\d+)?)\s*(.*)$")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w]|\.\d)")


def shape(answer: str) -> str:
    """Reduce a model's answer to the value only: no wrapper punctuation, label or unit."""
    a = unicodedata.normalize("NFKC", answer or "").translate(_PUNCT).strip()
    prev = None
    while prev != a:
        prev = a
        a = a.strip("`'\" ").rstrip(".,;:").strip()
    a = _LABEL.sub("", a, count=1).strip()
    a = re.sub(r"^(?:\$|usd\s*|eur\s*|€)\s*(?=\d)", "", a, flags=re.I)
    m = _NUM_UNIT.match(a)
    if m and m.group(2) and m.group(2).strip().lower().rstrip(".") in _UNITS:
        a = m.group(1)
    return " ".join(a.split())


def answer_pattern(answer: str) -> re.Pattern | None:
    """Regex that finds `answer` in text whatever its separators, never inside a longer token (4.3 ≠ 4.3.2)."""
    core = official(answer)
    if not core:
        return None
    body = r"[\s\-.·_]*".join(re.escape(ch) for ch in core)
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9]|[.\-_][A-Za-z0-9])", re.I)


def contains_answer(answer: str, text: str) -> bool:
    p = answer_pattern(answer)
    return bool(p and p.search(unicodedata.normalize("NFKC", text or "")))


_QUARTER = re.compile(r"^Q([1-4])$", re.I)
_JOINED = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+")


def complete_qualifier(answer: str, quote: str) -> str:
    """Expand a truncated identifier to the full token printed in the evidence: Q3 -> Q3 FY27, C2 -> REV-C2.

    Only answers containing a letter are expanded (numbers never are), and only when the answer never occurs
    on its own in the quote."""
    if not answer or not quote or not re.search(r"[A-Za-z]", answer):
        return answer
    m = _QUARTER.match(answer.strip())
    if m:
        q = re.search(rf"\bQ{m.group(1)}\s*(?:FY|CY)\s*'?\d{{2,4}}\b", quote, re.I)
        return " ".join(q.group(0).split()) if q else answer
    alone = re.compile(r"(?<![A-Za-z0-9_\-])" + re.escape(answer) + r"(?![A-Za-z0-9_\-])", re.I)
    if alone.search(quote):
        return answer
    key = official(answer)
    for tok in _JOINED.findall(quote):
        parts = [official(p) for p in re.split(r"[-_]", tok)]
        for i in range(len(parts)):
            for j in range(i + 1, len(parts) + 1):
                if "".join(parts[i:j]) == key and (i > 0 or j < len(parts)):
                    return tok
    return answer


def align_number(answer: str, quote: str) -> str:
    """If the answer is a number written differently from the evidence (84.50 vs 84.5), use the evidence's form."""
    try:
        val = float(answer.replace(",", ""))
    except ValueError:
        return answer
    for lit in _NUMBER.findall(quote or ""):
        try:
            if float(lit.replace(",", "")) == val:
                return lit
        except ValueError:
            continue
    return answer
