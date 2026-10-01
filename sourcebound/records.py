"""Segments, file records, and the supersession model: status flags and document families (spec §4)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath


@dataclass
class Segment:
    file: str      # corpus-relative POSIX path
    locator: str   # "page 3", "Parts!4", "lines 1-24", "image", "row 7"
    kind: str      # prose | row | code | log | transcript
    text: str


@dataclass
class FileRecord:
    rel: str
    path: str
    ftype: str     # pdf | docx | xlsx | csv | text | code | image
    size: int = 0
    status: str = "CURRENT"   # CURRENT | WITHDRAWN | SUPERSEDED
    family: str = ""
    revision: str = ""
    superseded_by: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


_NAME_WITHDRAWN = re.compile(r"(?:^|[^a-z])(withdrawn|superseded|obsolete|deprecated)(?:[^a-z]|$)", re.I)
_HEAD_WITHDRAWN = re.compile(r"\b(withdrawn|superseded by|obsolete|deprecated|do not use|do not design)\b", re.I)
_REV = re.compile(r"(?:^|[_\-\s.])(r|rev|revision|v|ver|version)[_\-\s]?([0-9]{1,3}|[a-z]\d?)(?=$|[_\-\s.])", re.I)
_STATUS_WORDS = re.compile(r"(?:^|[_\-\s])(withdrawn|superseded|obsolete|deprecated|draft|final|current|latest)(?=$|[_\-\s])", re.I)


# "supersedes revision 1" and "Revision 1 is withdrawn" describe ANOTHER document, not this one
_OTHER_DOC = re.compile(r"\bsupersedes\b|\b(?:revision|rev\.?|r|version|v)\s*[0-9a-z]{1,3}\s+(?:is|was|has been)\s+"
                        r"(?:withdrawn|superseded|obsolete|deprecated)", re.I)


def head_lines(text: str, n: int = 3) -> str:
    return "\n".join([ln for ln in (text or "").splitlines() if ln.strip()][:n])


def withdrawn_heading(text: str) -> bool:
    return any(_HEAD_WITHDRAWN.search(ln) and not _OTHER_DOC.search(ln) for ln in head_lines(text).splitlines())


def revision_of(name: str) -> tuple[str, tuple]:
    """('2', (0, 2)) for tq40_datasheet_r2; ('B', (1, 66, 0)) for spec_revB; ('', ()) when absent."""
    m = None
    for m in _REV.finditer(PurePosixPath(name).stem):
        pass
    if not m:
        return "", ()
    tok = m.group(2).upper()
    if tok.isdigit():
        return tok, (0, int(tok))
    return tok, (1, ord(tok[0]), int(tok[1:] or 0))


def family_of(rel: str) -> str:
    p = PurePosixPath(rel)
    stem = _REV.sub("", p.stem)
    stem = _STATUS_WORDS.sub("", stem)
    stem = re.sub(r"[_\-\s]+", "_", stem).strip("_").lower()
    return f"{p.parent.as_posix()}/{stem}{p.suffix.lower()}"


def assign_status(records: dict[str, FileRecord], heads: dict[str, str]) -> None:
    """WITHDRAWN from the file name or the first three lines; SUPERSEDED for an older revision in a family."""
    for rel, rec in records.items():
        name_hit = _NAME_WITHDRAWN.search(PurePosixPath(rel).stem.replace("_", " "))
        if name_hit or withdrawn_heading(heads.get(rel, "")):
            rec.status = "WITHDRAWN"
        rec.family = family_of(rel)
        rec.revision = revision_of(rel)[0]
    fams: dict[str, list[FileRecord]] = {}
    for rec in records.values():
        fams.setdefault(rec.family, []).append(rec)
    for members in fams.values():
        if len(members) < 2:
            continue
        live = [r for r in members if r.status != "WITHDRAWN" and revision_of(r.rel)[1]]
        if not live:
            live = [r for r in members if r.status != "WITHDRAWN"]
        if not live:
            continue
        current = max(live, key=lambda r: revision_of(r.rel)[1] or (-1,))
        for r in members:
            if r is current:
                continue
            older = revision_of(r.rel)[1] and revision_of(current.rel)[1] and revision_of(r.rel)[1] < revision_of(current.rel)[1]
            if r.status == "WITHDRAWN" or older:
                r.status = "WITHDRAWN" if r.status == "WITHDRAWN" else "SUPERSEDED"
                r.superseded_by = current.rel
