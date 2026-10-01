"""Defensive corpus walk (spec §3). One bad file or directory never stops the walk.

Every skipped path lands in the ledger with a reason, so a run can be audited afterwards.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

from .records import FileRecord

KNOWN = {".pdf": "pdf", ".docx": "docx", ".docm": "docx", ".xlsx": "xlsx", ".xlsm": "xlsx", ".csv": "csv",
         ".tsv": "csv", ".txt": "text", ".log": "text", ".py": "code", ".png": "image", ".jpg": "image",
         ".jpeg": "image", ".tif": "image", ".tiff": "image", ".bmp": "image", ".gif": "image", ".webp": "image"}
TEXT_ALLOW = {".md": "text", ".rst": "text", ".json": "text", ".yaml": "text", ".yml": "text", ".toml": "text",
              ".ini": "text", ".cfg": "text", ".conf": "text", ".html": "text", ".htm": "text", ".xml": "text",
              ".sh": "code", ".js": "code", ".ts": "code", ".c": "code", ".h": "code", ".cpp": "code",
              ".java": "code", ".go": "code", ".rs": "code", ".sql": "code"}
MAX_BYTES = int(os.environ.get("SB_MAX_FILE_BYTES", str(64 * 2**20)))
OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def detect(ext: str, head: bytes) -> tuple[str | None, str]:
    """(ftype, "") or (None, reason). The extension proposes, the magic bytes confirm."""
    if head.startswith(OLE):
        return None, "OLE container (encrypted Office file or legacy format)"
    kind = KNOWN.get(ext)
    if kind == "pdf":
        return ("pdf", "") if head.startswith(b"%PDF") else (None, "extension .pdf but not a PDF")
    if kind in ("docx", "xlsx"):
        return (kind, "") if head.startswith(b"PK") else (None, f"extension {ext} but not an OOXML zip")
    if kind == "image":
        return "image", ""
    if kind in ("csv", "text", "code") or ext in TEXT_ALLOW:
        if b"\x00" in head:
            return None, "binary content in a text-type file"
        if kind is None:
            sample = head.decode("utf-8", errors="replace")
            printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in sample)
            if sample and printable / len(sample) < 0.95:
                return None, "allow-listed extension but not printable text"
            return TEXT_ALLOW[ext], ""
        return kind, ""
    return None, f"unknown type {ext or '(no extension)'}"


def zip_problem(path: str) -> str:
    """Reason to refuse an OOXML zip (encrypted members, zip bomb), or ''."""
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        if any(i.flag_bits & 0x1 for i in infos):
            return "encrypted zip member"
        if sum(i.file_size for i in infos) > 512 * 2**20:
            return "uncompressed size over 512 MiB"
    return ""


def walk(root: Path, opener=open) -> tuple[list[FileRecord], list[dict]]:
    """Return readable, typed files under root (sorted, symlinks not followed) and a ledger of skips."""
    root = Path(root)
    files: list[FileRecord] = []
    ledger: list[dict] = []

    def skip(rel: str, reason: str) -> None:
        ledger.append({"path": rel, "reason": reason})

    def visit(d: Path) -> None:
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError as e:
            skip(_rel(root, d) + "/", f"cannot list directory: {type(e).__name__}")
            return
        for e in entries:
            rel = _rel(root, Path(e.path))
            try:
                if e.is_symlink():
                    skip(rel, "symlink not followed")
                elif e.is_dir(follow_symlinks=False):
                    visit(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    rec = _probe(e, rel, opener, skip)
                    if rec:
                        files.append(rec)
                else:
                    skip(rel, "not a regular file")
            except Exception as ex:  # noqa: BLE001 - one entry must never stop the walk
                skip(rel, f"{type(ex).__name__}: {ex}")

    visit(root)
    return files, ledger


def _probe(e: os.DirEntry, rel: str, opener, skip) -> FileRecord | None:
    size = e.stat(follow_symlinks=False).st_size
    if size == 0:
        skip(rel, "empty file")
        return None
    if size > MAX_BYTES:
        skip(rel, f"larger than {MAX_BYTES} bytes")
        return None
    try:
        with opener(e.path, "rb") as f:
            head = f.read(4096)
    except OSError as ex:
        skip(rel, f"unreadable: {type(ex).__name__}")
        return None
    ftype, reason = detect(Path(e.name).suffix.lower(), head)
    if not ftype:
        skip(rel, reason)
        return None
    return FileRecord(rel=rel, path=e.path, ftype=ftype, size=size)


def _rel(root: Path, p: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()
