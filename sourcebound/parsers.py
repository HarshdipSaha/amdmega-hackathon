"""Per-type parsers (spec §4). They run inside ParserPool children, so they may hang or crash safely.

parse_file(path, rel, ftype) -> {"segments": [dict], "head": str, "ocr_pages": [int], "media": [str]}
Raises Encrypted for any encrypted container: such files are never indexed.
"""
from __future__ import annotations

import ast
import csv
import io
import re
import xml.etree.ElementTree as ET
import zipfile

from .walk import zip_problem

CHUNK = 1800          # characters per prose segment
WHOLE_TEXT = 2000     # text/log files up to this size are one segment
WHOLE_CODE = 3000
MAX_PAGES = 300
MAX_ROWS = 20000


class Encrypted(Exception):
    pass


def seg(rel: str, locator: str, kind: str, text: str) -> dict:
    return {"file": rel, "locator": locator, "kind": kind, "text": text.strip()}


def read_text(path: str) -> str:
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def split_text(text: str, limit: int = CHUNK) -> list[str]:
    """Split on blank lines, then lines, keeping chunks under `limit` characters."""
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], ""
    for p in paras:
        pieces = [p] if len(p) <= limit else [p[i:i + limit] for i in range(0, len(p), limit)]
        for piece in pieces:
            if buf and len(buf) + len(piece) + 2 > limit:
                out.append(buf)
                buf = ""
            buf = f"{buf}\n\n{piece}" if buf else piece
    if buf:
        out.append(buf)
    return out


def rows_to_segments(rel: str, rows: list[list[str]], prefix: str, locator_fmt: str) -> list[dict]:
    """Header row + data rows -> one 'h: v | h: v' segment per row (header names repeated on every row)."""
    rows = [[(c or "").strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    hi = next((i for i, r in enumerate(rows) if sum(1 for c in r if c) >= 2), None)
    if hi is None:
        return [seg(rel, locator_fmt.format(1), "prose", prefix + "\n".join(" ".join(c for c in r if c) for r in rows))]
    out = []
    if hi > 0:
        out.append(seg(rel, locator_fmt.format(1), "prose", prefix + "\n".join(" ".join(c for c in r if c) for r in rows[:hi])))
    header = [h or f"col{j + 1}" for j, h in enumerate(rows[hi])]
    for i, r in enumerate(rows[hi + 1:], start=hi + 2):
        cells = [f"{header[j] if j < len(header) else f'col{j + 1}'}: {c}" for j, c in enumerate(r) if c]
        if cells:
            out.append(seg(rel, locator_fmt.format(i), "row", prefix + " | ".join(cells)))
    return out


# ---------------------------------------------------------------- PDF
def parse_pdf(path: str, rel: str) -> dict:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    try:
        # needs_pass: user password required. is_encrypted / metadata["encryption"]: owner-password-only PDFs
        # open without a password and report is_encrypted False, so the metadata check is required too.
        if doc.needs_pass or doc.is_encrypted or (doc.metadata or {}).get("encryption"):
            raise Encrypted("encrypted PDF")
        segs, ocr, head = [], [], ""
        for i, page in enumerate(doc):
            if i >= MAX_PAGES:
                break
            text = page.get_text("text", sort=True)
            if i == 0:
                head = text
            if len(text.strip()) < 20:
                if page.get_images() or page.get_drawings():
                    ocr.append(i)
                continue
            for j, chunk in enumerate(split_text(text)):
                segs.append(seg(rel, f"page {i + 1}" + (f" part {j + 1}" if j else ""), "prose", chunk))
        return {"segments": segs, "head": head, "ocr_pages": ocr, "media": []}
    finally:
        doc.close()


# ---------------------------------------------------------------- DOCX (stdlib only)
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _para_text(p: ET.Element) -> str:
    out = []
    for el in p.iter():
        if el.tag == W + "t" and el.text:
            out.append(el.text)
        elif el.tag == W + "tab":
            out.append("\t")
        elif el.tag in (W + "br", W + "cr"):
            out.append("\n")
    return "".join(out)


def _table_rows(tbl: ET.Element) -> list[list[str]]:
    rows = []
    for tr in tbl.findall(W + "tr"):
        rows.append([" ".join(_para_text(p) for p in tc.iter(W + "p")).strip() for tc in tr.findall(W + "tc")])
    return rows


def parse_docx(path: str, rel: str) -> dict:
    problem = zip_problem(path)
    if problem:
        raise Encrypted(problem)
    segs, paras, head = [], [], ""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        body = ET.fromstring(z.read("word/document.xml")).find(W + "body")
        tcount = 0

        def flush() -> None:
            if paras:
                for k, chunk in enumerate(split_text("\n".join(paras))):
                    segs.append(seg(rel, f"text {len(segs) + 1}", "prose", chunk))
                paras.clear()

        for el in list(body) if body is not None else []:
            if el.tag == W + "tbl":
                flush()
                tcount += 1
                segs.extend(rows_to_segments(rel, _table_rows(el), f"table {tcount} | ", f"table {tcount} row {{}}"))
            elif el.tag in (W + "p", W + "sdt"):
                for p in ([el] if el.tag == W + "p" else el.iter(W + "p")):
                    t = _para_text(p)
                    if t.strip():
                        paras.append(t)
                        if not head:
                            head = t
        flush()
        for part in sorted(n for n in names if re.match(r"word/(header|footer|footnotes|endnotes)\d*\.xml$", n)):
            text = "\n".join(t for t in (_para_text(p) for p in ET.fromstring(z.read(part)).iter(W + "p")) if t.strip())
            if text.strip():
                segs.append(seg(rel, part.split("/")[-1].removesuffix(".xml"), "prose", text))
        media = [n for n in names if n.startswith("word/media/") and n.lower().endswith((".png", ".jpg", ".jpeg"))]
    head = "\n".join(s["text"] for s in segs[:2]) or head
    return {"segments": segs, "head": head, "ocr_pages": [], "media": media}


# ---------------------------------------------------------------- XLSX
def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def parse_xlsx(path: str, rel: str) -> dict:
    import openpyxl

    problem = zip_problem(path)
    if problem:
        raise Encrypted(problem)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    segs = []
    try:
        for ws in wb.worksheets:
            rows = []
            for i, r in enumerate(ws.iter_rows(values_only=True)):
                if i >= MAX_ROWS:
                    break
                rows.append([_cell(v) for v in r])
            segs.extend(rows_to_segments(rel, rows, f"sheet {ws.title} | ", f"{ws.title}!{{}}"))
    finally:
        wb.close()
    return {"segments": segs, "head": "\n".join(s["text"] for s in segs[:2]), "ocr_pages": [], "media": []}


# ---------------------------------------------------------------- CSV
def parse_csv(path: str, rel: str) -> dict:
    text = read_text(path)
    try:
        dialect = csv.Sniffer().sniff(text[:65536], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = []
    for i, r in enumerate(csv.reader(io.StringIO(text), dialect)):
        if i > MAX_ROWS:
            break
        rows.append(r)
    segs = rows_to_segments(rel, rows, "", "row {}")
    return {"segments": segs, "head": text[:500], "ocr_pages": [], "media": []}


# ---------------------------------------------------------------- text, logs, code
def parse_text(path: str, rel: str) -> dict:
    text = read_text(path)
    if len(text) <= WHOLE_TEXT:
        segs = [seg(rel, "all", "prose", text)] if text.strip() else []
    elif rel.lower().endswith(".log") or _looks_like_log(text):
        lines = text.splitlines()
        segs = [seg(rel, f"lines {i + 1}-{min(i + 24, len(lines))}", "log", "\n".join(lines[i:i + 24]))
                for i in range(0, len(lines), 18) if "\n".join(lines[i:i + 24]).strip()]
    else:
        segs = [seg(rel, f"part {k + 1}", "prose", c) for k, c in enumerate(split_text(text))]
    return {"segments": segs, "head": text[:500], "ocr_pages": [], "media": []}


def _looks_like_log(text: str) -> bool:
    lines = [ln for ln in text.splitlines()[:50] if ln.strip()]
    return bool(lines) and sum(bool(re.match(r"^\[?\d{4}-\d{2}-\d{2}", ln)) for ln in lines) / len(lines) > 0.6


def parse_code(path: str, rel: str) -> dict:
    text = read_text(path)
    if len(text) <= WHOLE_CODE:
        return {"segments": [seg(rel, "all", "code", text)] if text.strip() else [], "head": text[:500],
                "ocr_pages": [], "media": []}
    lines = text.splitlines()
    segs = []
    try:
        tree = ast.parse(text) if rel.endswith(".py") else None
    except SyntaxError:
        tree = None
    if tree is None:
        segs = [seg(rel, f"lines {i + 1}-{min(i + 60, len(lines))}", "code", "\n".join(lines[i:i + 60]))
                for i in range(0, len(lines), 50)]
    else:
        spans = [(n.lineno, n.end_lineno) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        inside = {ln for a, b in spans for ln in range(a, (b or a) + 1)}
        module = [lines[i - 1] for i in range(1, len(lines) + 1) if i not in inside]
        segs.append(seg(rel, "module", "code", "\n".join(module)))
        for a, b in spans:
            # include decorators/comments directly above the definition
            start = a
            while start > 1 and lines[start - 2].strip().startswith(("#", "@")):
                start -= 1
            segs.append(seg(rel, f"lines {start}-{b}", "code", "\n".join(lines[start - 1:b])))
    return {"segments": [s for s in segs if s["text"]], "head": text[:500], "ocr_pages": [], "media": []}


# ---------------------------------------------------------------- images
def parse_image(path: str, rel: str) -> dict:
    from PIL import Image

    with Image.open(path) as im:
        im.verify()          # raises on truncated or corrupt files
    return {"segments": [], "head": "", "ocr_pages": [], "media": [], "image": True}


PARSERS = {"pdf": parse_pdf, "docx": parse_docx, "xlsx": parse_xlsx, "csv": parse_csv, "text": parse_text,
           "code": parse_code, "image": parse_image}


def parse_file(path: str, rel: str, ftype: str) -> dict:
    return PARSERS[ftype](path, rel)
