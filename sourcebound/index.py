"""The corpus index: segments, file records, BM25, identifier index, dense vectors; build, save, load (spec §4-5)."""
from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .pool import ParserPool
from .records import FileRecord, Segment, assign_status
from .terms import id_key, identifiers, terms
from .walk import walk

IMAGE_STUB = "image file {name} (no transcript available)"


@dataclass
class Index:
    root: str
    files: dict[str, FileRecord]
    segments: list[Segment]
    ledger: list[dict] = field(default_factory=list)
    vectors: np.ndarray | None = None
    fingerprint: str = ""
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_file: dict[str, list[int]] = {}
        for i, s in enumerate(self.segments):
            self.by_file.setdefault(s.file, []).append(i)
        docs = [terms(s.text) + terms(s.file.replace("/", " ")) for s in self.segments]
        self.bm25 = BM25(docs)
        self.ids: dict[str, set[int]] = {}
        for i, s in enumerate(self.segments):
            for tok in identifiers(s.text):
                self.ids.setdefault(id_key(tok), set()).add(i)

    def file_text(self, rel: str) -> str:
        return "\n".join(self.segments[i].text for i in self.by_file.get(rel, []))

    # ------------------------------------------------------------ persistence
    def save(self, d: Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "index.json.tmp"
        tmp.write_text(json.dumps({
            "root": self.root, "fingerprint": self.fingerprint, "stats": self.stats, "ledger": self.ledger,
            "files": {k: v.to_dict() for k, v in self.files.items()},
            "segments": [asdict(s) for s in self.segments]}, ensure_ascii=False), encoding="utf-8")
        if self.vectors is not None:
            np.save(d / "vectors.npy", self.vectors)
        elif (d / "vectors.npy").exists():
            (d / "vectors.npy").unlink()
        os.replace(tmp, d / "index.json")

    @classmethod
    def load(cls, d: Path) -> "Index | None":
        d = Path(d)
        try:
            data = json.loads((d / "index.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        vec = np.load(d / "vectors.npy") if (d / "vectors.npy").exists() else None
        return cls(root=data["root"], files={k: FileRecord(**v) for k, v in data["files"].items()},
                   segments=[Segment(**s) for s in data["segments"]], ledger=data["ledger"], vectors=vec,
                   fingerprint=data["fingerprint"], stats=data.get("stats", {}))


def fingerprint(root: Path) -> str:
    h = hashlib.sha1()
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        dirnames.sort()
        for n in sorted(filenames):
            try:
                st = os.stat(os.path.join(dirpath, n), follow_symlinks=False)
                h.update(f"{os.path.relpath(os.path.join(dirpath, n), root)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
            except OSError:
                h.update(f"{n}|unreadable\n".encode())
    return h.hexdigest()


def _sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_index(root: Path, engine, index_dir: Path, deadline: float, workers: int = 4,
                parse_timeout_s: float = 20.0, opener=open, log=print) -> Index:
    """Walk, parse, transcribe, embed, persist. `deadline` is a time.monotonic() value; optional work stops early."""
    t0 = time.monotonic()
    root, index_dir = Path(root), Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    recs, ledger = walk(root, opener=opener)
    records = {r.rel: r for r in recs}
    log(f"walk: {len(records)} files, {len(ledger)} skipped")

    # 1. parse text-bearing files in killable children
    jobs = [(r.rel, (r.path, r.rel, r.ftype)) for r in recs]
    results = ParserPool(workers=workers, timeout_s=parse_timeout_s).run(jobs, deadline=deadline - 90) if jobs else {}
    segments: list[Segment] = []
    heads: dict[str, str] = {}
    ocr_jobs: list[tuple[str, str, str]] = []      # (rel, kind, locator)
    for rel, (status, res) in sorted(results.items()):
        if status != "ok":
            ledger.append({"path": rel, "reason": str(res)})
            records.pop(rel, None)
            continue
        heads[rel] = res.get("head", "")
        segments.extend(Segment(**s) for s in res["segments"] if s["text"])
        if res.get("image"):
            ocr_jobs.append((rel, "image", "image"))
        ocr_jobs += [(rel, "page", str(p)) for p in res.get("ocr_pages", [])]
        ocr_jobs += [(rel, "media", m) for m in res.get("media", [])]
    log(f"parse: {len(segments)} segments in {time.monotonic() - t0:.1f}s")

    # 2. transcribe images, then text-less PDF pages, then embedded document images, while time remains
    order = {"image": 0, "page": 1, "media": 2}
    ocr_jobs.sort(key=lambda j: (order[j[1]], j[0], j[2]))
    cache_path = index_dir / "transcripts.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    render_dir = index_dir / "renders"
    render_dir.mkdir(exist_ok=True)
    pending: list[tuple[str, str, str, str, str]] = []   # (rel, kind, locator, image_path, cache_key)
    for rel, kind, loc in ocr_jobs:
        try:
            img_path = _materialize(records[rel], kind, loc, render_dir)
            key = f"{getattr(engine, 'name', '?')}:{_sha1(img_path)}"
            pending.append((rel, kind, loc, img_path, key))
        except Exception as e:  # noqa: BLE001
            ledger.append({"path": f"{rel}#{loc}", "reason": f"render failed: {type(e).__name__}: {e}"})
    transcribed = 0
    batch = int(os.environ.get("SB_OCR_BATCH", "4"))
    for i in range(0, len(pending), batch):
        chunk = pending[i:i + batch]
        todo = [p for p in chunk if p[4] not in cache]
        if todo and time.monotonic() < deadline - 45:
            texts = engine.transcribe([p[3] for p in todo])
            for p, text in zip(todo, texts):
                cache[p[4]] = text
        for rel, kind, loc, img_path, key in chunk:
            text = cache.get(key, "")
            if text.strip():
                transcribed += 1
                segments.append(Segment(rel, "image" if kind == "image" else f"{kind} {loc}", "transcript", text))
            elif kind == "image":
                segments.append(Segment(rel, "image", "transcript", IMAGE_STUB.format(name=Path(rel).name)))
    for rel, kind, loc, img_path, key in pending:
        if kind == "image":
            records[rel].extra["image_path"] = records[rel].path
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    log(f"transcribe: {transcribed}/{len(pending)} in {time.monotonic() - t0:.1f}s")

    # 3. supersession, dense vectors, persist
    assign_status(records, heads)
    segments = [s for s in segments if s.file in records]
    vectors = None
    if segments and time.monotonic() < deadline - 10:
        vectors = engine.embed([f"{s.file}\n{s.text}"[:2000] for s in segments], kind="document")
    idx = Index(root=str(root), files=records, segments=segments, ledger=ledger, vectors=vectors,
                fingerprint=fingerprint(root),
                stats={"files": len(records), "skipped": len(ledger), "segments": len(segments),
                       "transcribed": transcribed, "seconds": round(time.monotonic() - t0, 2)})
    idx.save(index_dir)
    log(f"index: {idx.stats}")
    return idx


def _materialize(rec: FileRecord, kind: str, loc: str, render_dir: Path) -> str:
    """Path of an image file to transcribe: the file itself, a rendered PDF page, or an extracted DOCX image."""
    if kind == "image":
        return rec.path
    safe = hashlib.sha1(f"{rec.rel}#{loc}".encode()).hexdigest()[:16]
    if kind == "page":
        import fitz
        out = render_dir / f"{safe}.png"
        with fitz.open(rec.path) as doc:
            doc[int(loc)].get_pixmap(dpi=150).save(str(out))
        return str(out)
    out = render_dir / f"{safe}{Path(loc).suffix.lower()}"
    with zipfile.ZipFile(rec.path) as z:
        out.write_bytes(z.read(loc))
    return str(out)
