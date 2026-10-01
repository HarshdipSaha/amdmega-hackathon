import os
import sys

import pytest

from sourcebound.walk import OLE, detect, walk


def test_kit_hazards_are_skipped_and_everything_else_found(kit_corpus):
    files, ledger = walk(kit_corpus)
    rels = {f.rel for f in files}
    assert "vendor/telemetry_capture.dat" not in rels
    assert {"specs/tq40_datasheet_r2.pdf", "planning/roadmap_fy27.docx", "support/rma_parts.xlsx",
            "support/bug_database.csv", "logs/prod_inference_2026-09-02.log", "engineering/ingest_service.py",
            "specs/backplane_pinout.png", "support/asset_label.jpg"} <= rels
    assert any(e["path"] == "vendor/telemetry_capture.dat" and "unknown type" in e["reason"] for e in ledger)


def _deny(name):
    def opener(path, mode="r", *a, **k):
        if os.path.basename(path) == name:
            raise PermissionError(13, "Permission denied", path)
        return open(path, mode, *a, **k)
    return opener


@pytest.mark.parametrize("victim", ["internal_audit.txt", "asset_label.jpg", "backplane_pinout.png"])
def test_unreadable_file_never_stops_the_walk(kit_corpus, victim):
    files, ledger = walk(kit_corpus, opener=_deny(victim))
    rels = {f.rel for f in files}
    assert all(os.path.basename(r) != victim for r in rels)
    assert len(rels) == 11                         # 13 files - unknown .dat - the denied one
    assert any(victim in e["path"] and "PermissionError" in e["reason"] for e in ledger)


@pytest.mark.skipif(sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="chmod 000 only blocks a non-root POSIX user")
def test_real_chmod_000_file_and_directory(kit_corpus):
    (kit_corpus / "vendor" / "internal_audit.txt").chmod(0)
    locked = kit_corpus / "zz_locked"
    locked.mkdir()
    (locked / "x.txt").write_text("hidden")
    locked.chmod(0)
    try:
        files, ledger = walk(kit_corpus)
        assert "vendor/internal_audit.txt" not in {f.rel for f in files}
        assert any(e["path"] == "zz_locked/" for e in ledger)
    finally:
        locked.chmod(0o755)


def test_detect_magic_bytes():
    assert detect(".pdf", b"%PDF-1.7") == ("pdf", "")
    assert detect(".pdf", b"hello")[0] is None
    assert detect(".xlsx", OLE)[0] is None                 # encrypted OOXML is an OLE container
    assert detect(".docx", b"PK\x03\x04") == ("docx", "")
    assert detect(".txt", b"abc\x00def")[0] is None
    assert detect(".md", b"# notes") == ("text", "")
    assert detect(".dat", bytes(range(16)))[0] is None
    assert detect("", b"plain")[0] is None


def test_empty_corpus_and_empty_file(tmp_path):
    (tmp_path / "empty_dir").mkdir()
    (tmp_path / "zero.txt").write_bytes(b"")
    files, ledger = walk(tmp_path)
    assert files == [] and ledger == [{"path": "zero.txt", "reason": "empty file"}]
