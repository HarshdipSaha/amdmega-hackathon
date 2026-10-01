import pytest

from sourcebound.parsers import Encrypted, parse_file
from tests.sb_helpers import write_docx


def texts(res):
    return "\n".join(s["text"] for s in res["segments"])


def test_pdf_pages_and_head(kit_corpus):
    r = parse_file(str(kit_corpus / "specs/tq40_datasheet_r2.pdf"), "specs/tq40_datasheet_r2.pdf", "pdf")
    assert "Maximum junction temperature .......... 94 C" in texts(r)
    assert r["segments"][0]["locator"] == "page 1"
    assert "supersedes revision 1" in r["head"]


def test_password_pdf_is_never_read(kit_corpus):
    with pytest.raises(Encrypted):
        parse_file(str(kit_corpus / "vendor/supplier_agreement_ENCRYPTED.pdf"), "x.pdf", "pdf")


def test_owner_password_only_pdf_is_also_refused(tmp_path):
    fitz = pytest.importorskip("fitz")
    d = fitz.open()
    d.new_page().insert_text((72, 72), "unit price 6412")
    d.save(tmp_path / "owner.pdf", encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="")
    assert fitz.open(tmp_path / "owner.pdf").is_encrypted is False     # why the metadata check exists
    with pytest.raises(Encrypted):
        parse_file(str(tmp_path / "owner.pdf"), "owner.pdf", "pdf")


def test_docx_paragraphs_and_tables(kit_corpus, tmp_path):
    r = parse_file(str(kit_corpus / "planning/roadmap_fy27.docx"), "planning/roadmap_fy27.docx", "docx")
    assert "TQ-60 enters customer sampling in Q3 FY27." in texts(r)
    write_docx(tmp_path / "t.docx", ["Milestones"], [["product", "milestone", "quarter"], ["KV-9", "tape-out", "Q1 FY28"]])
    r = parse_file(str(tmp_path / "t.docx"), "t.docx", "docx")
    assert "table 1 | product: KV-9 | milestone: tape-out | quarter: Q1 FY28" in texts(r)


def test_xlsx_every_sheet_with_headers_on_every_row(kit_corpus):
    r = parse_file(str(kit_corpus / "support/rma_parts.xlsx"), "support/rma_parts.xlsx", "xlsx")
    t = texts(r)
    assert "part_number: ORR-FAN-2214-B | description: Fan assembly, dual-rotor, field replaceable | compatible_with: TQ-40" in t
    assert "unit_cost_usd: 84.5" in t
    assert "Lead times are supplier-quoted" in t                    # the second sheet
    assert {s["kind"] for s in r["segments"]} >= {"row"}


def test_csv_rows_carry_headers(kit_corpus):
    r = parse_file(str(kit_corpus / "support/bug_database.csv"), "support/bug_database.csv", "csv")
    assert any(s["text"].startswith("ticket: ORR-1847 |") and "fixed_in: 4.3.2" in s["text"] for s in r["segments"])


def test_semicolon_latin1_csv(tmp_path):
    (tmp_path / "t.csv").write_bytes("part;cost\nGRÜN-1;12,5\n".encode("cp1252"))
    r = parse_file(str(tmp_path / "t.csv"), "t.csv", "csv")
    assert texts(r) == "part: GRÜN-1 | cost: 12,5"


def test_log_and_code(kit_corpus, tmp_path):
    r = parse_file(str(kit_corpus / "logs/prod_inference_2026-09-02.log"), "l.log", "text")
    assert "ERROR E7731: thermal throttle engaged" in texts(r)
    r = parse_file(str(kit_corpus / "engineering/ingest_service.py"), "i.py", "code")
    assert "DEFAULT_BATCH_TIMEOUT_S = 180" in texts(r) and "Raised from 60" in texts(r)
    big = "\n".join([f"# constant {i}\nC{i} = {i}" for i in range(200)]) + "\n\ndef f():\n    return 1\n"
    (tmp_path / "big.py").write_text(big)
    r = parse_file(str(tmp_path / "big.py"), "big.py", "code")
    assert any(s["locator"] == "module" and "C199 = 199" in s["text"] for s in r["segments"])
    assert any("def f()" in s["text"] for s in r["segments"])


def test_long_log_is_windowed(tmp_path):
    lines = [f"2026-09-02T03:{i // 60:02d}:{i % 60:02d}Z node INFO tick {i}" for i in range(100)]
    (tmp_path / "a.log").write_text("\n".join(lines))
    r = parse_file(str(tmp_path / "a.log"), "a.log", "text")
    assert len(r["segments"]) > 3 and all(s["kind"] == "log" for s in r["segments"])
    assert "tick 99" in texts(r)


def test_image_is_validated_not_read(kit_corpus):
    r = parse_file(str(kit_corpus / "support/asset_label.jpg"), "support/asset_label.jpg", "image")
    assert r["image"] is True and r["segments"] == []
