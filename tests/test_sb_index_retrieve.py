import time

from sourcebound.engine import FakeEngine
from sourcebound.index import IMAGE_STUB, Index, build_index
from sourcebound.retrieve import bridge, rank_files, search
from tests.fixtures.kit_fake import TRANSCRIPTS


def _build(corpus, tmp_path, transcripts=TRANSCRIPTS):
    eng = FakeEngine(transcripts=transcripts)
    return build_index(corpus, eng, tmp_path / "idx", time.monotonic() + 300, workers=2, log=lambda *a: None), eng


def test_kit_index_skips_hazards_and_marks_supersession(kit_corpus, tmp_path):
    idx, _ = _build(kit_corpus, tmp_path)
    skipped = {e["path"] for e in idx.ledger}
    assert {"vendor/telemetry_capture.dat", "vendor/supplier_agreement_ENCRYPTED.pdf"} <= skipped
    assert "vendor/supplier_agreement_ENCRYPTED.pdf" not in idx.files
    assert idx.files["specs/tq40_datasheet_r1_WITHDRAWN.pdf"].status == "WITHDRAWN"
    assert idx.files["specs/tq40_datasheet_r1_WITHDRAWN.pdf"].superseded_by == "specs/tq40_datasheet_r2.pdf"
    assert "B14: THERM_ALERT#" in idx.file_text("specs/backplane_pinout.png")
    assert idx.vectors is not None and idx.vectors.shape[0] == len(idx.segments)


def test_index_roundtrip(kit_corpus, tmp_path):
    idx, _ = _build(kit_corpus, tmp_path)
    again = Index.load(tmp_path / "idx")
    assert again.stats == idx.stats and len(again.segments) == len(idx.segments)
    assert again.files["support/asset_label.jpg"].extra["image_path"].endswith("asset_label.jpg")
    assert again.fingerprint == idx.fingerprint


def test_untranscribed_image_gets_a_filename_stub(kit_corpus, tmp_path):
    idx, _ = _build(kit_corpus, tmp_path, transcripts={})
    assert idx.file_text("support/asset_label.jpg") == IMAGE_STUB.format(name="asset_label.jpg")


def test_retrieval_finds_each_single_hop_file(kit_corpus, tmp_path, kit_questions):
    idx, eng = _build(kit_corpus, tmp_path)
    for q in kit_questions:
        if not q["expected_citations"] or len(q["expected_citations"]) > 1:
            continue
        top = [r.rel for r in rank_files(idx, search(idx, eng, q["query"]))[:8]]
        assert q["expected_citations"][0] in top, (q["query"], top)


def test_bridge_follows_ticket_from_log_but_not_from_question(kit_corpus, tmp_path):
    idx, eng = _build(kit_corpus, tmp_path)
    q9 = "The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?"
    ranked = rank_files(idx, search(idx, eng, q9))
    links = bridge(idx, q9, ranked)
    assert any(b.ident == "ORR-1847" and idx.segments[b.seg].file == "support/bug_database.csv"
               and b.from_file == "logs/prod_inference_2026-09-02.log" for b in links)
    q4 = "Which firmware version fixed ticket ORR-1847?"
    assert not any(b.ident == "ORR-1847" for b in bridge(idx, q4, rank_files(idx, search(idx, eng, q4))))
