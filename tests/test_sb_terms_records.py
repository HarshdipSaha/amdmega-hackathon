from sourcebound.records import FileRecord, assign_status, family_of, revision_of
from sourcebound.terms import identifiers, terms


def test_terms_keep_codes_whole_and_split():
    t = terms("incident logged against ORR-1847 (E7731) on 4.3.2")
    for want in ("orr-1847", "orr", "1847", "orr1847", "e7731", "4.3.2"):
        assert want in t
    assert "the" not in terms("what is the value", query=True)


def test_identifiers_are_letter_led_codes_with_digits():
    ids = identifiers("ERROR E7731: throttle; incident logged against ORR-1847; fan ORR-FAN-2214-B; die 0 at 91C")
    assert ids == ["E7731", "ORR-1847", "ORR-FAN-2214-B"]
    assert identifiers("2026-09-02T03:14:41Z meridian[2211]") == []


def test_revision_and_family():
    assert revision_of("specs/tq40_datasheet_r2.pdf")[0] == "2"
    assert revision_of("spec_revB.pdf")[0] == "B"
    assert revision_of("support/rma_parts.xlsx") == ("", ())
    assert family_of("specs/tq40_datasheet_r1_WITHDRAWN.pdf") == family_of("specs/tq40_datasheet_r2.pdf")
    assert family_of("logs/a_2026-09-02.log") != family_of("logs/a_2026-09-03.log")


def _recs(*names):
    return {n: FileRecord(rel=n, path=n, ftype="pdf") for n in names}


def test_withdrawn_by_name_points_to_current_sibling():
    r = _recs("specs/tq40_datasheet_r1_WITHDRAWN.pdf", "specs/tq40_datasheet_r2.pdf")
    assign_status(r, {"specs/tq40_datasheet_r2.pdf": "Datasheet, revision 2 - supersedes revision 1\n...\nRevision 1 is withdrawn."})
    assert r["specs/tq40_datasheet_r1_WITHDRAWN.pdf"].status == "WITHDRAWN"
    assert r["specs/tq40_datasheet_r1_WITHDRAWN.pdf"].superseded_by == "specs/tq40_datasheet_r2.pdf"
    assert r["specs/tq40_datasheet_r2.pdf"].status == "CURRENT"


def test_withdrawn_by_heading_and_older_revision_superseded():
    r = _recs("a/spec_v1.pdf", "a/spec_v2.pdf", "a/spec_v3.pdf")
    assign_status(r, {"a/spec_v3.pdf": "Spec v3\nWITHDRAWN - do not use", "a/spec_v1.pdf": "Spec v1", "a/spec_v2.pdf": "Spec v2"})
    assert r["a/spec_v3.pdf"].status == "WITHDRAWN" and r["a/spec_v3.pdf"].superseded_by == "a/spec_v2.pdf"
    assert r["a/spec_v1.pdf"].status == "SUPERSEDED" and r["a/spec_v1.pdf"].superseded_by == "a/spec_v2.pdf"
    assert r["a/spec_v2.pdf"].status == "CURRENT"
