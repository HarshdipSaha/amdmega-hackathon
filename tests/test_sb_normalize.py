import pytest

from sourcebound.normalize import align_number, complete_qualifier, contains_answer, official, shape


def test_official_matches_grader_rules():
    assert official("Q3 FY27") == official("q3fy27") == "Q3FY27"
    assert official("ORR-FAN-2214-B") == "ORRFAN2214B"
    assert official("4.3.2") == "432"


@pytest.mark.parametrize("raw,want", [
    ("94 °C", "94"), ("94°C", "94"), ("94 C", "94"), ("180 seconds", "180"), ("180s", "180"),
    ("4.1 TB/s", "4.1"), ("$84.50", "84.50"), ("`E7731`", "E7731"), ('"REV-C2".', "REV-C2"),
    ("pin B14", "B14"), ("Version: 4.3.2", "4.3.2"), ("4.3.2", "4.3.2"), ("12V-2x6", "12V-2x6"),
    ("Q3 FY27", "Q3 FY27"), ("PIN-7", "PIN-7"), ("10,000", "10,000"),
])
def test_shape_keeps_value_drops_label_and_unit(raw, want):
    assert shape(raw) == want


def test_contains_answer_respects_token_boundaries():
    assert contains_answer("94", "Maximum junction temperature .......... 94 C")
    assert not contains_answer("94", "batch 1940 accepted")
    assert contains_answer("4.3.2", "status: closed | fixed_in: 4.3.2")
    assert not contains_answer("4.3", "fixed_in: 4.3.2")
    assert contains_answer("Q3 FY27", "enters customer sampling in Q3 FY27.")
    assert contains_answer("Q3FY27", "in Q3 FY27")


def test_complete_qualifier_expands_only_truncated_identifiers():
    assert complete_qualifier("Q3", "TQ-60 enters customer sampling in Q3 FY27.") == "Q3 FY27"
    assert complete_qualifier("C2", "BOARD REVISION: REV-C2") == "REV-C2"
    assert complete_qualifier("REV-C2", "BOARD REVISION: REV-C2") == "REV-C2"
    assert complete_qualifier("40", "MODEL: TQ-40") == "40"            # numbers are never expanded
    assert complete_qualifier("B14", "B14: THERM_ALERT#") == "B14"     # already standalone
    assert complete_qualifier("4.3.2", "Meridian 4.3.2") == "4.3.2"


def test_align_number_prefers_the_printed_form():
    assert align_number("84.50", "unit_cost_usd: 84.5") == "84.5"
    assert align_number("94", "94 C") == "94"
    assert align_number("E7731", "E7731") == "E7731"
