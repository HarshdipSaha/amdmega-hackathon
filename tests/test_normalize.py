from roadread.normalize import normalize, matches

def test_normalize_rules_from_brief():
    assert normalize("京A·12345") == "京A12345"
    assert normalize("speed limit 65") == "SPEEDLIMIT65"
    assert normalize("7-AB.C_1 23") == "7ABC123"

def test_normalize_keeps_other_punctuation_and_cjk():
    assert normalize("A/B!") == "A/B!"          # only - . · _ and whitespace are removed
    assert normalize("沪b·88888") == "沪B88888"
    assert normalize("京A•12345") == "京A•12345"  # look-alikes are NOT removed by the evaluator (see Task 3)

def test_matches():
    assert matches("ROAD WORK AHEAD", "road work ahead")
    assert matches("JHT2951", "JHT 2951")
    assert not matches("7ABC123", "7ABC128")
