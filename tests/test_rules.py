import random
from roadread.rules import postprocess, is_chinese_plate, chinese_format_ok

def test_repeated_digits_never_collapsed():
    for s in ["沪B·88888", "京A·11111", "粤B·D88888"]:
        assert postprocess(s, kind="plate").text == s

def test_chinese_length_preserved_property():
    rnd = random.Random(0)
    for _ in range(500):
        s = rnd.choice("京沪粤川") + rnd.choice("ABCDE") + "·" + "".join(rnd.choice("0123456789ABDF8") for _ in range(rnd.choice([4, 5, 6, 7])))
        assert len(postprocess(s, kind="plate").text) == len(s)

def test_chinese_format_check_flags_not_edits():
    r = postprocess("沪b·888888", kind="plate")          # 6-char serial without D/F -> suspect, unchanged
    assert r.text == "沪B·888888" and r.suspect
    assert not postprocess("京A·12345", kind="plate").suspect
    assert not postprocess("粤B·12345F", kind="plate").suspect   # large new-energy: D/F at the end

def test_chinese_routed_by_content_even_if_kind_sign():
    assert postprocess("京A·1O2I5", kind="sign").text == "京A·10215"

def test_io_mapping_only_inside_chinese_serial():
    assert postprocess("京A·1O2I5", kind="plate").text == "京A·10215"
    assert postprocess("京I·12345", kind="plate").text == "京I·12345"   # province letter untouched
    assert postprocess("OIL AHEAD", kind="sign").text == "OIL AHEAD"
    assert postprocess("7OIL123", kind="plate").text == "7OIL123"       # US plate untouched

def test_canonicalizes_separators_case_and_fullwidth():
    assert postprocess("京A•12345", kind="plate").text == "京A·12345"
    assert postprocess("京a･ 12345", kind="plate").text == "京A·12345"
    assert postprocess("ＳＴＯＰ", kind="sign").text == "STOP"

def test_banner_filter_plate_only_and_keeps_vanity():
    assert postprocess("CALIFORNIA 7ABC123", kind="plate").text == "7ABC123"
    assert postprocess("NEW YORK JHT 2951 EXCELSIOR", kind="plate").text == "JHT 2951"   # sample 3
    assert postprocess("7ABC123 THE LONE STAR STATE", kind="plate").text == "7ABC123"
    assert postprocess("TEXAS", kind="plate").text == "TEXAS"            # nothing plausible left -> keep
    assert postprocess("STOP", kind="sign").text == "STOP"
    assert postprocess("WASHINGTON ST", kind="sign").text == "WASHINGTON ST"

def test_helpers():
    assert is_chinese_plate("京A·12345") and not is_chinese_plate("7ABC123")
    assert chinese_format_ok("京A·12345") and chinese_format_ok("粤B·D12345")
    assert not chinese_format_ok("京A·1234")
