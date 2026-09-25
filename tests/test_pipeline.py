from PIL import Image
from roadread.engine_fake import FakeEngine
from roadread.pipeline import Pipeline, parse_reply

IMG = Image.new("RGB", (64, 32))

def test_parse_reply():
    assert parse_reply("KIND: plate\nTEXT: 7ABC123") == ("plate", "7ABC123", True)
    assert parse_reply("KIND: Plate\nTEXT: 京A·12345\n") == ("plate", "京A·12345", True)
    assert parse_reply("KIND: sign\nTEXT: ROAD\nWORK\nAHEAD") == ("sign", "ROAD WORK AHEAD", True)
    assert parse_reply("KIND: sign\nTEXT: STOP<|im_end|>") == ("sign", "STOP", True)
    assert parse_reply("```\nKIND: sign\nTEXT: STOP\n```") == ("sign", "STOP", True)
    assert parse_reply("7ABC123") == ("sign", "7ABC123", False)            # malformed -> flagged

def test_single_read_when_confident():
    eng = FakeEngine(["KIND: sign\nTEXT: SPEED LIMIT 65"])
    r = Pipeline(eng).run(IMG, deadline_s=27)
    assert r.text == "SPEED LIMIT 65" and eng.calls == 1 and r.confidence == 0.9

def test_suspect_triggers_one_reread_on_a_different_view():
    eng = FakeEngine(["KIND: plate\nTEXT: 沪B·888888", "KIND: plate\nTEXT: 沪B·88888"])
    r = Pipeline(eng).run(IMG, deadline_s=27)
    assert r.text == "沪B·88888" and eng.calls == 2
    assert eng.sizes == [(64, 32), (128, 64)]

def test_never_more_than_one_escalation_and_keeps_first_when_both_suspect():
    eng = FakeEngine(["KIND: plate\nTEXT: 沪B·888888", "KIND: plate\nTEXT: 沪B·8888888"])
    r = Pipeline(eng).run(IMG, deadline_s=27)
    assert eng.calls == 2 and r.text == "沪B·888888" and r.confidence == 0.3

def test_no_escalation_when_budget_too_small():
    eng = FakeEngine(["KIND: plate\nTEXT: 沪B·888888"], seconds=5)
    r = Pipeline(eng, reread_upper_s=10).run(IMG, deadline_s=12)
    assert eng.calls == 1 and r.text == "沪B·888888"

def test_truncated_output_is_suspect():
    eng = FakeEngine(["KIND: sign\nTEXT: ROAD WORK", "KIND: sign\nTEXT: ROAD WORK AHEAD"], truncated=[True, False])
    assert Pipeline(eng).run(IMG, deadline_s=27).text == "ROAD WORK AHEAD"

def test_cross_view_disagreement_triggers_escalation():
    eng = FakeEngine(["KIND: plate\nTEXT: 7ABC123", "KIND: plate\nTEXT: 7ABC128", "KIND: plate\nTEXT: 7ABC123"])
    r = Pipeline(eng, cross_view=True).run(IMG, deadline_s=27)
    assert eng.calls == 3 and r.text == "7ABC123" and r.diag["uncertain"]

def test_cross_view_agreement_stops_early():
    eng = FakeEngine(["KIND: plate\nTEXT: 7ABC123", "KIND: plate\nTEXT: 7ABC 123"])
    r = Pipeline(eng, cross_view=True).run(IMG, deadline_s=27)
    assert eng.calls == 2 and r.text == "7ABC123" and not r.diag["uncertain"]
