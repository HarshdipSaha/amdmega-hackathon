import time

import pytest

from sourcebound import pipeline
from sourcebound.engine import FakeEngine
from sourcebound.gates import grounded, judge
from sourcebound.index import build_index
from sourcebound.reader import Pack, parse_reply
from tests.fixtures import kit_fake as kf


@pytest.fixture
def kit_index(kit_corpus, tmp_path):
    return build_index(kit_corpus, FakeEngine(transcripts=kf.TRANSCRIPTS), tmp_path / "idx", time.monotonic() + 300,
                       workers=2, log=lambda *a: None)


def reply(answer, *ev, status="answered"):
    return parse_reply(__import__("json").dumps(kf._r(answer, *ev, status=status)))


PACK = Pack(text="")


def test_grounding_is_strict_but_tolerates_spacing():
    doc = "Maximum junction temperature .......... 94 C\nBoard power (TBP) ..................... 410 W"
    assert grounded("maximum junction temperature .......... 94 C", doc)
    assert grounded("Maximum  junction temperature .......... 94 C", doc)
    assert not grounded("Maximum junction temperature .......... 105 C", doc)
    assert not grounded("94 C", "Board power 410 W")


def test_fabricated_quote_is_refused(kit_index):
    v = judge(kit_index, PACK, "q", reply("99", (kf.R2, "Maximum junction temperature 99 C", "value")))
    assert not v.answer and v.retry


def test_withdrawn_value_is_rejected_with_exclusion(kit_index):
    v = judge(kit_index, PACK, "q", reply("105", (kf.R1, "Maximum junction temperature .......... 105 C", "value")))
    assert not v.answer and v.exclude == [kf.R1] and kf.R2 in v.retry


def test_answer_must_appear_in_its_evidence(kit_index):
    v = judge(kit_index, PACK, "q", reply("4.3.1", (kf.CSV, kf.ROW, "value")))
    assert not v.answer and "does not appear" in v.retry


def test_link_dropped_when_question_supplies_the_identifier(kit_index):
    r = reply("4.3.2", (kf.CSV, kf.ROW, "value"), (kf.LOG, "incident logged against ORR-1847", "link"))
    assert judge(kit_index, PACK, "Which firmware version fixed ticket ORR-1847?", r).citations == [kf.CSV]
    v = judge(kit_index, PACK, "The production log shows an incident. Which release fixed it?", r)
    assert v.answer == "4.3.2" and v.citations == [kf.CSV, kf.LOG]


def test_topical_neighbour_is_never_cited(kit_index):
    r = reply("4.3.2", (kf.CSV, kf.ROW, "value"), ("engineering/meridian_release_notes.txt", "4.3.1  Fixes a rare hang", "link"))
    assert judge(kit_index, PACK, "Which release fixed the incident?", r).citations == [kf.CSV]


def test_unparseable_and_not_found():
    assert parse_reply("no json here") is None
    assert parse_reply('```json\n{"status": "answered", "answer": "B14", "evidence": []}\n```')["answer"] == "B14"
    assert parse_reply('<think>x</think>{"answer": "", "evidence": []}')["status"] == "not_found"


def test_pipeline_retries_past_withdrawn_and_completes_qualifier(kit_index):
    eng = FakeEngine(replies=kf.REPLIES, warrant=kf.warrant)
    out = pipeline.answer(kit_index, eng, "What is the maximum junction temperature of the TQ-40?", 20)
    assert (out["answer"], out["citations"]) == ("94", [kf.R2])
    out = pipeline.answer(kit_index, eng, "What board revision is printed on the asset label?", 20)
    assert (out["answer"], out["citations"]) == ("REV-C2", ["support/asset_label.jpg"])
    assert any(n == 1 for task, q, n in eng.calls if "asset label" in q and task == "read")   # image attached


def test_warrant_no_leads_to_refusal(kit_index):
    eng = FakeEngine(replies=kf.REPLIES, warrant=kf.warrant)
    out = pipeline.answer(kit_index, eng, "What is the unit price of the TQ-40 at 10,000 unit volume?", 20)
    assert (out["answer"], out["citations"]) == ("", [])


def test_need_lookup_runs_one_more_hop(kit_index):
    first = {"status": "need_lookup", "answer": "", "evidence": [], "lookup": ["ORR-1847"]}
    second = kf._r("4.3.2", (kf.CSV, kf.ROW, "value"), (kf.LOG, "incident logged against ORR-1847", "link"))
    eng = FakeEngine(replies=[("which defect fix", [first, second])])
    out = pipeline.answer(kit_index, eng, "For the logged incident, which defect fix release applies?", 20)
    assert out["answer"] == "4.3.2" and out["citations"] == [kf.CSV, kf.LOG]


def test_deadline_short_circuits_to_refusal(kit_index):
    out = pipeline.answer(kit_index, FakeEngine(replies=kf.REPLIES), "What error code is logged?", 0.5)
    assert out["answer"] == "" and out["citations"] == []
