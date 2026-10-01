import time

from sourcebound.pool import ParserPool


def test_hung_crashed_and_raising_jobs_do_not_block_the_rest():
    pool = ParserPool("tests.sb_helpers:slow_or_fast", workers=2, timeout_s=3)
    jobs = [(k, ("p", k, "text")) for k in ("a", "hang", "b", "crash", "c", "raise", "d")]
    t = time.monotonic()
    res = pool.run(jobs)
    assert time.monotonic() - t < 30
    assert res["hang"][0] == "err" and "timeout" in res["hang"][1]
    assert res["crash"] == ("err", "parser process died")
    assert res["raise"] == ("err", "ValueError: bad file")
    for k in "abcd":
        assert res[k] == ("ok", {"segments": [], "head": k, "ocr_pages": [], "media": []})


def test_deadline_marks_remaining_jobs():
    pool = ParserPool("tests.sb_helpers:slow_or_fast", workers=1, timeout_s=30)
    res = pool.run([("hang", ("p", "hang", "text")), ("b", ("p", "b", "text"))], deadline=time.monotonic() + 4)
    assert res["hang"][0] == "err" and res["b"][0] == "err"
