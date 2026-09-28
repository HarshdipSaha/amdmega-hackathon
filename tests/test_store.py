from silentpath.record import Confidence, CostSample, ObservedPath, Output, RunRecord
from silentpath.store import RecordStore


def make(backend="ROCM_ATTN", workload="w1"):
    return RunRecord.build(
        config={"backend": backend}, workload_id=workload,
        path=ObservedPath(requested=backend, observed=backend,
                          confidence=Confidence.CONFIRMED, signals={}),
        cost=CostSample(wall_seconds=1.0, samples=1),
        output=Output(text="42", token_ids=[1], chosen_logprobs=[-0.5]))


def test_append_then_load_round_trips(tmp_store_dir):
    r = make()
    RecordStore(tmp_store_dir).append(r)
    assert RecordStore(tmp_store_dir).load()[0] == r


def test_has_reports_membership_by_id(tmp_store_dir):
    s, r = RecordStore(tmp_store_dir), make()
    assert s.has(r.record_id) is False
    s.append(r)
    assert s.has(r.record_id) is True


def test_reopened_store_still_knows_completed_cells(tmp_store_dir):
    """Resume depends on this: a fresh process must not recompute a done cell."""
    r = make()
    RecordStore(tmp_store_dir).append(r)
    assert RecordStore(tmp_store_dir).has(r.record_id) is True


def test_append_is_idempotent(tmp_store_dir):
    s, r = RecordStore(tmp_store_dir), make()
    s.append(r)
    s.append(r)
    assert len(s.load()) == 1


def test_by_workload_groups_records(tmp_store_dir):
    s = RecordStore(tmp_store_dir)
    for b, w in [("ROCM_ATTN", "w1"), ("TRITON_ATTN", "w1"), ("ROCM_ATTN", "w2")]:
        s.append(make(b, w))
    groups = s.by_workload()
    assert sorted(groups) == ["w1", "w2"]
    assert len(groups["w1"]) == 2


def test_corrupt_line_is_skipped_not_fatal(tmp_store_dir):
    """A truncated final line is the normal result of an interrupted run and
    must not make the whole store unreadable."""
    s = RecordStore(tmp_store_dir)
    s.append(make())
    with open(s.path, "a", encoding="utf-8") as fh:
        fh.write('{"record_id": "trunc\n')
    assert len(RecordStore(tmp_store_dir).load()) == 1
