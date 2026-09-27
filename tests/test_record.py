import pytest
from pydantic import ValidationError

from silentpath.record import Confidence, CostSample, ObservedPath, Output, RunRecord


def test_identical_requested_and_observed_is_never_a_fallback():
    """FakeProducer sets observed == requested, so this is the path the GPU-free
    demo takes. Revision 2 reported a fallback on all four demo records."""
    p = ObservedPath(requested="ROCM_ATTN", observed="ROCM_ATTN",
                     confidence=Confidence.CONFIRMED, signals={})
    assert p.is_silent_fallback is False
    assert p.is_unresolved is False


def test_alias_match_is_not_a_silent_fallback():
    """The realistic case: the class name never contains the env token."""
    p = ObservedPath(requested="ROCM_ATTN", observed="ROCmFlashAttentionBackend",
                     confidence=Confidence.CONFIRMED, signals={})
    assert p.is_silent_fallback is False


def test_wrong_backend_is_a_silent_fallback():
    p = ObservedPath(requested="ROCM_ATTN", observed="TritonAttentionBackend",
                     confidence=Confidence.CONFIRMED, signals={})
    assert p.is_silent_fallback is True
    # A genuine fallback is resolved, not unknown. Without this, `is_unresolved`
    # could be written as `not backend_matches(...)` and silently count every
    # real fallback as unresolved as well.
    assert p.is_unresolved is False


def test_unknown_requested_token_is_unresolved_not_a_fallback():
    p = ObservedPath(requested="SOME_FUTURE_BACKEND", observed="TritonAttentionBackend",
                     confidence=Confidence.REPORTED, signals={})
    assert p.is_silent_fallback is False
    assert p.is_unresolved is True


def test_unrecorded_observed_name_is_unresolved_not_a_fallback():
    p = ObservedPath(requested="ROCM_ATTN", observed="SomeBrandNewBackendV9",
                     confidence=Confidence.REPORTED, signals={})
    assert p.is_silent_fallback is False
    assert p.is_unresolved is True


def test_fallback_is_false_when_observation_is_unknown():
    """Absence of data is not data."""
    p = ObservedPath(requested="ROCM_ATTN", observed=None,
                     confidence=Confidence.UNKNOWN, signals={})
    assert p.is_silent_fallback is False
    assert p.is_unresolved is True


def test_cost_sample_rejects_zero_repetitions():
    with pytest.raises(ValidationError):
        CostSample(wall_seconds=1.0, samples=0)


def test_cost_sample_separates_billable_cell_time_from_generate_time():
    """The budget is charged for the whole cell; the speed claim uses generate()
    only. Conflating them charges the credit seconds for a cell that took
    minutes, because engine init and model load dominate."""
    c = CostSample(wall_seconds=2.0, cell_wall_seconds=310.0, samples=3)
    assert c.wall_seconds == 2.0
    assert c.cell_wall_seconds == 310.0


def test_record_id_is_stable_for_identical_inputs():
    common = dict(
        path=ObservedPath(requested="ROCM_ATTN", observed="ROCmFlashAttentionBackend",
                          confidence=Confidence.CONFIRMED, signals={}),
    )
    a = RunRecord.build(config={"model": "m", "backend": "ROCM_ATTN"}, workload_id="w1",
                        cost=CostSample(wall_seconds=1.0, samples=3),
                        output=Output(text="42", token_ids=[1], chosen_logprobs=[-0.5]),
                        **common)
    b = RunRecord.build(config={"backend": "ROCM_ATTN", "model": "m"}, workload_id="w1",
                        cost=CostSample(wall_seconds=99.0, samples=1),
                        output=Output(text="different", token_ids=[9], chosen_logprobs=[-1.0]),
                        **common)
    # Identity depends on what was requested, not on what came back — otherwise
    # the cache could never recognise a completed cell.
    assert a.record_id == b.record_id


def test_record_id_changes_with_workload():
    common = dict(
        path=ObservedPath(requested="A", observed="A", confidence=Confidence.CONFIRMED, signals={}),
        cost=CostSample(wall_seconds=1.0, samples=1),
        output=Output(text="x", token_ids=[1], chosen_logprobs=[-0.1]),
    )
    assert (RunRecord.build(config={"model": "m"}, workload_id="w1", **common).record_id
            != RunRecord.build(config={"model": "m"}, workload_id="w2", **common).record_id)


def test_record_id_tolerates_non_json_config_values():
    """A Path in a config must not blow up identity computation."""
    from pathlib import Path
    rid = RunRecord.compute_id({"model": Path("/tmp/m")}, "w1")
    assert isinstance(rid, str) and len(rid) == 16


def test_round_trips_through_json():
    r = RunRecord.build(config={"model": "m"}, workload_id="w1",
                        path=ObservedPath(requested="A", observed="A",
                                          confidence=Confidence.CONFIRMED, signals={}),
                        cost=CostSample(wall_seconds=1.0, samples=1),
                        output=Output(text="x", token_ids=[1], chosen_logprobs=[-0.1]))
    assert RunRecord.model_validate_json(r.model_dump_json()) == r
