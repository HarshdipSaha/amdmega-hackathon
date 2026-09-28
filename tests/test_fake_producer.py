from silentpath.matrix import Cell
from silentpath.producers.fake import FakeProducer


def cell(backend="ROCM_ATTN", workload="w1"):
    return Cell(backend=backend, workload_id=workload, prompt="p",
                config={"model": "m", "backend": backend})


def test_produces_a_record_matching_the_cell():
    r = FakeProducer().run(cell())
    assert r.workload_id == "w1"
    assert r.path.requested == "ROCM_ATTN"
    assert r.ok is True


def test_scripted_outputs_let_tests_stage_divergence():
    p = FakeProducer(outputs={("ROCM_ATTN", "w1"): "42.00", ("TRITON_ATTN", "w1"): "47.00"})
    assert p.run(cell("ROCM_ATTN")).output.text == "42.00"
    assert p.run(cell("TRITON_ATTN")).output.text == "47.00"


def test_scripted_fallback_is_recorded_as_observed_path():
    """Uses realistic names: a class name never contains the env-var token."""
    r = FakeProducer(fallbacks={"ROCM_ATTN": "TritonAttentionBackend"}).run(cell("ROCM_ATTN"))
    assert r.path.observed == "TritonAttentionBackend"
    assert r.path.is_silent_fallback is True


def test_correct_backend_class_name_is_not_a_fallback():
    r = FakeProducer(fallbacks={"ROCM_ATTN": "ROCmFlashAttentionBackend"}).run(cell("ROCM_ATTN"))
    assert r.path.is_silent_fallback is False


def test_default_producer_reports_no_fallback():
    """The GPU-free demo path. Revision 2 reported a fallback on every record
    here, which broke Task 14's own sanity check before any GPU was involved."""
    r = FakeProducer().run(cell("ROCM_ATTN"))
    assert r.path.is_silent_fallback is False
    assert r.path.is_unresolved is False


def test_cell_time_defaults_to_generate_time_but_can_differ():
    r = FakeProducer(seconds=2.0, cell_seconds=600.0).run(cell())
    assert r.cost.wall_seconds == 2.0
    assert r.cost.cell_wall_seconds == 600.0


def test_failure_is_recorded_not_raised():
    r = FakeProducer(failures={"ROCM_ATTN"}).run(cell("ROCM_ATTN"))
    assert r.ok is False
    assert r.error is not None
