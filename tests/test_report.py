from silentpath.record import Confidence, CostSample, ObservedPath, Output, RunRecord
from silentpath.report import findings_only, render_divergence_table


def make(backend, text, decision, seconds=1.0, workload="w1", samples=3, logprobs=None):
    return RunRecord.build(
        config={"backend": backend}, workload_id=workload,
        path=ObservedPath(requested=backend, observed=backend,
                          confidence=Confidence.CONFIRMED, signals={}),
        cost=CostSample(wall_seconds=seconds, device_seconds=seconds, samples=samples,
                        wall_stdev=0.01 if samples > 1 else None),
        output=Output(text=text, token_ids=[1], chosen_logprobs=logprobs or [-0.5],
                      decision=decision))


def test_table_reports_a_decision_divergence():
    rows = render_divergence_table([make("ROCM_ATTN", "42.00", "42.00"),
                                    make("TRITON_ATTN", "47.00", "47.00")])
    assert len(rows) == 1
    assert rows[0]["level"] == "decision"
    assert rows[0]["is_finding"] is True


def test_findings_only_filters_out_genuine_numeric_noise():
    """The fixture must actually reach NUMERIC — identical logprobs would make
    this test vacuous, since it would only ever exercise IDENTICAL."""
    rows = render_divergence_table([
        make("ROCM_ATTN", "42.00", "42.00", logprobs=[-0.5]),
        make("TRITON_ATTN", "42.00", "42.00", logprobs=[-0.50000001])])
    assert rows[0]["level"] == "numeric"
    assert findings_only(rows) == []


def test_failed_records_are_excluded_from_comparison():
    bad = make("TRITON_ATTN", "", None).model_copy(update={"ok": False})
    assert render_divergence_table([make("ROCM_ATTN", "42.00", "42.00"), bad]) == []


def test_records_from_different_workloads_are_not_cross_compared():
    assert render_divergence_table([
        make("ROCM_ATTN", "42.00", "42.00", workload="w1"),
        make("TRITON_ATTN", "47.00", "47.00", workload="w2")]) == []


def test_cost_ratio_is_orientation_independent():
    """A 5.5x slowdown must read as 5.5 regardless of insertion order, and the
    slower backend must be named."""
    slow_first = render_divergence_table([make("TRITON_ATTN", "42.00", "42.00", seconds=5.5),
                                          make("ROCM_ATTN", "42.00", "42.00", seconds=1.0)])
    fast_first = render_divergence_table([make("ROCM_ATTN", "42.00", "42.00", seconds=1.0),
                                          make("TRITON_ATTN", "42.00", "42.00", seconds=5.5)])
    assert slow_first[0]["cost_ratio"] == 5.5
    assert fast_first[0]["cost_ratio"] == 5.5
    assert slow_first[0]["slower"] == "TRITON_ATTN"
    assert fast_first[0]["slower"] == "TRITON_ATTN"


def test_row_carries_repetition_count_and_variance():
    """The stdev columns are asserted, not merely present — revision 2 could
    have both columns deleted with a green suite."""
    rows = render_divergence_table([make("ROCM_ATTN", "42.00", "42.00"),
                                    make("TRITON_ATTN", "42.00", "42.00", seconds=5.5)])
    assert rows[0]["samples_min"] == 3
    assert rows[0]["cost_ratio_admissible"] is True
    assert rows[0]["a_wall_stdev"] == 0.01
    assert rows[0]["b_wall_stdev"] == 0.01


def test_single_shot_timing_is_not_an_admissible_speed_result():
    rows = render_divergence_table([make("ROCM_ATTN", "42.00", "42.00", samples=1),
                                    make("TRITON_ATTN", "42.00", "42.00", seconds=5.5, samples=1)])
    assert rows[0]["samples_min"] == 1
    assert rows[0]["cost_ratio_admissible"] is False


def test_cost_ratio_uses_generate_time_not_whole_cell_time():
    """Cell time is dominated by engine init and model load. Comparing it would
    measure startup, not kernels — the ratio here must be 5.5, not ~1.02."""
    a = RunRecord.build(
        config={"backend": "ROCM_ATTN"}, workload_id="w1",
        path=ObservedPath(requested="ROCM_ATTN", observed="ROCM_ATTN",
                          confidence=Confidence.CONFIRMED, signals={}),
        cost=CostSample(wall_seconds=1.0, device_seconds=1.0, cell_wall_seconds=600.0,
                        samples=3, wall_stdev=0.01),
        output=Output(text="42.00", token_ids=[1], chosen_logprobs=[-0.5], decision="42.00"))
    b = RunRecord.build(
        config={"backend": "TRITON_ATTN"}, workload_id="w1",
        path=ObservedPath(requested="TRITON_ATTN", observed="TRITON_ATTN",
                          confidence=Confidence.CONFIRMED, signals={}),
        cost=CostSample(wall_seconds=5.5, device_seconds=5.5, cell_wall_seconds=610.0,
                        samples=3, wall_stdev=0.01),
        output=Output(text="42.00", token_ids=[1], chosen_logprobs=[-0.5], decision="42.00"))
    assert render_divergence_table([a, b])[0]["cost_ratio"] == 5.5


def test_row_reports_fallback_and_unresolved_columns():
    """Both columns are load-bearing for D1's headline and were unasserted."""
    clean = make("ROCM_ATTN", "42.00", "42.00")
    rows = render_divergence_table([clean, make("TRITON_ATTN", "42.00", "42.00")])
    assert rows[0]["silent_fallback"] is False
    assert rows[0]["unresolved_path"] is False

    fell_back = clean.model_copy(update={"path": ObservedPath(
        requested="ROCM_ATTN", observed="TritonAttentionBackend",
        confidence=Confidence.CONFIRMED, signals={})})
    rows = render_divergence_table([fell_back, make("TRITON_ATTN", "42.00", "42.00")])
    assert rows[0]["silent_fallback"] is True
