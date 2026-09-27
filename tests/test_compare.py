from silentpath.compare import DivergenceLevel, compare, is_finding
from silentpath.record import Confidence, CostSample, ObservedPath, Output, RunRecord


def make(text, ids, logprobs, decision=None, backend="ROCM_ATTN", workload="w1", samples=3):
    return RunRecord.build(
        config={"backend": backend}, workload_id=workload,
        path=ObservedPath(requested=backend, observed=backend,
                          confidence=Confidence.CONFIRMED, signals={}),
        cost=CostSample(wall_seconds=1.0, device_seconds=1.0, samples=samples),
        output=Output(text=text, token_ids=ids, chosen_logprobs=logprobs, decision=decision),
    )


def test_bit_identical_records_are_identical():
    d = compare(make("42.00", [1, 2], [-0.5, -0.25], "42.00", "ROCM_ATTN"),
                make("42.00", [1, 2], [-0.5, -0.25], "42.00", "TRITON_ATTN"))
    assert d.level is DivergenceLevel.IDENTICAL
    assert d.max_abs_logprob_delta == 0.0
    assert is_finding(d) is False


def test_numerically_close_but_same_decision_is_not_a_finding():
    """The case the project must NOT report as a discovery. Different kernels
    producing different bits is expected behaviour."""
    d = compare(make("42.00", [1, 2], [-0.5, -0.25], "42.00", "ROCM_ATTN"),
                make("42.00", [1, 2], [-0.50000001, -0.25], "42.00", "TRITON_ATTN"))
    assert d.level is DivergenceLevel.NUMERIC
    assert d.max_abs_logprob_delta > 0.0
    assert d.text_equal is True
    assert is_finding(d) is False


def test_different_decision_is_a_finding():
    d = compare(make("42.00", [1, 2], [-0.5, -0.25], "42.00", "ROCM_ATTN"),
                make("47.00", [1, 3], [-0.5, -0.25], "47.00", "TRITON_ATTN"))
    assert d.level is DivergenceLevel.DECISION
    assert is_finding(d) is True


def test_same_text_but_different_extracted_decision_is_a_finding():
    """Two runs can emit the same prose and still disagree on the extracted value."""
    d = compare(make("The total is 42.00 dollars", [1, 2], [-0.5, -0.25], "42.00", "ROCM_ATTN"),
                make("The total is 42.00 dollars", [1, 2], [-0.5, -0.25], "4200", "TRITON_ATTN"))
    assert d.level is DivergenceLevel.DECISION
    assert is_finding(d) is True


def test_differing_token_counts_are_decision_level():
    assert compare(make("42", [1], [-0.5], "42", "ROCM_ATTN"),
                   make("42.00", [1, 2], [-0.5, -0.25], "42.00", "TRITON_ATTN")
                   ).level is DivergenceLevel.DECISION


def test_differing_token_counts_report_no_delta_rather_than_infinity():
    """No element-wise delta exists. Revision 2 used float('inf'), which
    json.dumps emits as bare `Infinity` — not valid JSON, and it reached the
    MCP tool surface."""
    import json
    d = compare(make("42", [1], [-0.5], "42", "ROCM_ATTN"),
                make("42.00", [1, 2], [-0.5, -0.25], "42.00", "TRITON_ATTN"))
    assert d.max_abs_logprob_delta is None
    json.loads(json.dumps({"delta": d.max_abs_logprob_delta}))   # must not raise


def test_comparing_different_workloads_raises():
    try:
        compare(make("42", [1], [-0.5], "42", "ROCM_ATTN", workload="w1"),
                make("42", [1], [-0.5], "42", "TRITON_ATTN", workload="w2"))
    except ValueError as exc:
        assert "workload" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for mismatched workloads")


def test_failed_run_comparison_raises():
    bad = make("", [], [], None, "TRITON_ATTN").model_copy(update={"ok": False})
    try:
        compare(make("42", [1], [-0.5], "42", "ROCM_ATTN"), bad)
    except ValueError as exc:
        assert "failed" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for failed run")


def test_missing_decisions_fall_back_to_text_equality():
    d = compare(make("42.00", [1, 2], [-0.5, -0.25], None, "ROCM_ATTN"),
                make("42.00", [1, 2], [-0.5, -0.25], None, "TRITON_ATTN"))
    assert d.decision_equal is None
    assert d.level is DivergenceLevel.IDENTICAL
