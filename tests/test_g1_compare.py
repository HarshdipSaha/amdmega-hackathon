from gate.g1_divergence import Verdict, classify


def _run(text, ids, logprobs, backend="ROCM_ATTN", observed=None, ok=True):
    return {"backend_requested": backend, "backend_observed": observed or backend, "ok": ok,
            "text": text, "token_ids": ids, "chosen_logprobs": logprobs}


def test_identical_runs_are_identical():
    assert classify([_run("42", [1], [-.5]), _run("42", [1], [-.5], "TRITON_ATTN")]).verdict is Verdict.IDENTICAL


def test_logprob_difference_with_same_text_is_numeric_only():
    assert classify([_run("42", [1], [-.5]), _run("42", [1], [-.5000001], "TRITON_ATTN")]).verdict is Verdict.NUMERIC_ONLY


def test_different_text_is_decision_divergent():
    assert classify([_run("42", [1], [-.5]), _run("47", [2], [-.5], "TRITON_ATTN")]).verdict is Verdict.DECISION_DIVERGENT


def test_fewer_than_two_successful_runs_is_blocked():
    assert classify([_run("42", [1], [-.5]), _run("", [], [], "TRITON_ATTN", ok=False)]).verdict is Verdict.BLOCKED


def test_decision_divergence_wins_over_numeric():
    runs = [_run("42", [1], [-.5]), _run("42", [1], [-.5000001], "TRITON_ATTN"), _run("47", [2], [-.5], "AITER_MLA")]
    assert classify(runs).verdict is Verdict.DECISION_DIVERGENT


def test_alias_match_is_not_reported_as_a_silent_fallback():
    runs = [_run("42", [1], [-.5], "ROCM_ATTN", "ROCmFlashAttentionBackend"), _run("42", [1], [-.5], "TRITON_ATTN", "TritonAttentionBackend")]
    assert classify(runs).silent_fallbacks == []


def test_genuine_fallback_is_reported():
    runs = [_run("42", [1], [-.5], "ROCM_ATTN", "TritonAttentionBackend"), _run("42", [1], [-.5], "TRITON_ATTN", "TritonAttentionBackend")]
    assert classify(runs).silent_fallbacks == [("ROCM_ATTN", "TritonAttentionBackend")]
