import pytest

from silentpath.backends import backend_matches, names_agree, observed_backends_in


@pytest.mark.parametrize("token", ["ROCM_ATTN", "TRITON_ATTN", "ROCM_AITER_FA", "TRITON_MLA", "AITER_MLA"])
def test_identical_requested_and_observed_is_always_a_match(token):
    assert backend_matches(token, token) is True


def test_known_alias_matches_its_env_token():
    assert backend_matches("ROCM_ATTN", "ROCmFlashAttentionBackend") is True


def test_recognised_but_wrong_backend_is_a_mismatch():
    assert backend_matches("ROCM_ATTN", "TritonAttentionBackend") is False


def test_matching_is_case_and_punctuation_insensitive():
    assert backend_matches("triton_attn", "Triton Attention Backend") is True


def test_unknown_env_token_yields_no_evidence():
    assert backend_matches("SOME_FUTURE_BACKEND", "TritonAttentionBackend") is None


def test_unrecorded_observed_name_yields_no_evidence():
    assert backend_matches("ROCM_ATTN", "SomeBrandNewBackendV9") is None


@pytest.mark.parametrize("intro,log", [
    ("TritonAttentionBackend", "Triton Attention"),
    ("TritonAttentionBackend", "Triton Attention (V1)"),
    ("TritonAttentionBackend", "V1 Triton Attention"),
    ("ROCmFlashAttentionBackend", "ROCm Flash Attention"),
    ("AiterFlashAttentionBackend", "Aiter Flash Attention v3"),
])
def test_qualified_names_still_agree(intro, log):
    assert names_agree(intro, log) is True


def test_a_shorter_lookalike_does_not_agree():
    assert names_agree("ROCmFlashAttentionBackend", "Flash Attention") is False


def test_different_versions_are_different_backends():
    assert names_agree("FlashAttentionV2Backend", "FlashAttentionV3Backend") is False


def test_a_name_stripped_to_nothing_does_not_match_everything():
    assert names_agree("AV1", "AiterFlashAttentionBackend") is False
    assert names_agree("XV1Backend", "XLAAttentionBackend") is False


def test_default_selector_line_is_not_read_as_a_backend_name():
    assert observed_backends_in("Using the default attention backend selector\n") == []


def test_single_log_line_is_extracted():
    assert observed_backends_in("INFO Using Triton Attention backend.\n") == ["Triton Attention"]


def test_distinct_log_lines_are_all_returned():
    text = "Using Triton Attention backend.\nUsing ROCm Flash Attention backend.\n"
    assert len(observed_backends_in(text)) == 2


def test_repeated_identical_lines_collapse():
    text = "Using Triton Attention backend.\nUsing Triton Attention backend.\n"
    assert observed_backends_in(text) == ["Triton Attention"]


def test_no_match_returns_empty():
    assert observed_backends_in("nothing relevant here") == []
