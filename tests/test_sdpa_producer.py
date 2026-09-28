import pytest
from silentpath.producers.sdpa import map_profiler_operators, parse_observed_backend, setup_gpu_device


def test_maps_flash_sdpa_operator():
    assert parse_observed_backend(["aten::_scaled_dot_product_flash_attention"] ) == "FLASH_ATTENTION"


def test_maps_math_sdpa_operator():
    assert parse_observed_backend(["aten::_scaled_dot_product_attention_math"]) == "MATH"


def test_ambiguous_or_unrelated_profiler_operators_stay_unresolved():
    assert parse_observed_backend(["aten::linear", "aten::matmul"]) is None
    assert parse_observed_backend([
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_attention_math",
    ]) is None


def test_operator_parser_returns_explicit_signals():
    mapped, signals = map_profiler_operators(["aten::_scaled_dot_product_flash_attention"])
    assert mapped == "FLASH_ATTENTION"
    assert signals == {"profiler": "aten::_scaled_dot_product_flash_attention"}


def test_gpu_setup_rejects_a_machine_without_rocm_or_cuda():
    class Cuda:
        @staticmethod
        def is_available(): return False
    class Torch:
        cuda = Cuda()
    with pytest.raises(RuntimeError, match="no CUDA/HIP GPU"):
        setup_gpu_device(Torch)
