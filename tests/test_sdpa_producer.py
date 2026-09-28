import pytest
from silentpath.producers.sdpa import (
    map_profiler_operators,
    model_revision_kwargs,
    parse_observed_backend,
    resolve_sdpa_backend,
    setup_gpu_device,
    run_timed_and_profile,
)


def test_model_revision_is_passed_to_transformers_loaders():
    revision = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert model_revision_kwargs({"model_revision": revision}) == {"revision": revision}
    assert model_revision_kwargs({}) == {}


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


def test_profiler_evidence_pass_is_separate_from_timed_repetitions(monkeypatch):
    import silentpath.producers.sdpa as sdpa

    events = []
    class Profiler:
        active = False
        def __enter__(self):
            self.active = True
            events.append("profiler-enter")
            return self
        def __exit__(self, *exc):
            self.active = False
            events.append("profiler-exit")
        def key_averages(self):
            return [type("Event", (), {"key": "aten::_scaled_dot_product_flash_attention"})()]

    profiler = Profiler()
    def repeat(fn, repetitions):
        events.append("timing-start")
        result = None
        for _ in range(repetitions):
            assert not profiler.active
            result = fn()
        events.append("timing-end")
        return "sample", result

    monkeypatch.setattr(sdpa, "repeat_timed", repeat)
    def timed_call():
        assert not profiler.active
        events.append("timed-generation")
        return "measured-output"
    def profiled_call():
        assert profiler.active
        events.append("profiled-generation")

    sample, output, operators, prof, error = run_timed_and_profile(
        timed_call, profiled_call, profiler, repetitions=2
    )
    assert (sample, output) == ("sample", "measured-output")
    assert events.index("timing-end") < events.index("profiler-enter")
    assert events.count("profiled-generation") == 1
    assert operators == ["aten::_scaled_dot_product_flash_attention"]
    assert prof is profiler
    assert error is None


def test_default_selector_leaves_sdpa_dispatch_unconstrained():
    from silentpath.producers.sdpa import _BACKENDS
    assert _BACKENDS["DEFAULT"] == "DEFAULT"
    class Enum:
        FLASH_ATTENTION = "flash"
    assert resolve_sdpa_backend("DEFAULT", Enum) is None
    assert resolve_sdpa_backend("FLASH_ATTENTION", Enum) == "flash"
