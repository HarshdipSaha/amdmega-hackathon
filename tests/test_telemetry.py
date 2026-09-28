import time

from silentpath.telemetry import Stopwatch, repeat_timed


def test_stopwatch_measures_elapsed():
    with Stopwatch() as sw:
        time.sleep(0.01)
    assert sw.seconds >= 0.01


def test_repeat_timed_runs_n_times_and_reports_variance():
    calls = []
    sample, _ = repeat_timed(lambda: calls.append(1), repetitions=3)
    assert len(calls) == 3
    assert sample.samples == 3
    assert sample.wall_stdev is not None


def test_repeat_timed_returns_the_last_value():
    """The caller needs the inference output, not just its timing."""
    _, value = repeat_timed(lambda: "result", repetitions=2)
    assert value == "result"


def test_single_repetition_reports_no_stdev():
    """One measurement has no variance; reporting 0.0 would imply precision
    that was never measured."""
    sample, _ = repeat_timed(lambda: None, repetitions=1)
    assert sample.samples == 1
    assert sample.wall_stdev is None


def test_repeat_timed_rejects_zero_repetitions():
    try:
        repeat_timed(lambda: None, repetitions=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_device_seconds_is_absent_rather_than_faked_without_a_gpu(monkeypatch):
    """Asserted by forcing the no-GPU branch, so the test means something on a
    machine that happens to have one. `x is None or x >= 0` would be vacuous."""
    import silentpath.telemetry as t
    monkeypatch.setattr(t, "_torch_with_gpu", lambda: None)
    sample, _ = t.repeat_timed(lambda: None, repetitions=2)
    assert sample.device_seconds is None


def test_torch_peak_allocation_is_recorded_and_amd_smi_sample_is_not_called_peak(monkeypatch):
    import silentpath.telemetry as t

    class FakeCuda:
        resets = 0
        def reset_peak_memory_stats(self): self.resets += 1
        def max_memory_allocated(self): return 1234
        def synchronize(self): pass

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(t, "_torch_with_gpu", lambda: FakeTorch())
    monkeypatch.setattr(t, "_amd_smi_memory_sample", lambda: {"memory_sample_bytes": 5678})
    sample, _ = t.repeat_timed(lambda: None, repetitions=1)
    assert sample.peak_memory_bytes == 1234
    assert sample.memory_sample_bytes == 5678
    assert FakeTorch.cuda.resets == 1


def test_timed_function_is_bracketed_by_gpu_synchronization(monkeypatch):
    import silentpath.telemetry as t

    calls = []
    class Event:
        def __init__(self, enable_timing): pass
        def record(self): calls.append("event")
        def elapsed_time(self, other): return 10
    class Cuda:
        def synchronize(self): calls.append("sync")
    Cuda.Event = Event
    class FakeTorch:
        cuda = Cuda()
    monkeypatch.setattr(t, "_torch_with_gpu", lambda: FakeTorch())
    def fn(): calls.append("fn")
    t.repeat_timed(fn, repetitions=1)
    assert calls.index("sync") < calls.index("fn") < calls.index("sync", calls.index("fn"))
