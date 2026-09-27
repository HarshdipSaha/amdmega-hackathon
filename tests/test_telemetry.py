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
