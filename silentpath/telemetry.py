"""Measured cost. HIP events for device time, amd-smi for memory, wall clock always.

rocprofiler-sdk is deliberately not used: no first-class Python binding, and the
LD_PRELOAD C++ pattern costs about two weeks for a need hackathon-grade timing
already meets.

This module is the single timing implementation. The vLLM worker imports it
rather than keeping its own loop.
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Callable

from silentpath.record import CostSample


class Stopwatch:
    def __enter__(self) -> Stopwatch:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0


def repeat_timed(fn: Callable[[], Any], repetitions: int = 3) -> tuple[CostSample, Any]:
    """Run fn repeatedly; return the cost sample and the last returned value.

    Variance matters: AITER has been reported to show 2-16x higher measurement
    variability than other paths, so a single-shot timing is not an admissible
    speed claim, and report rows carry `samples` so that can be enforced.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")

    torch = _torch_with_gpu()
    wall: list[float] = []
    device: list[float] = []
    value: Any = None

    for _ in range(repetitions):
        start = end = None
        if torch is not None:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        with Stopwatch() as sw:
            value = fn()
        if torch is not None:
            end.record()
            torch.cuda.synchronize()
            device.append(start.elapsed_time(end) / 1000.0)   # ms -> s
        wall.append(sw.seconds)

    sample = CostSample(
        wall_seconds=statistics.fmean(wall),
        device_seconds=statistics.fmean(device) if device else None,
        samples=repetitions,
        wall_stdev=statistics.stdev(wall) if len(wall) > 1 else None,
        **_device_memory())
    return sample, value


def _torch_with_gpu():
    """torch.cuda maps to HIP on ROCm builds. Absent on the dev laptop."""
    try:
        import torch
    except ImportError:
        return None
    try:
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def _device_memory() -> dict:
    """Best-effort VRAM reading; absent on non-ROCm machines."""
    try:
        import amdsmi
    except ImportError:
        return {}
    try:
        amdsmi.amdsmi_init()
        handles = amdsmi.amdsmi_get_processor_handles()
        if not handles:
            return {}
        mem = amdsmi.amdsmi_get_gpu_memory_usage(handles[0], amdsmi.AmdSmiMemoryType.VRAM)
        return {"peak_memory_bytes": int(mem)}
    except Exception:
        return {}          # telemetry is never allowed to fail a measurement
    finally:
        try:
            amdsmi.amdsmi_shut_down()
        except Exception:
            pass
