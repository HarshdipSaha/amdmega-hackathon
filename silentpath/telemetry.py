"""Measured cost. HIP event timing is elapsed event-region time, not GPU busy time."""
from __future__ import annotations

import statistics
import threading
import time
from typing import Any, Callable

from silentpath.record import CostSample

_TIMING_LOCK = threading.RLock()


class Stopwatch:
    def __enter__(self) -> Stopwatch:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0


def repeat_timed(fn: Callable[[], Any], repetitions: int = 3) -> tuple[CostSample, Any]:
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    torch = _torch_with_gpu()
    wall: list[float] = []
    device: list[float] = []
    value: Any = None
    with _TIMING_LOCK:
        if torch is not None:
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        for _ in range(repetitions):
            start = end = None
            if torch is not None:
                try:
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                except Exception:
                    start = end = None
            with Stopwatch() as sw:
                value = fn()
            if start is not None and end is not None:
                try:
                    end.record()
                    torch.cuda.synchronize()
                    device.append(start.elapsed_time(end) / 1000.0)
                except Exception:
                    pass
            wall.append(sw.seconds)
        peak = None
        if torch is not None:
            try:
                peak = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass
    sample = CostSample(
        wall_seconds=statistics.fmean(wall),
        device_seconds=statistics.fmean(device) if device else None,
        samples=repetitions,
        wall_stdev=statistics.stdev(wall) if len(wall) > 1 else None,
        peak_memory_bytes=peak,
        **_amd_smi_memory_sample(),
    )
    return sample, value


def _torch_with_gpu():
    """torch.cuda maps to HIP on ROCm builds. Absent on machines without GPU."""
    try:
        import torch
    except ImportError:
        return None
    try:
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def _amd_smi_memory_sample() -> dict:
    """Best-effort one-time AMD-SMI memory usage sample (not a peak)."""
    amdsmi = None
    try:
        import amdsmi
        amdsmi.amdsmi_init()
        handles = amdsmi.amdsmi_get_processor_handles()
        if not handles:
            return {}
        mem = amdsmi.amdsmi_get_gpu_memory_usage(handles[0], amdsmi.AmdSmiMemoryType.VRAM)
        return {"memory_sample_bytes": int(mem)}
    except Exception:
        return {}
    finally:
        try:
            if amdsmi is not None:
                amdsmi.amdsmi_shut_down()
        except Exception:
            pass
