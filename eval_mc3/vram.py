"""Sample used VRAM once per second with rocm-smi while an evaluation runs (no-op where rocm-smi is absent)."""
from __future__ import annotations

import json
import shutil
import subprocess
import threading


def used_bytes() -> int | None:
    if not shutil.which("rocm-smi"):
        return None
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], capture_output=True, text=True, timeout=10).stdout
        data = json.loads(out)
        return sum(int(v.get("VRAM Total Used Memory (B)", 0)) for v in data.values() if isinstance(v, dict))
    except Exception:
        return None


class VramSampler:
    def __init__(self, period_s: float = 1.0):
        self.period_s, self.peak, self._stop = period_s, None, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            b = used_bytes()
            if b is not None:
                self.peak = max(self.peak or 0, b)
            self._stop.wait(self.period_s)

    def start(self) -> None:
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        self._t.join(5)

    @property
    def peak_gib(self) -> float | None:
        return None if self.peak is None else round(self.peak / 2**30, 2)
