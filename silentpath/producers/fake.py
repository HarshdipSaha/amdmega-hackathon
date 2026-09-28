"""Deterministic in-memory producer, so the runner is testable without a GPU."""
from __future__ import annotations

from silentpath.matrix import Cell
from silentpath.record import Confidence, CostSample, ObservedPath, Output, RunRecord


class FakeProducer:
    def __init__(self, outputs: dict[tuple[str, str], str] | None = None,
                 fallbacks: dict[str, str] | None = None,
                 failures: set[str] | None = None,
                 seconds: float = 1.0, samples: int = 3,
                 cell_seconds: float | None = None,
                 cell_seconds_by_backend: dict[str, float] | None = None) -> None:
        self.outputs = outputs or {}
        self.fallbacks = fallbacks or {}
        self.failures = failures or set()
        self.seconds = seconds
        self.samples = samples
        self.cell_seconds = cell_seconds
        self.cell_seconds_by_backend = cell_seconds_by_backend or {}
        self.calls: list[Cell] = []

    def _cell_seconds(self, backend: str) -> float:
        if backend in self.cell_seconds_by_backend:
            return self.cell_seconds_by_backend[backend]
        return self.cell_seconds if self.cell_seconds is not None else self.seconds

    def run(self, cell: Cell) -> RunRecord:
        self.calls.append(cell)
        observed = self.fallbacks.get(cell.backend, cell.backend)
        path = ObservedPath(requested=cell.backend, observed=observed,
                            confidence=Confidence.CONFIRMED, signals={"fake": observed})
        cost = CostSample(wall_seconds=self.seconds,
                          cell_wall_seconds=self._cell_seconds(cell.backend),
                          device_seconds=self.seconds,
                          samples=self.samples, wall_stdev=0.0 if self.samples > 1 else None)

        if cell.backend in self.failures:
            return RunRecord.build(
                config=cell.config, workload_id=cell.workload_id, path=path, cost=cost,
                output=Output(text="", token_ids=[], chosen_logprobs=[]),
                ok=False, error="scripted failure")

        text = self.outputs.get((cell.backend, cell.workload_id), "42.00")
        return RunRecord.build(
            config=cell.config, workload_id=cell.workload_id, path=path, cost=cost,
            output=Output(text=text, token_ids=[1, 2], chosen_logprobs=[-0.5, -0.25],
                          decision=text))
