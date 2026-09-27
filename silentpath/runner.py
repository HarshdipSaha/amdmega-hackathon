"""Drive a matrix through a producer into a store, respecting cache and budget."""
from __future__ import annotations

from silentpath.budget import GpuBudget
from silentpath.matrix import Cell
from silentpath.producers.base import Producer
from silentpath.record import RunRecord
from silentpath.store import RecordStore


def billable_seconds(record: RunRecord) -> float:
    """What the credit is actually charged for one cell.

    `cell_wall_seconds` covers process start, engine init and model load, which
    dominate a subprocess-per-cell design. `wall_seconds` covers generate() only
    and would under-charge by orders of magnitude — and would charge **zero**
    for a cell that ran for thirty minutes and then OOM'd.

    Explicit None check, not `or`: a genuinely measured 0.0 is falsy and would
    silently fall through to generate time.
    """
    if record.cost.cell_wall_seconds is not None:
        return record.cost.cell_wall_seconds
    return record.cost.wall_seconds


def run_matrix(cells: list[Cell], producer: Producer, store: RecordStore,
               budget: GpuBudget, estimate_seconds: float) -> int:
    """Run every not-yet-completed cell. Returns how many were newly executed.

    `estimate_seconds` is required, with no default, because the *first* cell
    has no observation behind it and an unbounded first cell means the ceiling
    never fires at all on a single-cell sweep.

    Thereafter the estimate is the **maximum** observed cost, not the mean: a
    mean lags a rising cost curve and lets the cap be exceeded on the way up.

    Two checks, not one. The pre-flight `check_or_raise` acts on a guess; the
    post-hoc `raise_if_over` catches a guess that was too low. Together, a bad
    estimate costs one cell rather than the whole credit.

    Raises BudgetExceeded before starting an unaffordable cell. Work already
    written to the store is preserved, so a later call resumes.
    """
    if estimate_seconds <= 0:
        raise ValueError("estimate_seconds must be positive")

    observed: list[float] = []
    executed = 0
    for cell in cells:
        record_id = RunRecord.compute_id(cell.config, cell.workload_id)
        if store.has(record_id):
            continue

        est = max([estimate_seconds, *observed])
        budget.check_or_raise(est)

        record = producer.run(cell)
        store.append(record)

        spent = billable_seconds(record)     # charged even when the cell failed
        budget.record(spent)
        observed.append(spent)
        executed += 1

        budget.raise_if_over()
    return executed
