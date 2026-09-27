import pytest

from silentpath.budget import BudgetExceeded, GpuBudget
from silentpath.matrix import expand, load_matrix
from silentpath.producers.fake import FakeProducer
from silentpath.runner import run_matrix
from silentpath.store import RecordStore

YAML = """
model: m
seed: 0
backends: [ROCM_ATTN, TRITON_ATTN]
workloads:
  - id: w1
    prompt: p1
"""


ONE_CELL_YAML = """
model: m
seed: 0
backends: [ROCM_ATTN]
workloads:
  - id: w1
    prompt: p1
"""

THREE_CELL_YAML = """
model: m
seed: 0
backends: [ROCM_ATTN, TRITON_ATTN, AITER_MLA]
workloads:
  - id: w1
    prompt: p1
"""


def test_runs_every_cell_once(tmp_store_dir):
    store = RecordStore(tmp_store_dir)
    assert run_matrix(expand(load_matrix(YAML)), FakeProducer(), store,
                      GpuBudget(cap_hours=10), estimate_seconds=300) == 2
    assert len(store.load()) == 2


def test_second_run_recomputes_nothing(tmp_store_dir):
    cells = expand(load_matrix(YAML))
    run_matrix(cells, FakeProducer(), RecordStore(tmp_store_dir),
               GpuBudget(cap_hours=10), estimate_seconds=300)
    second = FakeProducer()
    assert run_matrix(cells, second, RecordStore(tmp_store_dir),
                      GpuBudget(cap_hours=10), estimate_seconds=300) == 0
    assert second.calls == []      # the producer was never invoked


def test_estimate_seconds_must_be_positive(tmp_store_dir):
    with pytest.raises(ValueError):
        run_matrix(expand(load_matrix(YAML)), FakeProducer(), RecordStore(tmp_store_dir),
                   GpuBudget(cap_hours=10), estimate_seconds=0)


def test_budget_learns_from_observed_cost_and_stops(tmp_store_dir):
    store = RecordStore(tmp_store_dir)
    with pytest.raises(BudgetExceeded):
        run_matrix(expand(load_matrix(YAML)), FakeProducer(cell_seconds=1800), store,
                   GpuBudget(cap_hours=0.75), estimate_seconds=300)
    assert len(store.load()) == 1               # the first cell survived


def test_a_single_expensive_cell_is_bounded_after_the_fact(tmp_store_dir):
    """Revision 2 never bounded the first cell: one 7200s cell against a 0.5h
    cap raised nothing and overspent 4x. The post-hoc check is what fires."""
    budget = GpuBudget(cap_hours=0.5)
    with pytest.raises(BudgetExceeded):
        run_matrix(expand(load_matrix(ONE_CELL_YAML)), FakeProducer(cell_seconds=7200),
                   RecordStore(tmp_store_dir), budget, estimate_seconds=300)
    assert budget.spent_hours == pytest.approx(2.0)   # the 4x overspend, now caught


def test_estimate_uses_the_maximum_not_the_mean(tmp_store_dir):
    """A mean lags a rising cost curve and lets the cap be exceeded on the way
    up. Cells of 400s then 2500s against a 0.8h cap must stop at two."""
    store = RecordStore(tmp_store_dir)
    producer = FakeProducer(cell_seconds_by_backend={
        "ROCM_ATTN": 400, "TRITON_ATTN": 2500, "AITER_MLA": 2500})
    with pytest.raises(BudgetExceeded):
        run_matrix(expand(load_matrix(THREE_CELL_YAML)), producer, store,
                   GpuBudget(cap_hours=0.8), estimate_seconds=300)
    assert len(store.load()) == 2


def test_failed_cells_are_recorded_and_still_charged(tmp_store_dir):
    """A cell that ran for half an hour and then OOM'd cost real money."""
    store = RecordStore(tmp_store_dir)
    budget = GpuBudget(cap_hours=10)
    run_matrix(expand(load_matrix(YAML)), FakeProducer(failures={"ROCM_ATTN"},
                                                      cell_seconds=1800),
               store, budget, estimate_seconds=300)
    by_backend = {r.config["backend"]: r for r in store.load()}
    assert by_backend["ROCM_ATTN"].ok is False
    assert by_backend["TRITON_ATTN"].ok is True
    assert budget.spent_hours == pytest.approx(1.0)      # both cells charged


def test_budget_charges_cell_time_not_generate_time(tmp_store_dir):
    budget = GpuBudget(cap_hours=10)
    run_matrix(expand(load_matrix(ONE_CELL_YAML)),
               FakeProducer(seconds=2.0, cell_seconds=600.0),
               RecordStore(tmp_store_dir), budget, estimate_seconds=300)
    assert budget.spent_hours == pytest.approx(600 / 3600)


def test_resume_after_budget_stop_completes_the_rest(tmp_store_dir):
    cells = expand(load_matrix(YAML))
    with pytest.raises(BudgetExceeded):
        run_matrix(cells, FakeProducer(cell_seconds=1800), RecordStore(tmp_store_dir),
                   GpuBudget(cap_hours=0.75), estimate_seconds=300)
    assert run_matrix(cells, FakeProducer(cell_seconds=1), RecordStore(tmp_store_dir),
                      GpuBudget(cap_hours=10), estimate_seconds=300) == 1
    assert len(RecordStore(tmp_store_dir).load()) == 2
