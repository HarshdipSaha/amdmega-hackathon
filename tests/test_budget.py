import pytest

from silentpath.budget import BudgetExceeded, GpuBudget


def test_allows_spend_within_cap():
    b = GpuBudget(cap_hours=1.0, rate_per_hour=1.99)
    b.check_or_raise(estimated_seconds=600)
    b.record(600)
    assert b.spent_hours == pytest.approx(1 / 6)


def test_refuses_before_starting_a_cell_that_would_exceed():
    with pytest.raises(BudgetExceeded):
        GpuBudget(cap_hours=0.5).check_or_raise(estimated_seconds=3600)


def test_refusal_happens_before_any_spend_is_recorded():
    b = GpuBudget(cap_hours=0.5)
    with pytest.raises(BudgetExceeded):
        b.check_or_raise(estimated_seconds=3600)
    assert b.spent_hours == 0.0


def test_accumulates_across_cells():
    b = GpuBudget(cap_hours=1.0)
    b.record(1800)
    b.record(900)
    assert b.spent_hours == pytest.approx(0.75)
    with pytest.raises(BudgetExceeded):
        b.check_or_raise(estimated_seconds=1800)


def test_remaining_never_goes_negative():
    b = GpuBudget(cap_hours=1.0)
    b.record(7200)
    assert b.remaining_hours == 0.0


def test_estimated_cost_usd():
    b = GpuBudget(cap_hours=50.0, rate_per_hour=1.99)
    b.record(3600)
    assert b.spent_usd == pytest.approx(1.99)


def test_raise_if_over_fires_when_an_estimate_proved_too_low():
    """The pre-flight check acts on a guess; this is the backstop."""
    b = GpuBudget(cap_hours=1.0)
    b.check_or_raise(estimated_seconds=60)     # cheap by estimate
    b.record(7200)                              # expensive in reality
    with pytest.raises(BudgetExceeded):
        b.raise_if_over()


def test_raise_if_over_is_silent_within_cap():
    b = GpuBudget(cap_hours=1.0)
    b.record(1800)
    b.raise_if_over()


def test_seed_charges_prior_spend():
    """A resumed run must not re-arm the full cap."""
    b = GpuBudget(cap_hours=1.0)
    b.seed(1800)
    assert b.spent_hours == pytest.approx(0.5)
    with pytest.raises(BudgetExceeded):
        b.check_or_raise(estimated_seconds=2700)
