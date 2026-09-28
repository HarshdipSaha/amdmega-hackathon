"""Hard spend ceiling. $100 of credit is ~50 MI300X-hours and cannot be refilled."""
from __future__ import annotations


class BudgetExceeded(RuntimeError):
    pass


class GpuBudget:
    def __init__(self, cap_hours: float, rate_per_hour: float = 1.99) -> None:
        self.cap_hours = cap_hours
        self.rate_per_hour = rate_per_hour
        self._spent_seconds = 0.0

    @property
    def spent_hours(self) -> float:
        return self._spent_seconds / 3600.0

    @property
    def remaining_hours(self) -> float:
        return max(0.0, self.cap_hours - self.spent_hours)

    @property
    def spent_usd(self) -> float:
        return self.spent_hours * self.rate_per_hour

    def check_or_raise(self, estimated_seconds: float) -> None:
        """Refuse before starting. Never records anything."""
        if self.spent_hours + estimated_seconds / 3600.0 > self.cap_hours:
            raise BudgetExceeded(
                f"cell needs ~{estimated_seconds / 3600.0:.2f}h; "
                f"{self.remaining_hours:.2f}h of {self.cap_hours:.2f}h remain")

    def record(self, actual_seconds: float) -> None:
        self._spent_seconds += actual_seconds

    def raise_if_over(self) -> None:
        """Stop after the fact when an estimate proved too low.

        The pre-flight check can only ever act on a guess. Without this, a cell
        that costs far more than estimated silently blows the cap; with it, a
        bad estimate costs one cell rather than the whole credit.
        """
        if self.spent_hours > self.cap_hours:
            raise BudgetExceeded(
                f"spent {self.spent_hours:.2f}h against a {self.cap_hours:.2f}h cap; "
                f"the last cell cost more than estimated")

    def seed(self, seconds: float) -> None:
        """Charge prior spend, so a resumed run does not re-arm the full cap."""
        self._spent_seconds += seconds
