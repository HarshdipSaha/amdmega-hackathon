"""Producers are the only GPU-dependent part of the system."""
from __future__ import annotations

from typing import Protocol

from silentpath.matrix import Cell
from silentpath.record import RunRecord


class Producer(Protocol):
    def run(self, cell: Cell) -> RunRecord:
        """Execute one cell and return a record. Must not raise on inference
        failure — a backend that refuses to load is data, not an exception."""
        ...
