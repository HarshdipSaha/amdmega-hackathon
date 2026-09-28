"""The single seam. Producers emit RunRecords; everything else consumes them."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from silentpath.backends import backend_matches


class Confidence(str, Enum):
    CONFIRMED = "confirmed"   # two independent signals agree
    REPORTED = "reported"     # exactly one signal
    UNKNOWN = "unknown"       # no signal, or signals contradict each other


class ObservedPath(BaseModel):
    """What we asked the stack to do, and what it appears to have actually done.

    Separate fields on purpose: reading back the environment variable proves
    nothing, because the silent-fallback bug is precisely the case where the
    stack ignores it.
    """
    requested: str | None = None
    observed: str | None = None
    confidence: Confidence = Confidence.UNKNOWN
    signals: dict[str, str] = Field(default_factory=dict)

    @property
    def is_silent_fallback(self) -> bool:
        """True only on positive evidence of a different backend.

        `backend_matches` returns None when there is no evidence either way;
        that is reported through `is_unresolved`, never as a fallback.
        """
        if self.requested is None or self.observed is None:
            return False
        return backend_matches(self.requested, self.observed) is False

    @property
    def is_unresolved(self) -> bool:
        """We observed a path but cannot say whether it is the one requested."""
        if self.requested is None or self.observed is None:
            return True
        return backend_matches(self.requested, self.observed) is None


class CostSample(BaseModel):
    """Measured cost. Never estimated from token counts.

    `wall_seconds` times the generate() call only — never engine init or model
    load — and is what speed claims are made from. `cell_wall_seconds` is the
    whole cell including process start, engine init and model load, and is what
    the **budget is charged**; on a subprocess-per-cell design the two differ by
    orders of magnitude. `device_seconds` is elapsed event-region time from
    HIP events, not active GPU busy time. `peak_memory_bytes` is the torch
    allocated peak where available. `memory_sample_bytes` is a one-off AMD-SMI
    sample and is not a peak.
    """
    wall_seconds: float
    cell_wall_seconds: float | None = None
    device_seconds: float | None = None
    energy_joules: float | None = None
    peak_memory_bytes: int | None = None
    memory_sample_bytes: int | None = None
    samples: int = Field(ge=1)          # a timing with no repetitions is not admissible
    wall_stdev: float | None = None


class Output(BaseModel):
    text: str
    token_ids: list[int] = Field(default_factory=list)
    chosen_logprobs: list[float] = Field(default_factory=list)
    decision: str | None = None     # the extracted decision-level value, when the workload defines one


class RunRecord(BaseModel):
    record_id: str
    created_at: datetime
    config: dict[str, Any]
    workload_id: str
    path: ObservedPath
    cost: CostSample
    output: Output
    ok: bool = True
    error: str | None = None

    @staticmethod
    def compute_id(config: dict[str, Any], workload_id: str) -> str:
        """Identity is (what was requested, on what input) — never the result.

        If results contributed, a cached cell could never be recognised as
        already done, which is what makes resume work.
        """
        payload = json.dumps({"config": config, "workload_id": workload_id},
                             sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def build(cls, *, config: dict[str, Any], workload_id: str, path: ObservedPath,
              cost: CostSample, output: Output, ok: bool = True,
              error: str | None = None) -> RunRecord:
        return cls(record_id=cls.compute_id(config, workload_id),
                   created_at=datetime.now(timezone.utc), config=config,
                   workload_id=workload_id, path=path, cost=cost, output=output,
                   ok=ok, error=error)
