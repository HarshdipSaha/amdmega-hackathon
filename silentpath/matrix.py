"""The sweep is data, not code, so it can shrink to fit the credit budget."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import yaml


@dataclass(frozen=True)
class Cell:
    backend: str
    workload_id: str
    prompt: str
    config: dict[str, Any]


def load_matrix(text: str) -> dict[str, Any]:
    return yaml.safe_load(text)


def expand(matrix: dict[str, Any]) -> list[Cell]:
    backends = matrix.get("backends") or []
    workloads = matrix.get("workloads") or []
    if not backends:
        raise ValueError("matrix must define at least one backend")
    if not workloads:
        raise ValueError("matrix must define at least one workload")

    seen: set[str] = set()
    for w in workloads:
        if w["id"] in seen:
            raise ValueError(f"duplicate workload id: {w['id']}")
        seen.add(w["id"])

    base = {k: v for k, v in matrix.items() if k not in ("backends", "workloads")}
    return [Cell(backend=b, workload_id=w["id"], prompt=w["prompt"],
                 config={**base, "backend": b,
                         "prompt_sha256": hashlib.sha256(w["prompt"].encode("utf-8")).hexdigest()})
            for b in backends for w in workloads]
