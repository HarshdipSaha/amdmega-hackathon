"""Render comparisons. Findings and noise are never merged into one number."""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any

from silentpath.compare import compare, is_finding
from silentpath.record import RunRecord

MIN_SAMPLES_FOR_SPEED_CLAIM = 2


def render_divergence_table(records: list[RunRecord]) -> list[dict[str, Any]]:
    """One row per comparable pair, grouped by workload. Failures excluded."""
    groups: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        if r.ok:
            groups[r.workload_id].append(r)

    rows: list[dict[str, Any]] = []
    for workload_id, group in sorted(groups.items()):
        for a, b in itertools.combinations(group, 2):
            d = compare(a, b)
            # Speed claims use generate() time, never cell time: cell time is
            # dominated by engine init and would compare startup, not kernels.
            a_sec = a.cost.device_seconds or a.cost.wall_seconds
            b_sec = b.cost.device_seconds or b.cost.wall_seconds

            # Orientation-independent: a 5.5x slowdown reads as 5.5 whichever
            # record happened to be stored first.
            if a_sec and b_sec:
                ratio = round(max(a_sec, b_sec) / min(a_sec, b_sec), 3)
                slower = (d.a_backend if a_sec >= b_sec else d.b_backend)
            else:
                ratio, slower = None, None

            samples_min = min(a.cost.samples, b.cost.samples)
            rows.append({
                "workload": workload_id,
                "a": d.a_backend, "b": d.b_backend,
                "level": d.level.value,
                "is_finding": is_finding(d),
                "max_abs_logprob_delta": d.max_abs_logprob_delta,
                "cost_ratio": ratio,
                "slower": slower,
                "samples_min": samples_min,
                "cost_ratio_admissible": (ratio is not None
                                          and samples_min >= MIN_SAMPLES_FOR_SPEED_CLAIM),
                "a_wall_stdev": a.cost.wall_stdev,
                "b_wall_stdev": b.cost.wall_stdev,
                "silent_fallback": a.path.is_silent_fallback or b.path.is_silent_fallback,
                "unresolved_path": a.path.is_unresolved or b.path.is_unresolved,
            })
    return rows


def findings_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["is_finding"]]
