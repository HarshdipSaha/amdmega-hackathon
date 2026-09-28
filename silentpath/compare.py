"""Three-level divergence, reported separately. Only the third is a finding."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from silentpath.record import RunRecord


class DivergenceLevel(str, Enum):
    IDENTICAL = "identical"   # same tokens, same logprobs, bit for bit
    NUMERIC = "numeric"       # logprobs differ; the decoded answer does not
    DECISION = "decision"     # the answer itself differs — the only reportable finding


class Divergence(BaseModel):
    level: DivergenceLevel
    # None when the two runs produced different numbers of tokens, so no
    # element-wise delta exists. float("inf") was used here in revision 2 and
    # reached json.dumps as bare `Infinity`, which is not valid JSON and which
    # the MCP tool surface then emitted to callers.
    max_abs_logprob_delta: float | None
    text_equal: bool
    decision_equal: bool | None
    a_id: str
    b_id: str
    a_backend: str | None
    b_backend: str | None


def compare(a: RunRecord, b: RunRecord) -> Divergence:
    if a.workload_id != b.workload_id:
        raise ValueError(
            f"cannot compare different workloads: {a.workload_id!r} vs {b.workload_id!r}")
    if not a.ok or not b.ok:
        raise ValueError("cannot compare a failed run; filter failures out first")

    text_equal = a.output.text == b.output.text
    tokens_equal = a.output.token_ids == b.output.token_ids
    decision_equal = (None if a.output.decision is None or b.output.decision is None
                      else a.output.decision == b.output.decision)

    la, lb = a.output.chosen_logprobs, b.output.chosen_logprobs
    # A zero-length pair contains no measurements, so a default value of 0.0
    # would claim equality without evidence. Equal nonempty vectors are measured;
    # unequal vectors have no element-wise maximum delta.
    max_delta = (max(abs(x - y) for x, y in zip(la, lb))
                 if la and len(la) == len(lb) else None)

    # Decision level dominates. An explicit decision mismatch counts even when
    # the surrounding text is identical.
    if decision_equal is False or not text_equal or not tokens_equal:
        level = DivergenceLevel.DECISION
    elif max_delta and max_delta > 0.0:
        level = DivergenceLevel.NUMERIC
    else:
        level = DivergenceLevel.IDENTICAL

    return Divergence(
        level=level, max_abs_logprob_delta=max_delta, text_equal=text_equal,
        decision_equal=decision_equal, a_id=a.record_id, b_id=b.record_id,
        a_backend=a.path.observed or a.path.requested,
        b_backend=b.path.observed or b.path.requested)


def is_finding(d: Divergence) -> bool:
    """Only decision-level divergence is a finding.

    Numeric divergence between different kernels is expected floating-point
    behaviour; reporting it as a discovery would be dishonest.
    """
    return d.level is DivergenceLevel.DECISION
