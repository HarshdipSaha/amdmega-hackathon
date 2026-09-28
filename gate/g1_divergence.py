"""G1 verdict calculation, intentionally independent of any GPU runtime."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

from silentpath.backends import backend_matches


class Verdict(str, Enum):
    DECISION_DIVERGENT = "DECISION-DIVERGENT"
    NUMERIC_ONLY = "NUMERIC-ONLY"
    IDENTICAL = "IDENTICAL"
    BLOCKED = "BLOCKED"


@dataclass
class Classification:
    verdict: Verdict
    max_abs_logprob_delta: float = 0.0
    divergent_pairs: list[tuple[str, str]] = field(default_factory=list)
    silent_fallbacks: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def classify(runs: list[dict]) -> Classification:
    """Classify successful same-prompt results without overstating unknown paths."""
    result = Classification(Verdict.BLOCKED)
    successful = [run for run in runs if run.get("ok")]
    for run in runs:
        requested, observed = run.get("backend_requested"), run.get("backend_observed")
        if requested and observed:
            match = backend_matches(str(requested), str(observed))
            if match is False:
                result.silent_fallbacks.append((str(requested), str(observed)))
            elif match is None:
                result.notes.append(f"unresolved path for {requested}: {observed}")
    if len(successful) < 2:
        result.notes.append(f"only {len(successful)} of {len(runs)} runs succeeded; not a result")
        return result

    numeric = decision = False
    for first, second in itertools.combinations(successful, 2):
        pair = (first["backend_requested"], second["backend_requested"])
        if first["text"] != second["text"] or first["token_ids"] != second["token_ids"]:
            decision = True
            result.divergent_pairs.append(pair)
            continue
        left, right = first["chosen_logprobs"], second["chosen_logprobs"]
        if len(left) != len(right):
            decision = True
            result.divergent_pairs.append(pair)
            continue
        for x, y in zip(left, right):
            delta = abs(x - y)
            result.max_abs_logprob_delta = max(result.max_abs_logprob_delta, delta)
            numeric = numeric or delta > 0.0
    result.verdict = Verdict.DECISION_DIVERGENT if decision else Verdict.NUMERIC_ONLY if numeric else Verdict.IDENTICAL
    return result
