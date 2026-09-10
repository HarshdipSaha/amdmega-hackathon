# SILENTPATH — Core and D1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove or kill the claim that AMD's several attention backends produce different answers for the same input, then build the measurement core that turns that claim into a shippable tool — a probe that reports which backend your stack silently selected, what it cost, and whether the choice changed your output.

**Architecture:** One seam, the `RunRecord`. A *producer* runs one inference under one configuration and emits an immutable record describing what was requested, what path was actually observed, what it cost, and what came out. Every other module — comparator, store, runner, reporter, MCP server — consumes records and nothing else. This is what makes a GPU-dependent project testable on a laptop with no GPU: every consumer is tested against fixture records, and only the producer needs hardware.

**Tech Stack:** Python 3.12, `pydantic` v2 for the record schema, `pytest` for tests, `PyYAML` for the config matrix, `mcp` (FastMCP) for the D1 tool surface, `amdsmi` + torch HIP events for device telemetry, vLLM for the producer. Explicitly **not** rocprofiler-sdk.

**Spec:** `docs/SPEC.md` · **Status and gates:** `docs/STATUS.md` · **Council verdict:** `docs/council/verdict.md` · **Brief:** `docs/hackathon-brief.md`

**Plan 1 of a series.** Covers the three gates, the measurement core, and **D1** (probe + MCP server, program theme 1). D2–D6 get their own plans, written *after* the divergence gate returns, because D2's entire premise depends on its result.

**Revision note — this is revision 4.** Every revision was reviewed by an agent that reconstructed all modules and test files into a scratch venv and executed them, plus a mutation battery. That is the only review that finds this class of defect. Revision 1: 74 collected, **3 failed**. Revision 2: 94 passed but two headline fixes were still wrong. Revision 3: 121 passed, no blockers, seven mutation survivors. Revision 4 closes those.

- **Revision 1 → 2, blockers:** `run_matrix` never enforced the budget — it checked a fixed 300s estimate regardless of real cell cost, so with a 0.75h cap and 1800s cells it ran both cells and **overspent the cap by 33% without raising**, failing two of its own tests. And `is_silent_fallback` was a substring test that was wrong in *both* directions: `"A" in "TRITON_ATTN"` reported no fallback, while every realistic pair (`requested="ROCM_ATTN"`, `observed="ROCmFlashAttentionBackend"`) reported a fallback, because the env-var token is never a substring of the backend class name. D1's headline MCP tool would have reported **100% silent fallback on the first real sweep** — a false positive on the project's second-headline claim, published without anyone noticing.
- **Revision 1 → 2, significant:** `cost_ratio` was order-dependent, so a genuine 5.5x slowdown printed as a 0.18x speedup purely from store insertion order. The whole telemetry module was **dead code** — nothing imported it, no producer populated `device_seconds`, so the spec's central "measured GPU-seconds" claim shipped as wall clock. Report rows carried no repetition count or variance despite the spec forbidding single-shot speed claims. `derive_path` over-confirmed, declaring `ROCmFlashAttentionBackend` and `Flash Attention` to be the same backend. And the gate ordering was wrong: G1 needed an environment that G2 is what determines.
- **Revision 1 → 2, vacuous tests:** two tests *named* for the numeric-noise case used bit-identical fixtures and never exercised `NUMERIC` at all. Sabotaging `is_finding` to count numeric divergence — exactly the "report floating-point noise as a discovery" failure the spec forbids — was caught by only **one** test. Both are now genuinely numeric.

**Revision 2 → 3.** Reviewed again by reconstruct-and-run: 94 passed, 0 failed, and the per-task test counts were exact — but a mutation battery and targeted probes found the two headline fixes were still wrong.

- **Blockers.** `backend_matches("ROCM_ATTN", "ROCM_ATTN")` returned **False**, because the normalised env token was not in its own alias set. `FakeProducer` sets `observed = requested`, so the plan's own GPU-free demo reported `"silent_fallbacks": 4` on 4 records — revision 1's headline defect reproduced in the demo a judge would watch, and Task 14's own sanity check failed before any GPU was involved. The budget fix was also still unsound: the *first* cell was never bounded at all (a single 7,200s cell against a 0.5h cap raised nothing and overspent 4x), and a mean-based estimate lags a rising cost curve. And the budget metered the wrong quantity entirely — `wall_seconds` covers `generate()` only, so a subprocess-per-cell design charged the credit seconds while engine init and model load, which dominate, were free; a cell that burned 30 minutes and OOM'd cost **zero**.
- **Significant.** Two mutations survived the whole suite: deleting the `a_wall_stdev`/`b_wall_stdev` columns, and removing the admissibility gate on `worst_cost_ratio` — revision 2's own headline additions were untested. `BACKEND_LOG_RE` matched `"Using the default attention backend selector"`, so real vLLM stderr always yielded ≥2 names, `backend_observed` became `None`, and G1's fallback check was **skipped entirely on the gate run**. The prefix rule rejected qualifier-bearing log lines (`"Triton Attention (V1)"`, `"V1 Triton Attention"`) as disagreements, so `CONFIRMED` would rarely be reached. And the alias table was asymmetrically safe: silent on unknown keys, loud on unknown values, asserting a fallback on no evidence in one direction while refusing to in the other.
- **Also:** the ⛔ gate blocked Tasks 5–16, which are GPU-free and which no verdict branch actually cancels; `float("inf")` reached `json.dumps` as bare `Infinity`, which is not valid JSON; and the CLI built a fresh `GpuBudget` per invocation, so a resume re-armed the full cap and two runs under a declared 0.75h cap spent 1.0h.

Five lessons carried forward. **A budget check against a constant is not a budget check**, and neither is one that only looks backwards — the first cell must be bounded too, and a mis-estimate must cost one cell rather than the whole cap. **Meter the quantity you are actually billed for**, not the one you are reporting. **A substring match between an env-var token and a class name is never a valid identity test** — and the first case to test is the trivial one, `X` against `X`. **Absence of evidence must be represented explicitly**, or it gets asserted as evidence in whichever direction the code happens to fall through. And **a fix is not verified by a green suite** — only by a mutation that the suite catches.

---

## ⛔ Gate

**Task 3 is the go/no-go gate for the D2 plan. Do not begin D2 until it returns a verdict.** The decision rule is fixed *before* any numbers exist, so it cannot be rationalised afterwards.

Tasks 4–16 are **not** blocked by the gate. They are GPU-free, they are the whole of D1, and no verdict branch cancels them — every branch below keeps D1 alive. Blocking pure-Python work behind a Linux/ROCm gate on a Windows machine would contradict the architecture and buy nothing. Run them in parallel with the gate.

| Task 3 verdict | Meaning | What happens |
|---|---|---|
| **DECISION-DIVERGENT** | At least one input produces a different decoded output across backends | Full plan proceeds. This is the headline result |
| **NUMERIC-ONLY** | Logprobs differ but every decoded output is identical | Proceed, but the thesis narrows to cost. D2 becomes a cost demo, not a correctness demo. Report the null honestly — still the first such measurement on AMD |
| **IDENTICAL** | Bit-identical across all backends | The correctness half is dead. Keep D1/D3/D4/D5 on the cost thesis alone, which stands on documented 5.5x evidence. Do not manufacture a finding |
| **BLOCKED** | Fewer than two backends actually run | Not a result. Escalate to the cloud droplet before concluding anything |

A null result is a legitimate outcome and must be published as one. The single failure mode this plan most guards against is reporting expected floating-point noise as a discovery.

## Where this tree lives

Everything below is created at the repository root, **`H:/augsepthacks/amd`**, alongside the existing `docs/`. The Python package is `silentpath/`. There is no `src/` layer.

## Platform reality — read before Task 2

The development machine runs **Windows 11**. vLLM and SGLang on ROCm are **Linux-only**. Therefore:

- **Pure-Python tasks (1, 4–12, 14, 15, 16) run anywhere**, including Windows. They need no GPU and no ROCm. This is the whole point of the seam.
- **GPU tasks (2, 3, and Task 14 Step 5) must run under WSL2 with ROCm, or on the AMD Developer Cloud droplet.** Task 13 *writes* the vLLM producer but its tests are pure-Python; only running it needs hardware.
- **Task 2 (G2) runs before Task 3 (G1)** and decides which of those two environments G1 uses. Running the gate in an unverified environment makes a `BLOCKED` verdict indistinguishable from "no ROCm installed."
- Do not install vLLM on native Windows. It will not work and the failure is confusing.

## Facts this plan hard-codes

| Fact | Value | Verified? |
|---|---|---|
| Backend selection env var | `VLLM_ATTENTION_BACKEND` | Yes — long-standing vLLM interface |
| Env var is read at **engine init**, not per request | Each backend needs a **fresh process** | Yes — consequence of vLLM's startup selection |
| Candidate ROCm backends | `ROCM_ATTN`, `ROCM_AITER_FA`, `TRITON_ATTN`, `TRITON_MLA`, `AITER_MLA` | **Partly** — the accepted set varies by vLLM version. The plan *discovers* which run rather than assuming |
| Backend announced in logs | A line matching `Using <name> backend` on stderr, excluding the `the default …` selector line | **Partly.** The `(?!the default\b)` lookahead is already in the regex because that line otherwise matches on every run and yields a phantom backend. **Task 3 Step 9 confirms the rest against real stderr and re-classifies** |
| Env-var token ↔ backend class name | **Not a substring relationship.** Requires the alias table in `silentpath/backends.py` | Task 3 Step 9 extends it from real stderr |
| Determinism knobs | `temperature=0.0`, fixed `seed`, `enforce_eager=True` | Yes. `enforce_eager` disables HIP graphs, which otherwise confound backend attribution |
| Telemetry | `amdsmi` Python package; torch HIP events for device time | Yes — `amdsmi` ships an official Python API; `torch.cuda.Event` maps to HIP events on ROCm |
| Timing boundary | `wall_seconds` times **only `generate()`**, never engine init or model load | Yes — the timing loop sits after `LLM()` construction |
| Not used | rocprofiler-sdk | Deliberate. No first-class Python binding; LD_PRELOAD C++ injection costs ~2 weeks for a hackathon-grade need |

## Review protocol for this plan

Any reviewer dispatched at this document must be told to **reconstruct the modules from the code blocks and execute the test files**, naming the specific files to run — not to read and comment. Give the reviewer this revision history so it verifies the fixes landed rather than re-deriving them. Budget at least two rounds.

---

## Task 1: Repository scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`
- Create: `silentpath/__init__.py`, `tests/__init__.py`, `tests/conftest.py`, `gate/__init__.py`

- [ ] **Step 1: Initialise the repository**

```bash
cd "H:/augsepthacks/amd"
git status 2>&1 | head -3      # expect: fatal: not a git repository
git init
git config user.email          # expect: harshdipsaha@gmail.com
```

If `git config user.email` prints nothing, stop and ask. Do **not** pass `-c user.email` on commits — the global config is already correct and overriding it has previously stamped commits with the wrong author.

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "silentpath"
version = "0.1.0"
description = "Which path did your AMD GPU silently take, what did it cost you, and did it change your answer?"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
gpu = ["amdsmi"]
mcp = ["mcp>=1.2"]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["silentpath*", "gate*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

vLLM is deliberately **not** a dependency — Linux/ROCm-only, and it would make the package uninstallable on the development machine. Tasks 3 and 13 install it separately in the Linux environment. `numpy` is not needed: all comparison is over plain Python floats.

- [ ] **Step 3: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
*.egg-info/
runs/*
!runs/.gitkeep
.playwright-mcp/
```

`runs/*` rather than `runs/`: git cannot re-include a file underneath an *excluded directory*, so `runs/` plus a negation silently ignores `.gitkeep` too, and `runs/` then does not exist in a fresh clone.

- [ ] **Step 4: Create package and test `__init__.py` files**

```bash
mkdir -p silentpath tests gate runs configs prompts
touch silentpath/__init__.py tests/__init__.py gate/__init__.py runs/.gitkeep
```

`tests/__init__.py` prevents basename collisions between test modules as the suite grows — with two same-named test files in different directories and no package marker, pytest's rootdir import raises on collection. It is cheap insurance, not a current bug.

- [ ] **Step 5: Create `tests/conftest.py`**

```python
"""Shared fixtures. Nothing in tests/ requires a GPU."""
from __future__ import annotations

import pytest


@pytest.fixture
def tmp_store_dir(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    return d
```

- [ ] **Step 6: Install and verify collection**

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; on Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Expected: `collected 0 items`, exit code 5. Not an error.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore silentpath tests gate runs configs prompts docs
git commit -m "chore: scaffold silentpath package and test harness"
```

---

## Task 2: G2 — local hardware viability *(run before the gate)*

This runs first because it determines **where** Task 3 executes.

**Files:**
- Create: `gate/g2_hardware.py`, `tests/test_g2_parse.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_g2_parse.py`

```python
from gate.g2_hardware import parse_rocminfo

SAMPLE = """
*******
Agent 1
*******
  Name:                    AMD Ryzen 9 7940HS
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1103
  Marketing Name:          AMD Radeon 780M
  Device Type:             GPU
"""


def test_extracts_gpu_gfx_targets_only():
    assert parse_rocminfo(SAMPLE) == ["gfx1103"]


def test_empty_output_yields_no_targets():
    assert parse_rocminfo("") == []


def test_cpu_only_output_yields_no_targets():
    assert parse_rocminfo("Agent 1\n  Name:  AMD Ryzen\n  Device Type:  CPU\n") == []
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_g2_parse.py -v
```
Expected: `ModuleNotFoundError: No module named 'gate.g2_hardware'`.

- [ ] **Step 3: Implement `gate/g2_hardware.py`**

```python
"""G2 — can the local AMD hardware run any of this?

The development machine is Windows 11 and vLLM/SGLang on ROCm are Linux-only,
so a local path exists only if ROCm under WSL2 sees a supported gfx target.
"""
from __future__ import annotations

import re
import shutil
import subprocess

AGENT_SPLIT = re.compile(r"\n(?=\*{3,}\s*\nAgent )")
NAME_RE = re.compile(r"^\s*Name:\s*(\S+)", re.MULTILINE)
TYPE_RE = re.compile(r"^\s*Device Type:\s*(\w+)", re.MULTILINE)


def parse_rocminfo(text: str) -> list[str]:
    """Return the gfx targets of GPU agents, in order, ignoring CPU agents."""
    targets: list[str] = []
    for block in AGENT_SPLIT.split(text):
        dev_type = TYPE_RE.search(block)
        if not dev_type or dev_type.group(1).upper() != "GPU":
            continue
        for name in NAME_RE.findall(block):
            if name.startswith("gfx"):
                targets.append(name)
                break
    return targets


def main() -> int:
    if shutil.which("rocminfo") is None:
        print("[g2] rocminfo not found. Run inside WSL2 with ROCm installed.")
        print("[g2] VERDICT: BLOCKED — no local ROCm. GPU work must use the cloud droplet.")
        return 1
    out = subprocess.run(["rocminfo"], capture_output=True, text=True).stdout
    targets = parse_rocminfo(out)
    print(f"[g2] GPU agents: {targets or 'none'}")
    if not targets:
        print("[g2] VERDICT: BLOCKED — ROCm sees no GPU agent.")
        return 1
    print("[g2] VERDICT: OK — local GPU visible. Confirm vLLM imports next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/test_g2_parse.py -v      # all green
```

- [ ] **Step 5: Run G2 for real, under WSL2**

```bash
wsl
python -m gate.g2_hardware
python -c "import vllm; print('vllm ok', vllm.__version__)"
```

Verify the parser against genuine output, since the test fixture is hand-written:

```bash
rocminfo > /tmp/r.txt && python -c "from gate.g2_hardware import parse_rocminfo; print(parse_rocminfo(open('/tmp/r.txt').read()))"
```

Record the outcome in `docs/STATUS.md`. If BLOCKED, every GPU task moves to the AMD Developer Cloud droplet and the run matrix must shrink to fit ~50 GPU-hours — but **still do not claim the credit until Task 14 Step 5 runs unattended end to end.**

- [ ] **Step 6: Commit**

```bash
git add gate/g2_hardware.py tests/test_g2_parse.py docs/STATUS.md
git commit -m "feat(gate): add G2 local ROCm hardware probe"
```

---

## Task 3: ⛔ G1 — the divergence kill-test

The gate. Answers one question with the least possible machinery: **do different attention backends produce different outputs for the same input?** No profiler, no telemetry, none of the framework built later.

**Files:**
- Create: `silentpath/backends.py`, `gate/g1_worker.py`, `gate/g1_divergence.py`
- Create: `tests/test_backends.py`, `tests/test_g1_compare.py`

- [ ] **Step 1: Write the failing tests for backend identity** — `tests/test_backends.py`

This module is created here, before `record.py`, because both the gate and the core need it.

```python
import pytest

from silentpath.backends import backend_matches, names_agree, observed_backends_in


def test_identical_requested_and_observed_is_always_a_match():
    """The trivial case, and the one revision 2 got wrong: the normalised env
    token was not in its own alias set, so X vs X reported a fallback and the
    GPU-free demo showed 100% false positives."""
    for token in ("ROCM_ATTN", "TRITON_ATTN", "ROCM_AITER_FA", "TRITON_MLA", "AITER_MLA"):
        assert backend_matches(token, token) is True


def test_known_alias_matches_its_env_token():
    assert backend_matches("ROCM_ATTN", "ROCmFlashAttentionBackend") is True


def test_recognised_but_wrong_backend_is_a_mismatch():
    assert backend_matches("ROCM_ATTN", "TritonAttentionBackend") is False


def test_matching_is_case_and_punctuation_insensitive():
    assert backend_matches("triton_attn", "Triton Attention Backend") is True


def test_unknown_env_token_yields_no_evidence():
    """Absence of an alias entry is absence of data, not evidence of fallback."""
    assert backend_matches("SOME_FUTURE_BACKEND", "TritonAttentionBackend") is None


def test_unrecorded_observed_name_yields_no_evidence():
    """The converse must be equally cautious: a name we have never recorded is
    not proof of a fallback either. Revision 2 was loud here and silent above."""
    assert backend_matches("ROCM_ATTN", "SomeBrandNewBackendV9") is None


@pytest.mark.parametrize("intro,log", [
    ("TritonAttentionBackend", "Triton Attention"),
    ("TritonAttentionBackend", "Triton Attention (V1)"),
    ("TritonAttentionBackend", "V1 Triton Attention"),
    ("ROCmFlashAttentionBackend", "ROCm Flash Attention"),
    ("AiterFlashAttentionBackend", "Aiter Flash Attention v3"),
])
def test_qualified_names_still_agree(intro, log):
    assert names_agree(intro, log) is True


def test_a_shorter_lookalike_does_not_agree():
    """`Flash Attention` is a substring of `ROCmFlashAttentionBackend` but names
    a different backend."""
    assert names_agree("ROCmFlashAttentionBackend", "Flash Attention") is False


def test_different_versions_are_different_backends():
    """Both reduce to 'flashattention' once qualifiers are stripped, so without
    the version guard these merge — and FA2 vs FA3 is a real vLLM distinction."""
    assert names_agree("FlashAttentionV2Backend", "FlashAttentionV3Backend") is False
    assert names_agree("ROCmAttentionV2", "ROCmAttentionV3") is False


def test_a_name_stripped_to_nothing_does_not_match_everything():
    """Qualifier stripping can reduce a short name to one character, after which
    a prefix rule matches almost anything."""
    assert names_agree("AV1", "AiterFlashAttentionBackend") is False
    assert names_agree("XV1Backend", "XLAAttentionBackend") is False


def test_default_selector_line_is_not_read_as_a_backend_name():
    """vLLM logs this on every run; without the negative lookahead it yields a
    phantom backend called 'the default attention' and blinds the detector."""
    assert observed_backends_in("Using the default attention backend selector\n") == []


def test_single_log_line_is_extracted():
    assert observed_backends_in("INFO Using Triton Attention backend.\n") == ["Triton Attention"]


def test_distinct_log_lines_are_all_returned():
    text = "Using Triton Attention backend.\nUsing ROCm Flash Attention backend.\n"
    assert len(observed_backends_in(text)) == 2


def test_repeated_identical_lines_collapse():
    text = "Using Triton Attention backend.\nUsing Triton Attention backend.\n"
    assert observed_backends_in(text) == ["Triton Attention"]


def test_no_match_returns_empty():
    assert observed_backends_in("nothing relevant here") == []
```

- [ ] **Step 2: Implement `silentpath/backends.py`**

```python
"""Backend identity. The env-var token and the runtime class name are different
vocabularies, so mapping between them needs an explicit table, never a substring
test. `ROCM_ATTN` is not a substring of `ROCmFlashAttentionBackend`, and `A` *is*
a substring of `TRITON_ATTN` — a substring rule is wrong in both directions.

Three-valued on purpose. `True` = same backend, `False` = genuinely different,
`None` = no evidence either way. Collapsing `None` into either boolean asserts a
conclusion on no data, and which way it falls decides whether the project
publishes false positives or hides real fallbacks.

EXTEND `BACKEND_ALIASES` FROM REAL STDERR in Task 3 Step 9. The entries below are
a starting guess and are explicitly not authoritative.
"""
from __future__ import annotations

import re

# The negative lookahead is load-bearing. vLLM logs "Using the default attention
# backend selector", which otherwise matches and yields a phantom backend named
# "the default attention" on every single run.
BACKEND_LOG_RE = re.compile(r"Using (?!the default\b)(.+?) backend", re.IGNORECASE)

BACKEND_ALIASES: dict[str, set[str]] = {
    "ROCM_ATTN": {"rocmattentionbackend", "rocmflashattentionbackend", "rocmflashattention"},
    "ROCM_AITER_FA": {"aiterflashattentionbackend", "rocmaiterflashattentionbackend"},
    "TRITON_ATTN": {"tritonattentionbackend", "tritonattention"},
    "TRITON_MLA": {"tritonmlabackend", "tritonmla"},
    "AITER_MLA": {"aitermlabackend", "aitermla"},
    "FLASH_ATTN": {"flashattentionbackend", "flashattention"},
}

ALL_KNOWN_NAMES: set[str] = {n for names in BACKEND_ALIASES.values() for n in names}

# Role suffixes that decorate a backend name without changing which backend it
# is: "Triton Attention" and "TritonAttentionBackend" are one thing.
QUALIFIER_TOKENS = ("backend", "impl")

# Version markers are stripped too, but only after checking that the two names
# do not carry *different* versions — FlashAttention V2 and V3 are genuinely
# different backends and must never be merged.
VERSION_TOKENS = ("v1", "v2", "v3")
VERSION_RE = re.compile(r"v[123]")

# Below this length a stripped name carries too little information for a prefix
# match to mean anything: "x" would match almost every backend.
MIN_COMPARABLE_LENGTH = 3


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def strip_qualifiers(normalised: str) -> str:
    """Remove leading and trailing version/role tokens from a normalised name."""
    changed = True
    while changed:
        changed = False
        for token in (*QUALIFIER_TOKENS, *VERSION_TOKENS):
            if normalised.endswith(token) and len(normalised) > len(token):
                normalised = normalised[: -len(token)]
                changed = True
            if normalised.startswith(token) and len(normalised) > len(token):
                normalised = normalised[len(token):]
                changed = True
    return normalised


def names_agree(a: str, b: str) -> bool:
    """Do two backend names, from different sources, denote the same backend?

    Three guards, each earning its place:

    1. If both names carry a version marker and the markers differ, they are
       different backends. FlashAttention V2 and V3 both reduce to
       "flashattention" once qualifiers are stripped, and merging them would
       confirm a genuine disagreement as agreement.
    2. Prefix, not substring: `Flash Attention` is a substring of
       `ROCmFlashAttentionBackend` but names a different backend.
    3. A name stripped below MIN_COMPARABLE_LENGTH carries too little
       information to prefix-match on — "x" would match nearly everything — so
       fall back to requiring exact equality of the unstripped names.
    """
    na, nb = normalise(a), normalise(b)
    va, vb = set(VERSION_RE.findall(na)), set(VERSION_RE.findall(nb))
    if va and vb and va != vb:
        return False

    x, y = strip_qualifiers(na), strip_qualifiers(nb)
    if len(x) < MIN_COMPARABLE_LENGTH or len(y) < MIN_COMPARABLE_LENGTH:
        return na == nb
    return x.startswith(y) or y.startswith(x)


def backend_matches(requested: str, observed: str) -> bool | None:
    """Does `observed` name the backend that `requested` asked for?

    True  — same backend.
    False — a different, recognised backend: a genuine silent fallback.
    None  — no evidence. Either the requested token has no alias entry, or the
            observed name is one we have never recorded. Claiming a fallback
            here is how a false headline gets published; claiming a match here
            is how a real one gets hidden. Callers must handle None explicitly.
    """
    req_norm, obs_norm = normalise(requested), normalise(observed)
    if req_norm == obs_norm:
        return True                       # the trivial case, and the one rev 2 got wrong

    aliases = BACKEND_ALIASES.get(requested.upper().strip())
    if aliases is None:
        return None                       # unrecorded env token
    if obs_norm in aliases:
        return True
    if obs_norm in ALL_KNOWN_NAMES:
        return False                      # a recognised name, but the wrong backend
    return None                           # unrecorded runtime name


def observed_backends_in(stderr: str) -> list[str]:
    """Every distinct backend named in a log, in order of first appearance.

    Returns all of them rather than the last: vLLM prints several backend lines
    per run (attention, MLA, per-worker), so picking one silently would be a
    guess. Callers treat more than one as an unresolved observation.
    """
    found = (m.strip() for m in BACKEND_LOG_RE.findall(stderr or ""))
    return list(dict.fromkeys(f for f in found if f))
```

- [ ] **Step 3: Write the failing tests for the gate classifier** — `tests/test_g1_compare.py`

```python
from gate.g1_divergence import Verdict, classify


def _run(text, ids, logprobs, backend="ROCM_ATTN", observed=None, ok=True):
    return {
        "backend_requested": backend,
        "backend_observed": observed if observed is not None else backend,
        "ok": ok, "text": text, "token_ids": ids, "chosen_logprobs": logprobs,
    }


def test_identical_runs_are_identical():
    a = _run("42 dollars", [1, 2], [-0.5, -0.25], "ROCM_ATTN")
    b = _run("42 dollars", [1, 2], [-0.5, -0.25], "TRITON_ATTN")
    assert classify([a, b]).verdict is Verdict.IDENTICAL


def test_logprob_difference_with_same_text_is_numeric_only():
    a = _run("42 dollars", [1, 2], [-0.5, -0.25], "ROCM_ATTN")
    b = _run("42 dollars", [1, 2], [-0.5000001, -0.25], "TRITON_ATTN")
    assert classify([a, b]).verdict is Verdict.NUMERIC_ONLY


def test_different_text_is_decision_divergent():
    a = _run("42 dollars", [1, 2], [-0.5, -0.25], "ROCM_ATTN")
    b = _run("47 dollars", [1, 3], [-0.5, -0.25], "TRITON_ATTN")
    assert classify([a, b]).verdict is Verdict.DECISION_DIVERGENT


def test_different_token_count_is_decision_divergent():
    a = _run("42", [1], [-0.5], "ROCM_ATTN")
    b = _run("42 dollars", [1, 2], [-0.5, -0.25], "TRITON_ATTN")
    assert classify([a, b]).verdict is Verdict.DECISION_DIVERGENT


def test_fewer_than_two_successful_runs_is_blocked():
    a = _run("42", [1], [-0.5], "ROCM_ATTN")
    b = _run("", [], [], "TRITON_ATTN", ok=False)
    assert classify([a, b]).verdict is Verdict.BLOCKED


def test_decision_divergence_wins_over_numeric():
    """Regardless of pair iteration order, the strongest finding is reported."""
    a = _run("42 dollars", [1, 2], [-0.5, -0.25], "ROCM_ATTN")
    b = _run("42 dollars", [1, 2], [-0.5000001, -0.25], "TRITON_ATTN")
    c = _run("47 dollars", [1, 3], [-0.5, -0.25], "AITER_MLA")
    assert classify([a, b, c]).verdict is Verdict.DECISION_DIVERGENT


def test_alias_match_is_not_reported_as_a_silent_fallback():
    """The realistic case: the class name never contains the env token."""
    runs = [_run("42", [1], [-0.5], "ROCM_ATTN", observed="ROCmFlashAttentionBackend"),
            _run("42", [1], [-0.5], "TRITON_ATTN", observed="TritonAttentionBackend")]
    assert classify(runs).silent_fallbacks == []


def test_genuine_fallback_is_reported():
    runs = [_run("42", [1], [-0.5], "ROCM_ATTN", observed="TritonAttentionBackend"),
            _run("42", [1], [-0.5], "TRITON_ATTN", observed="TritonAttentionBackend")]
    assert classify(runs).silent_fallbacks == [("ROCM_ATTN", "TritonAttentionBackend")]
```

- [ ] **Step 4: Run both test files**

```bash
pytest tests/test_backends.py tests/test_g1_compare.py -v
```
Expected: `test_backends.py` **passes (15)** — Step 2 just implemented it — while `test_g1_compare.py` errors on the missing `gate.g1_divergence`, which Step 6 creates.

- [ ] **Step 5: Write `gate/g1_worker.py`**

One backend per process, because vLLM reads `VLLM_ATTENTION_BACKEND` at engine init. The worker writes JSON; the parent captures stderr separately, which is far more robust than redirecting vLLM's logging from inside the process.

```python
"""Run one prompt under one attention backend and dump the result as JSON.

Usage: python -m gate.g1_worker <model> <prompt_file> <out_json>
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

MAX_TOKENS = 64


def main() -> int:
    model, prompt_file, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    result = {
        "backend_requested": os.environ.get("VLLM_ATTENTION_BACKEND", ""),
        "backend_observed": None, "ok": False, "error": None,
        "text": "", "token_ids": [], "chosen_logprobs": [],
    }
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(model=model, seed=0, enforce_eager=True,
                  max_model_len=2048, gpu_memory_utilization=0.85)
        params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=1, seed=0)
        out = llm.generate([Path(prompt_file).read_text(encoding="utf-8")], params)[0].outputs[0]

        token_ids = list(out.token_ids)
        chosen: list[float] = []
        if out.logprobs:
            for tok, step in zip(token_ids, out.logprobs):
                e = step.get(tok)
                chosen.append(float(getattr(e, "logprob", e)) if e is not None else float("nan"))

        result.update(ok=True, text=out.text, token_ids=token_ids,
                      chosen_logprobs=chosen, backend_observed=introspect_backend(llm))
    except Exception as exc:      # a backend that refuses to load is data, not a crash
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)

    Path(out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


def introspect_backend(llm) -> str | None:
    """Best-effort: ask the engine what it actually built. vLLM internals move
    between releases, so every step is defensive."""
    for path in (
        "llm_engine.model_executor.driver_worker.model_runner.attn_backend",
        "llm_engine.model_executor.driver_worker.worker.model_runner.attn_backend",
    ):
        obj = llm
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return getattr(obj, "__name__", type(obj).__name__)
        except AttributeError:
            continue
    return None


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Write `gate/g1_divergence.py`**

```python
"""G1 — the divergence kill-test.

Runs the same prompt under each backend in a fresh process and classifies the
outcome. The verdict thresholds were fixed before any numbers existed.

Usage:
    python -m gate.g1_divergence --model <hf-id> --prompt-file prompts/g1.txt \\
        --backends ROCM_ATTN TRITON_ATTN --out runs/g1
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from silentpath.backends import backend_matches, observed_backends_in


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
    """Classify single-backend runs of the same prompt.

    Decision-level divergence dominates numeric divergence: the strongest
    finding across all pairs is reported, not the first one encountered.
    """
    ok = [r for r in runs if r.get("ok")]
    c = Classification(verdict=Verdict.BLOCKED)

    for r in runs:
        req, obs = r.get("backend_requested"), r.get("backend_observed")
        if not (req and obs):
            continue
        match = backend_matches(req, str(obs))
        if match is False:
            c.silent_fallbacks.append((req, str(obs)))
        elif match is None:
            c.notes.append(f"unresolved path for {req}: observed {obs!r} is unrecorded")

    if len(ok) < 2:
        c.notes.append(f"only {len(ok)} of {len(runs)} runs succeeded; not a result")
        return c

    decision_divergent = numeric_divergent = False
    for a, b in itertools.combinations(ok, 2):
        pair = (a["backend_requested"], b["backend_requested"])
        if a["text"] != b["text"] or a["token_ids"] != b["token_ids"]:
            decision_divergent = True
            c.divergent_pairs.append(pair)
            continue
        la, lb = a["chosen_logprobs"], b["chosen_logprobs"]
        if len(la) != len(lb):
            decision_divergent = True
            c.divergent_pairs.append(pair)
            continue
        for x, y in zip(la, lb):
            delta = abs(x - y)
            c.max_abs_logprob_delta = max(c.max_abs_logprob_delta, delta)
            if delta > 0.0:
                numeric_divergent = True

    c.verdict = (Verdict.DECISION_DIVERGENT if decision_divergent
                 else Verdict.NUMERIC_ONLY if numeric_divergent
                 else Verdict.IDENTICAL)
    return c


def run_backend(model: str, prompt_file: Path, backend: str, out_dir: Path) -> dict:
    out_json = out_dir / f"{backend}.json"
    env = {**os.environ, "VLLM_ATTENTION_BACKEND": backend}
    proc = subprocess.run(
        [sys.executable, "-m", "gate.g1_worker", model, str(prompt_file), str(out_json)],
        env=env, capture_output=True, text=True, timeout=1800)
    (out_dir / f"{backend}.stderr.txt").write_text(proc.stderr, encoding="utf-8")

    result = (json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists()
              else {"backend_requested": backend, "backend_observed": None, "ok": False,
                    "error": f"worker exited {proc.returncode} without writing output",
                    "text": "", "token_ids": [], "chosen_logprobs": []})

    if not result.get("backend_observed"):
        names = observed_backends_in(proc.stderr)
        result["backend_observed"] = names[0] if len(names) == 1 else None
        if len(names) > 1:
            result.setdefault("log_backends", names)
    return result


def reclassify(out_dir: Path) -> list[dict]:
    """Rebuild the run list from saved JSON and stderr, with no GPU.

    Step 9 corrects the alias table after the real run; re-deriving the verdict
    must not cost another 50-GPU-hour-budget sweep.
    """
    runs = []
    for json_path in sorted(out_dir.glob("*.json")):
        if json_path.name == "verdict.json":
            continue
        result = json.loads(json_path.read_text(encoding="utf-8"))
        stderr_path = out_dir / f"{json_path.stem}.stderr.txt"
        if not result.get("backend_observed") and stderr_path.exists():
            names = observed_backends_in(stderr_path.read_text(encoding="utf-8"))
            result["backend_observed"] = names[0] if len(names) == 1 else None
        runs.append(result)
    return runs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--prompt-file", type=Path)
    p.add_argument("--backends", nargs="+")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--reclassify", action="store_true",
                   help="re-derive the verdict from saved output; no GPU needed")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.reclassify:
        runs = reclassify(args.out)
        print(f"[g1] re-classifying {len(runs)} saved run(s)")
    else:
        if not (args.model and args.prompt_file and args.backends):
            p.error("--model, --prompt-file and --backends are required unless --reclassify")
        runs = []
        for backend in args.backends:
            print(f"[g1] running {backend} ...", flush=True)
            r = run_backend(args.model, args.prompt_file, backend, args.out)
            print(f"[g1]   ok={r['ok']} observed={r.get('backend_observed')} err={r.get('error')}")
            runs.append(r)

    c = classify(runs)
    (args.out / "verdict.json").write_text(json.dumps({
        "verdict": c.verdict.value, "max_abs_logprob_delta": c.max_abs_logprob_delta,
        "divergent_pairs": c.divergent_pairs, "silent_fallbacks": c.silent_fallbacks,
        "notes": c.notes, "runs": runs,
    }, indent=2), encoding="utf-8")

    print(f"\n[g1] VERDICT: {c.verdict.value}")
    print(f"[g1] max abs delta logprob: {c.max_abs_logprob_delta:.3e}")
    if c.silent_fallbacks:
        print(f"[g1] SILENT FALLBACKS (requested -> observed): {c.silent_fallbacks}")
    for note in c.notes:
        print(f"[g1] note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
pytest tests/test_backends.py tests/test_g1_compare.py -v
```
Expected: `test_backends.py` all green, `test_g1_compare.py` all green. Runs on Windows with no GPU.

```bash
git add silentpath/backends.py gate tests/test_backends.py tests/test_g1_compare.py
git commit -m "feat(gate): add G1 divergence kill-test with alias-based backend identity"
```

- [ ] **Step 8: Create the prompt and run G1 for real** *(Linux/ROCm only, after Task 2 passed)*

Use a prompt whose answer is a **discrete value**, so divergence is legible rather than stylistic.

```bash
cat > prompts/g1.txt <<'EOF'
Invoice 7781. Line items: 3 units at 14.99, 2 units at 249.50, 1 unit at 8.25.
Shipping 12.00. Tax 8.875%. What is the invoice total? Answer with the number only.
EOF

python -c "import vllm; print(vllm.__version__)"
python -m gate.g1_divergence \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt-file prompts/g1.txt \
  --backends ROCM_ATTN TRITON_ATTN ROCM_AITER_FA \
  --out runs/g1
```

- [ ] **Step 9: Extend `BACKEND_ALIASES` from reality, then re-classify**

```bash
grep -i "backend" runs/g1/*.stderr.txt | sort -u
python -c "
from pathlib import Path
from silentpath.backends import observed_backends_in
for p in Path('runs/g1').glob('*.stderr.txt'):
    print(p.name, observed_backends_in(p.read_text()))"
```

Each file should yield **exactly one** name. If any yields more, the regex is still over-matching on this vLLM version — narrow it and re-run this check, because an ambiguous observation makes `run_backend` set `backend_observed = None` and `classify` then skips the fallback check entirely.

Then record each real class name into `BACKEND_ALIASES` and verify:

```bash
python -c "from silentpath.backends import backend_matches; print(backend_matches('ROCM_ATTN','<REAL_NAME_FROM_STDERR>'))"
```

Must print `True` — not `None`, which means the name is still unrecorded. **This step is not optional**: without it every observation is `None`/unresolved and the fallback detector reports nothing at all.

Because raw stderr and per-backend JSON were saved in Step 8, re-classifying costs no GPU time:

```bash
python -m gate.g1_divergence --reclassify --out runs/g1
```

- [ ] **Step 10: Record the verdict and decide**

Write the verdict into the G1 row of `docs/STATUS.md` with date, model, backend list and `max_abs_logprob_delta`, then follow the gate table.

```bash
git add runs/g1 docs/STATUS.md prompts silentpath/backends.py
git commit -m "test(gate): record G1 divergence verdict and real backend aliases"
```

> If the verdict is **IDENTICAL**, stop and re-plan. Do not proceed pretending otherwise.

---

## Task 4: The `RunRecord` seam

**Files:** Create `silentpath/record.py`, `tests/test_record.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_record.py`

```python
import pytest
from pydantic import ValidationError

from silentpath.record import Confidence, CostSample, ObservedPath, Output, RunRecord


def test_identical_requested_and_observed_is_never_a_fallback():
    """FakeProducer sets observed == requested, so this is the path the GPU-free
    demo takes. Revision 2 reported a fallback on all four demo records."""
    p = ObservedPath(requested="ROCM_ATTN", observed="ROCM_ATTN",
                     confidence=Confidence.CONFIRMED, signals={})
    assert p.is_silent_fallback is False
    assert p.is_unresolved is False


def test_alias_match_is_not_a_silent_fallback():
    """The realistic case: the class name never contains the env token."""
    p = ObservedPath(requested="ROCM_ATTN", observed="ROCmFlashAttentionBackend",
                     confidence=Confidence.CONFIRMED, signals={})
    assert p.is_silent_fallback is False


def test_wrong_backend_is_a_silent_fallback():
    p = ObservedPath(requested="ROCM_ATTN", observed="TritonAttentionBackend",
                     confidence=Confidence.CONFIRMED, signals={})
    assert p.is_silent_fallback is True
    # A genuine fallback is resolved, not unknown. Without this, `is_unresolved`
    # could be written as `not backend_matches(...)` and silently count every
    # real fallback as unresolved as well.
    assert p.is_unresolved is False


def test_unknown_requested_token_is_unresolved_not_a_fallback():
    p = ObservedPath(requested="SOME_FUTURE_BACKEND", observed="TritonAttentionBackend",
                     confidence=Confidence.REPORTED, signals={})
    assert p.is_silent_fallback is False
    assert p.is_unresolved is True


def test_unrecorded_observed_name_is_unresolved_not_a_fallback():
    p = ObservedPath(requested="ROCM_ATTN", observed="SomeBrandNewBackendV9",
                     confidence=Confidence.REPORTED, signals={})
    assert p.is_silent_fallback is False
    assert p.is_unresolved is True


def test_fallback_is_false_when_observation_is_unknown():
    """Absence of data is not data."""
    p = ObservedPath(requested="ROCM_ATTN", observed=None,
                     confidence=Confidence.UNKNOWN, signals={})
    assert p.is_silent_fallback is False
    assert p.is_unresolved is True


def test_cost_sample_rejects_zero_repetitions():
    with pytest.raises(ValidationError):
        CostSample(wall_seconds=1.0, samples=0)


def test_cost_sample_separates_billable_cell_time_from_generate_time():
    """The budget is charged for the whole cell; the speed claim uses generate()
    only. Conflating them charges the credit seconds for a cell that took
    minutes, because engine init and model load dominate."""
    c = CostSample(wall_seconds=2.0, cell_wall_seconds=310.0, samples=3)
    assert c.wall_seconds == 2.0
    assert c.cell_wall_seconds == 310.0


def test_record_id_is_stable_for_identical_inputs():
    common = dict(
        path=ObservedPath(requested="ROCM_ATTN", observed="ROCmFlashAttentionBackend",
                          confidence=Confidence.CONFIRMED, signals={}),
    )
    a = RunRecord.build(config={"model": "m", "backend": "ROCM_ATTN"}, workload_id="w1",
                        cost=CostSample(wall_seconds=1.0, samples=3),
                        output=Output(text="42", token_ids=[1], chosen_logprobs=[-0.5]),
                        **common)
    b = RunRecord.build(config={"backend": "ROCM_ATTN", "model": "m"}, workload_id="w1",
                        cost=CostSample(wall_seconds=99.0, samples=1),
                        output=Output(text="different", token_ids=[9], chosen_logprobs=[-1.0]),
                        **common)
    # Identity depends on what was requested, not on what came back — otherwise
    # the cache could never recognise a completed cell.
    assert a.record_id == b.record_id


def test_record_id_changes_with_workload():
    common = dict(
        path=ObservedPath(requested="A", observed="A", confidence=Confidence.CONFIRMED, signals={}),
        cost=CostSample(wall_seconds=1.0, samples=1),
        output=Output(text="x", token_ids=[1], chosen_logprobs=[-0.1]),
    )
    assert (RunRecord.build(config={"model": "m"}, workload_id="w1", **common).record_id
            != RunRecord.build(config={"model": "m"}, workload_id="w2", **common).record_id)


def test_record_id_tolerates_non_json_config_values():
    """A Path in a config must not blow up identity computation."""
    from pathlib import Path
    rid = RunRecord.compute_id({"model": Path("/tmp/m")}, "w1")
    assert isinstance(rid, str) and len(rid) == 16


def test_round_trips_through_json():
    r = RunRecord.build(config={"model": "m"}, workload_id="w1",
                        path=ObservedPath(requested="A", observed="A",
                                          confidence=Confidence.CONFIRMED, signals={}),
                        cost=CostSample(wall_seconds=1.0, samples=1),
                        output=Output(text="x", token_ids=[1], chosen_logprobs=[-0.1]))
    assert RunRecord.model_validate_json(r.model_dump_json()) == r
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_record.py -v`

- [ ] **Step 3: Implement `silentpath/record.py`**

```python
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
    orders of magnitude. `device_seconds` is GPU time from HIP events.
    """
    wall_seconds: float
    cell_wall_seconds: float | None = None
    device_seconds: float | None = None
    energy_joules: float | None = None
    peak_memory_bytes: int | None = None
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
```

- [ ] **Step 4: Verify** — `pytest tests/test_record.py -v` → all green

```bash
git add silentpath/record.py tests/test_record.py
git commit -m "feat(core): add RunRecord seam with alias-based fallback detection"
```

---

## Task 5: The comparator — the most important tests in the project

This is where the project either reports an honest finding or fools itself.

**Files:** Create `silentpath/compare.py`, `tests/test_compare.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_compare.py`

```python
from silentpath.compare import DivergenceLevel, compare, is_finding
from silentpath.record import Confidence, CostSample, ObservedPath, Output, RunRecord


def make(text, ids, logprobs, decision=None, backend="ROCM_ATTN", workload="w1", samples=3):
    return RunRecord.build(
        config={"backend": backend}, workload_id=workload,
        path=ObservedPath(requested=backend, observed=backend,
                          confidence=Confidence.CONFIRMED, signals={}),
        cost=CostSample(wall_seconds=1.0, device_seconds=1.0, samples=samples),
        output=Output(text=text, token_ids=ids, chosen_logprobs=logprobs, decision=decision),
    )


def test_bit_identical_records_are_identical():
    d = compare(make("42.00", [1, 2], [-0.5, -0.25], "42.00", "ROCM_ATTN"),
                make("42.00", [1, 2], [-0.5, -0.25], "42.00", "TRITON_ATTN"))
    assert d.level is DivergenceLevel.IDENTICAL
    assert d.max_abs_logprob_delta == 0.0
    assert is_finding(d) is False


def test_numerically_close_but_same_decision_is_not_a_finding():
    """The case the project must NOT report as a discovery. Different kernels
    producing different bits is expected behaviour."""
    d = compare(make("42.00", [1, 2], [-0.5, -0.25], "42.00", "ROCM_ATTN"),
                make("42.00", [1, 2], [-0.50000001, -0.25], "42.00", "TRITON_ATTN"))
    assert d.level is DivergenceLevel.NUMERIC
    assert d.max_abs_logprob_delta > 0.0
    assert d.text_equal is True
    assert is_finding(d) is False


def test_different_decision_is_a_finding():
    d = compare(make("42.00", [1, 2], [-0.5, -0.25], "42.00", "ROCM_ATTN"),
                make("47.00", [1, 3], [-0.5, -0.25], "47.00", "TRITON_ATTN"))
    assert d.level is DivergenceLevel.DECISION
    assert is_finding(d) is True


def test_same_text_but_different_extracted_decision_is_a_finding():
    """Two runs can emit the same prose and still disagree on the extracted value."""
    d = compare(make("The total is 42.00 dollars", [1, 2], [-0.5, -0.25], "42.00", "ROCM_ATTN"),
                make("The total is 42.00 dollars", [1, 2], [-0.5, -0.25], "4200", "TRITON_ATTN"))
    assert d.level is DivergenceLevel.DECISION
    assert is_finding(d) is True


def test_differing_token_counts_are_decision_level():
    assert compare(make("42", [1], [-0.5], "42", "ROCM_ATTN"),
                   make("42.00", [1, 2], [-0.5, -0.25], "42.00", "TRITON_ATTN")
                   ).level is DivergenceLevel.DECISION


def test_differing_token_counts_report_no_delta_rather_than_infinity():
    """No element-wise delta exists. Revision 2 used float('inf'), which
    json.dumps emits as bare `Infinity` — not valid JSON, and it reached the
    MCP tool surface."""
    import json
    d = compare(make("42", [1], [-0.5], "42", "ROCM_ATTN"),
                make("42.00", [1, 2], [-0.5, -0.25], "42.00", "TRITON_ATTN"))
    assert d.max_abs_logprob_delta is None
    json.loads(json.dumps({"delta": d.max_abs_logprob_delta}))   # must not raise


def test_comparing_different_workloads_raises():
    try:
        compare(make("42", [1], [-0.5], "42", "ROCM_ATTN", workload="w1"),
                make("42", [1], [-0.5], "42", "TRITON_ATTN", workload="w2"))
    except ValueError as exc:
        assert "workload" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for mismatched workloads")


def test_failed_run_comparison_raises():
    bad = make("", [], [], None, "TRITON_ATTN").model_copy(update={"ok": False})
    try:
        compare(make("42", [1], [-0.5], "42", "ROCM_ATTN"), bad)
    except ValueError as exc:
        assert "failed" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for failed run")


def test_missing_decisions_fall_back_to_text_equality():
    d = compare(make("42.00", [1, 2], [-0.5, -0.25], None, "ROCM_ATTN"),
                make("42.00", [1, 2], [-0.5, -0.25], None, "TRITON_ATTN"))
    assert d.decision_equal is None
    assert d.level is DivergenceLevel.IDENTICAL
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_compare.py -v`

- [ ] **Step 3: Implement `silentpath/compare.py`**

```python
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
    max_delta = (max((abs(x - y) for x, y in zip(la, lb)), default=0.0)
                 if len(la) == len(lb) else None)

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
```

- [ ] **Step 4: Verify, then confirm the tests are not vacuous**

```bash
pytest tests/test_compare.py -v      # all green
```

Sabotage check — temporarily make `is_finding` also return True for `NUMERIC`, i.e. exactly the "report noise as a discovery" failure the spec forbids:

```bash
pytest tests/ -k "numeric" -v
```
At least **three** tests must fail (in `test_compare.py`, `test_report.py`, `test_mcp_tools.py` once those exist). If only one fails, the other two have fixtures that never reach the NUMERIC state and are vacuous. Revert the sabotage.

```bash
git add silentpath/compare.py tests/test_compare.py
git commit -m "feat(core): add three-level divergence comparator"
```

---

## Task 6: Record store with resume

**Files:** Create `silentpath/store.py`, `tests/test_store.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_store.py`

```python
from silentpath.record import Confidence, CostSample, ObservedPath, Output, RunRecord
from silentpath.store import RecordStore


def make(backend="ROCM_ATTN", workload="w1"):
    return RunRecord.build(
        config={"backend": backend}, workload_id=workload,
        path=ObservedPath(requested=backend, observed=backend,
                          confidence=Confidence.CONFIRMED, signals={}),
        cost=CostSample(wall_seconds=1.0, samples=1),
        output=Output(text="42", token_ids=[1], chosen_logprobs=[-0.5]))


def test_append_then_load_round_trips(tmp_store_dir):
    r = make()
    RecordStore(tmp_store_dir).append(r)
    assert RecordStore(tmp_store_dir).load()[0] == r


def test_has_reports_membership_by_id(tmp_store_dir):
    s, r = RecordStore(tmp_store_dir), make()
    assert s.has(r.record_id) is False
    s.append(r)
    assert s.has(r.record_id) is True


def test_reopened_store_still_knows_completed_cells(tmp_store_dir):
    """Resume depends on this: a fresh process must not recompute a done cell."""
    r = make()
    RecordStore(tmp_store_dir).append(r)
    assert RecordStore(tmp_store_dir).has(r.record_id) is True


def test_append_is_idempotent(tmp_store_dir):
    s, r = RecordStore(tmp_store_dir), make()
    s.append(r)
    s.append(r)
    assert len(s.load()) == 1


def test_by_workload_groups_records(tmp_store_dir):
    s = RecordStore(tmp_store_dir)
    for b, w in [("ROCM_ATTN", "w1"), ("TRITON_ATTN", "w1"), ("ROCM_ATTN", "w2")]:
        s.append(make(b, w))
    groups = s.by_workload()
    assert sorted(groups) == ["w1", "w2"]
    assert len(groups["w1"]) == 2


def test_corrupt_line_is_skipped_not_fatal(tmp_store_dir):
    """A truncated final line is the normal result of an interrupted run and
    must not make the whole store unreadable."""
    s = RecordStore(tmp_store_dir)
    s.append(make())
    with open(s.path, "a", encoding="utf-8") as fh:
        fh.write('{"record_id": "trunc\n')
    assert len(RecordStore(tmp_store_dir).load()) == 1
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_store.py -v`

- [ ] **Step 3: Implement `silentpath/store.py`**

```python
"""Append-only JSONL record store. Content-addressed, so resume is free."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from silentpath.record import RunRecord


class RecordStore:
    def __init__(self, directory: Path | str, filename: str = "records.jsonl") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / filename
        self._ids: set[str] | None = None

    def _known_ids(self) -> set[str]:
        if self._ids is None:
            self._ids = {r.record_id for r in self.load()}
        return self._ids

    def has(self, record_id: str) -> bool:
        return record_id in self._known_ids()

    def append(self, record: RunRecord) -> bool:
        """Append unless already present. Returns True if written."""
        if self.has(record.record_id):
            return False
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        self._known_ids().add(record.record_id)
        return True

    def load(self) -> list[RunRecord]:
        if not self.path.exists():
            return []
        out: list[RunRecord] = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(RunRecord.model_validate_json(line))
                except (ValueError, json.JSONDecodeError):
                    continue   # an interrupted run leaves a truncated last line
        return out

    def by_workload(self) -> dict[str, list[RunRecord]]:
        groups: dict[str, list[RunRecord]] = defaultdict(list)
        for r in self.load():
            groups[r.workload_id].append(r)
        return dict(groups)
```

- [ ] **Step 4: Verify** — `pytest tests/test_store.py -v` → all green

```bash
git add silentpath/store.py tests/test_store.py
git commit -m "feat(core): add append-only record store with resume support"
```

---

## Task 7: Budget ceiling

**Files:** Create `silentpath/budget.py`, `tests/test_budget.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_budget.py`

```python
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
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_budget.py -v`

- [ ] **Step 3: Implement `silentpath/budget.py`**

```python
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
```

- [ ] **Step 4: Verify** — `pytest tests/test_budget.py -v` → all green

```bash
git add silentpath/budget.py tests/test_budget.py
git commit -m "feat(core): add hard GPU budget ceiling"
```

---

## Task 8: Configuration matrix

**Files:** Create `silentpath/matrix.py`, `tests/test_matrix.py`, `configs/smoke.yaml`

- [ ] **Step 1: Write the failing tests** — `tests/test_matrix.py`

```python
import pytest

from silentpath.matrix import expand, load_matrix

YAML = """
model: Qwen/Qwen2.5-1.5B-Instruct
seed: 0
max_tokens: 64
backends: [ROCM_ATTN, TRITON_ATTN]
workloads:
  - id: invoice_01
    prompt: "What is the total?"
  - id: invoice_02
    prompt: "What is the tax?"
"""


def test_expand_produces_backend_by_workload_cells():
    cells = expand(load_matrix(YAML))
    assert len(cells) == 4
    assert {(c.backend, c.workload_id) for c in cells} == {
        ("ROCM_ATTN", "invoice_01"), ("ROCM_ATTN", "invoice_02"),
        ("TRITON_ATTN", "invoice_01"), ("TRITON_ATTN", "invoice_02")}


def test_cell_config_carries_model_and_seed():
    c = expand(load_matrix(YAML))[0]
    assert c.config["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert c.config["seed"] == 0


def test_cell_config_excludes_the_prompt_text():
    """Identity depends on the workload id, not the prompt body, so editing
    whitespace in a prompt does not silently invalidate the whole cache."""
    assert "prompt" not in expand(load_matrix(YAML))[0].config


def test_duplicate_workload_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        expand(load_matrix(YAML + "\n  - id: invoice_01\n    prompt: 'dup'\n"))


def test_empty_backends_rejected():
    with pytest.raises(ValueError, match="backend"):
        expand(load_matrix("model: m\nbackends: []\nworkloads: [{id: a, prompt: p}]"))
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_matrix.py -v`

- [ ] **Step 3: Implement `silentpath/matrix.py`**

```python
"""The sweep is data, not code, so it can shrink to fit the credit budget."""
from __future__ import annotations

from dataclasses import dataclass
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
                 config={**base, "backend": b})
            for b in backends for w in workloads]
```

- [ ] **Step 4: Create `configs/smoke.yaml`**

```yaml
# Smallest useful sweep. This is the one that must work before any larger
# matrix is worth paying for.
model: Qwen/Qwen2.5-1.5B-Instruct
seed: 0
max_tokens: 64
repetitions: 3
backends:
  - ROCM_ATTN
  - TRITON_ATTN
workloads:
  - id: invoice_01
    prompt: |
      Invoice 7781. Line items: 3 units at 14.99, 2 units at 249.50, 1 unit at 8.25.
      Shipping 12.00. Tax 8.875%. What is the invoice total? Answer with the number only.
  - id: invoice_02
    prompt: |
      Invoice 9002. Line items: 7 units at 3.15, 1 unit at 1999.00.
      Discount 12.5%. Tax 6.25%. What is the invoice total? Answer with the number only.
```

- [ ] **Step 5: Verify** — `pytest tests/test_matrix.py -v` → all green

```bash
git add silentpath/matrix.py tests/test_matrix.py configs
git commit -m "feat(core): add config matrix expansion"
```

---

## Task 9: Producer protocol and fake producer

The fake producer is what lets Task 10's runner be tested with no GPU.

**Files:** Create `silentpath/producers/__init__.py`, `silentpath/producers/base.py`, `silentpath/producers/fake.py`, `tests/test_fake_producer.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_fake_producer.py`

```python
from silentpath.matrix import Cell
from silentpath.producers.fake import FakeProducer


def cell(backend="ROCM_ATTN", workload="w1"):
    return Cell(backend=backend, workload_id=workload, prompt="p",
                config={"model": "m", "backend": backend})


def test_produces_a_record_matching_the_cell():
    r = FakeProducer().run(cell())
    assert r.workload_id == "w1"
    assert r.path.requested == "ROCM_ATTN"
    assert r.ok is True


def test_scripted_outputs_let_tests_stage_divergence():
    p = FakeProducer(outputs={("ROCM_ATTN", "w1"): "42.00", ("TRITON_ATTN", "w1"): "47.00"})
    assert p.run(cell("ROCM_ATTN")).output.text == "42.00"
    assert p.run(cell("TRITON_ATTN")).output.text == "47.00"


def test_scripted_fallback_is_recorded_as_observed_path():
    """Uses realistic names: a class name never contains the env-var token."""
    r = FakeProducer(fallbacks={"ROCM_ATTN": "TritonAttentionBackend"}).run(cell("ROCM_ATTN"))
    assert r.path.observed == "TritonAttentionBackend"
    assert r.path.is_silent_fallback is True


def test_correct_backend_class_name_is_not_a_fallback():
    r = FakeProducer(fallbacks={"ROCM_ATTN": "ROCmFlashAttentionBackend"}).run(cell("ROCM_ATTN"))
    assert r.path.is_silent_fallback is False


def test_default_producer_reports_no_fallback():
    """The GPU-free demo path. Revision 2 reported a fallback on every record
    here, which broke Task 14's own sanity check before any GPU was involved."""
    r = FakeProducer().run(cell("ROCM_ATTN"))
    assert r.path.is_silent_fallback is False
    assert r.path.is_unresolved is False


def test_cell_time_defaults_to_generate_time_but_can_differ():
    r = FakeProducer(seconds=2.0, cell_seconds=600.0).run(cell())
    assert r.cost.wall_seconds == 2.0
    assert r.cost.cell_wall_seconds == 600.0


def test_failure_is_recorded_not_raised():
    r = FakeProducer(failures={"ROCM_ATTN"}).run(cell("ROCM_ATTN"))
    assert r.ok is False
    assert r.error is not None
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_fake_producer.py -v`

- [ ] **Step 3: Implement the producer package**

`silentpath/producers/__init__.py` — empty file.

`silentpath/producers/base.py`:

```python
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
```

`silentpath/producers/fake.py`:

```python
"""Deterministic in-memory producer, so the runner is testable without a GPU."""
from __future__ import annotations

from silentpath.matrix import Cell
from silentpath.record import Confidence, CostSample, ObservedPath, Output, RunRecord


class FakeProducer:
    def __init__(self, outputs: dict[tuple[str, str], str] | None = None,
                 fallbacks: dict[str, str] | None = None,
                 failures: set[str] | None = None,
                 seconds: float = 1.0, samples: int = 3,
                 cell_seconds: float | None = None,
                 cell_seconds_by_backend: dict[str, float] | None = None) -> None:
        self.outputs = outputs or {}
        self.fallbacks = fallbacks or {}
        self.failures = failures or set()
        self.seconds = seconds
        self.samples = samples
        self.cell_seconds = cell_seconds
        self.cell_seconds_by_backend = cell_seconds_by_backend or {}
        self.calls: list[Cell] = []

    def _cell_seconds(self, backend: str) -> float:
        if backend in self.cell_seconds_by_backend:
            return self.cell_seconds_by_backend[backend]
        return self.cell_seconds if self.cell_seconds is not None else self.seconds

    def run(self, cell: Cell) -> RunRecord:
        self.calls.append(cell)
        observed = self.fallbacks.get(cell.backend, cell.backend)
        path = ObservedPath(requested=cell.backend, observed=observed,
                            confidence=Confidence.CONFIRMED, signals={"fake": observed})
        cost = CostSample(wall_seconds=self.seconds,
                          cell_wall_seconds=self._cell_seconds(cell.backend),
                          device_seconds=self.seconds,
                          samples=self.samples, wall_stdev=0.0 if self.samples > 1 else None)

        if cell.backend in self.failures:
            return RunRecord.build(
                config=cell.config, workload_id=cell.workload_id, path=path, cost=cost,
                output=Output(text="", token_ids=[], chosen_logprobs=[]),
                ok=False, error="scripted failure")

        text = self.outputs.get((cell.backend, cell.workload_id), "42.00")
        return RunRecord.build(
            config=cell.config, workload_id=cell.workload_id, path=path, cost=cost,
            output=Output(text=text, token_ids=[1, 2], chosen_logprobs=[-0.5, -0.25],
                          decision=text))
```

- [ ] **Step 4: Verify** — `pytest tests/test_fake_producer.py -v` → all green

```bash
git add silentpath/producers tests/test_fake_producer.py
git commit -m "feat(core): add producer protocol and fake producer"
```

---

## Task 10: The runner — cache, resume, budget

**Files:** Create `silentpath/runner.py`, `tests/test_runner.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_runner.py`

```python
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
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_runner.py -v`

- [ ] **Step 3: Implement `silentpath/runner.py`**

```python
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
```

- [ ] **Step 4: Verify** — `pytest tests/test_runner.py -v` → all green, then `pytest` (all green)

```bash
git add silentpath/runner.py tests/test_runner.py
git commit -m "feat(core): add matrix runner with adaptive budget enforcement"
```

---

## Task 11: Report rendering

**Files:** Create `silentpath/report.py`, `tests/test_report.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_report.py`

```python
from silentpath.record import Confidence, CostSample, ObservedPath, Output, RunRecord
from silentpath.report import findings_only, render_divergence_table


def make(backend, text, decision, seconds=1.0, workload="w1", samples=3, logprobs=None):
    return RunRecord.build(
        config={"backend": backend}, workload_id=workload,
        path=ObservedPath(requested=backend, observed=backend,
                          confidence=Confidence.CONFIRMED, signals={}),
        cost=CostSample(wall_seconds=seconds, device_seconds=seconds, samples=samples,
                        wall_stdev=0.01 if samples > 1 else None),
        output=Output(text=text, token_ids=[1], chosen_logprobs=logprobs or [-0.5],
                      decision=decision))


def test_table_reports_a_decision_divergence():
    rows = render_divergence_table([make("ROCM_ATTN", "42.00", "42.00"),
                                    make("TRITON_ATTN", "47.00", "47.00")])
    assert len(rows) == 1
    assert rows[0]["level"] == "decision"
    assert rows[0]["is_finding"] is True


def test_findings_only_filters_out_genuine_numeric_noise():
    """The fixture must actually reach NUMERIC — identical logprobs would make
    this test vacuous, since it would only ever exercise IDENTICAL."""
    rows = render_divergence_table([
        make("ROCM_ATTN", "42.00", "42.00", logprobs=[-0.5]),
        make("TRITON_ATTN", "42.00", "42.00", logprobs=[-0.50000001])])
    assert rows[0]["level"] == "numeric"
    assert findings_only(rows) == []


def test_failed_records_are_excluded_from_comparison():
    bad = make("TRITON_ATTN", "", None).model_copy(update={"ok": False})
    assert render_divergence_table([make("ROCM_ATTN", "42.00", "42.00"), bad]) == []


def test_records_from_different_workloads_are_not_cross_compared():
    assert render_divergence_table([
        make("ROCM_ATTN", "42.00", "42.00", workload="w1"),
        make("TRITON_ATTN", "47.00", "47.00", workload="w2")]) == []


def test_cost_ratio_is_orientation_independent():
    """A 5.5x slowdown must read as 5.5 regardless of insertion order, and the
    slower backend must be named."""
    slow_first = render_divergence_table([make("TRITON_ATTN", "42.00", "42.00", seconds=5.5),
                                          make("ROCM_ATTN", "42.00", "42.00", seconds=1.0)])
    fast_first = render_divergence_table([make("ROCM_ATTN", "42.00", "42.00", seconds=1.0),
                                          make("TRITON_ATTN", "42.00", "42.00", seconds=5.5)])
    assert slow_first[0]["cost_ratio"] == 5.5
    assert fast_first[0]["cost_ratio"] == 5.5
    assert slow_first[0]["slower"] == "TRITON_ATTN"
    assert fast_first[0]["slower"] == "TRITON_ATTN"


def test_row_carries_repetition_count_and_variance():
    """The stdev columns are asserted, not merely present — revision 2 could
    have both columns deleted with a green suite."""
    rows = render_divergence_table([make("ROCM_ATTN", "42.00", "42.00"),
                                    make("TRITON_ATTN", "42.00", "42.00", seconds=5.5)])
    assert rows[0]["samples_min"] == 3
    assert rows[0]["cost_ratio_admissible"] is True
    assert rows[0]["a_wall_stdev"] == 0.01
    assert rows[0]["b_wall_stdev"] == 0.01


def test_single_shot_timing_is_not_an_admissible_speed_result():
    rows = render_divergence_table([make("ROCM_ATTN", "42.00", "42.00", samples=1),
                                    make("TRITON_ATTN", "42.00", "42.00", seconds=5.5, samples=1)])
    assert rows[0]["samples_min"] == 1
    assert rows[0]["cost_ratio_admissible"] is False


def test_cost_ratio_uses_generate_time_not_whole_cell_time():
    """Cell time is dominated by engine init and model load. Comparing it would
    measure startup, not kernels — the ratio here must be 5.5, not ~1.02."""
    a = RunRecord.build(
        config={"backend": "ROCM_ATTN"}, workload_id="w1",
        path=ObservedPath(requested="ROCM_ATTN", observed="ROCM_ATTN",
                          confidence=Confidence.CONFIRMED, signals={}),
        cost=CostSample(wall_seconds=1.0, device_seconds=1.0, cell_wall_seconds=600.0,
                        samples=3, wall_stdev=0.01),
        output=Output(text="42.00", token_ids=[1], chosen_logprobs=[-0.5], decision="42.00"))
    b = RunRecord.build(
        config={"backend": "TRITON_ATTN"}, workload_id="w1",
        path=ObservedPath(requested="TRITON_ATTN", observed="TRITON_ATTN",
                          confidence=Confidence.CONFIRMED, signals={}),
        cost=CostSample(wall_seconds=5.5, device_seconds=5.5, cell_wall_seconds=610.0,
                        samples=3, wall_stdev=0.01),
        output=Output(text="42.00", token_ids=[1], chosen_logprobs=[-0.5], decision="42.00"))
    assert render_divergence_table([a, b])[0]["cost_ratio"] == 5.5


def test_row_reports_fallback_and_unresolved_columns():
    """Both columns are load-bearing for D1's headline and were unasserted."""
    clean = make("ROCM_ATTN", "42.00", "42.00")
    rows = render_divergence_table([clean, make("TRITON_ATTN", "42.00", "42.00")])
    assert rows[0]["silent_fallback"] is False
    assert rows[0]["unresolved_path"] is False

    fell_back = clean.model_copy(update={"path": ObservedPath(
        requested="ROCM_ATTN", observed="TritonAttentionBackend",
        confidence=Confidence.CONFIRMED, signals={})})
    rows = render_divergence_table([fell_back, make("TRITON_ATTN", "42.00", "42.00")])
    assert rows[0]["silent_fallback"] is True
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_report.py -v`

- [ ] **Step 3: Implement `silentpath/report.py`**

```python
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
```

- [ ] **Step 4: Verify** — `pytest tests/test_report.py -v` → all green

```bash
git add silentpath/report.py tests/test_report.py
git commit -m "feat(core): add orientation-independent divergence and cost reporting"
```

---

## Task 12: Telemetry — actually wired in

**Files:** Create `silentpath/telemetry.py`, `tests/test_telemetry.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_telemetry.py`

```python
import time

from silentpath.telemetry import Stopwatch, repeat_timed


def test_stopwatch_measures_elapsed():
    with Stopwatch() as sw:
        time.sleep(0.01)
    assert sw.seconds >= 0.01


def test_repeat_timed_runs_n_times_and_reports_variance():
    calls = []
    sample, _ = repeat_timed(lambda: calls.append(1), repetitions=3)
    assert len(calls) == 3
    assert sample.samples == 3
    assert sample.wall_stdev is not None


def test_repeat_timed_returns_the_last_value():
    """The caller needs the inference output, not just its timing."""
    _, value = repeat_timed(lambda: "result", repetitions=2)
    assert value == "result"


def test_single_repetition_reports_no_stdev():
    """One measurement has no variance; reporting 0.0 would imply precision
    that was never measured."""
    sample, _ = repeat_timed(lambda: None, repetitions=1)
    assert sample.samples == 1
    assert sample.wall_stdev is None


def test_repeat_timed_rejects_zero_repetitions():
    try:
        repeat_timed(lambda: None, repetitions=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_device_seconds_is_absent_rather_than_faked_without_a_gpu(monkeypatch):
    """Asserted by forcing the no-GPU branch, so the test means something on a
    machine that happens to have one. `x is None or x >= 0` would be vacuous."""
    import silentpath.telemetry as t
    monkeypatch.setattr(t, "_torch_with_gpu", lambda: None)
    sample, _ = t.repeat_timed(lambda: None, repetitions=2)
    assert sample.device_seconds is None
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_telemetry.py -v`

- [ ] **Step 3: Implement `silentpath/telemetry.py`**

```python
"""Measured cost. HIP events for device time, amd-smi for memory, wall clock always.

rocprofiler-sdk is deliberately not used: no first-class Python binding, and the
LD_PRELOAD C++ pattern costs about two weeks for a need hackathon-grade timing
already meets.

This module is the single timing implementation. The vLLM worker imports it
rather than keeping its own loop.
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Callable

from silentpath.record import CostSample


class Stopwatch:
    def __enter__(self) -> Stopwatch:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0


def repeat_timed(fn: Callable[[], Any], repetitions: int = 3) -> tuple[CostSample, Any]:
    """Run fn repeatedly; return the cost sample and the last returned value.

    Variance matters: AITER has been reported to show 2-16x higher measurement
    variability than other paths, so a single-shot timing is not an admissible
    speed claim, and report rows carry `samples` so that can be enforced.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")

    torch = _torch_with_gpu()
    wall: list[float] = []
    device: list[float] = []
    value: Any = None

    for _ in range(repetitions):
        start = end = None
        if torch is not None:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        with Stopwatch() as sw:
            value = fn()
        if torch is not None:
            end.record()
            torch.cuda.synchronize()
            device.append(start.elapsed_time(end) / 1000.0)   # ms -> s
        wall.append(sw.seconds)

    sample = CostSample(
        wall_seconds=statistics.fmean(wall),
        device_seconds=statistics.fmean(device) if device else None,
        samples=repetitions,
        wall_stdev=statistics.stdev(wall) if len(wall) > 1 else None,
        **_device_memory())
    return sample, value


def _torch_with_gpu():
    """torch.cuda maps to HIP on ROCm builds. Absent on the dev laptop."""
    try:
        import torch
    except ImportError:
        return None
    try:
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def _device_memory() -> dict:
    """Best-effort VRAM reading; absent on non-ROCm machines."""
    try:
        import amdsmi
    except ImportError:
        return {}
    try:
        amdsmi.amdsmi_init()
        handles = amdsmi.amdsmi_get_processor_handles()
        if not handles:
            return {}
        mem = amdsmi.amdsmi_get_gpu_memory_usage(handles[0], amdsmi.AmdSmiMemoryType.VRAM)
        return {"peak_memory_bytes": int(mem)}
    except Exception:
        return {}          # telemetry is never allowed to fail a measurement
    finally:
        try:
            amdsmi.amdsmi_shut_down()
        except Exception:
            pass
```

- [ ] **Step 4: Verify** — `pytest tests/test_telemetry.py -v` → all green

```bash
git add silentpath/telemetry.py tests/test_telemetry.py
git commit -m "feat(core): add timing with HIP device events and variance"
```

---

## Task 13: The vLLM producer *(Linux/ROCm only)*

**Files:** Create `silentpath/producers/vllm_worker.py`, `silentpath/producers/vllm_subprocess.py`, `tests/test_vllm_producer_parse.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_vllm_producer_parse.py`

```python
from silentpath.producers.vllm_subprocess import derive_path

LOG_TRITON = "INFO 09-10 12:00:00 selector.py:120] Using Triton Attention backend.\n"
LOG_FLASH = "INFO Using Flash Attention backend.\n"


def test_observed_backend_read_from_worker_result_when_present():
    p = derive_path("ROCM_ATTN", {"backend_observed": "ROCmFlashAttentionBackend"}, "")
    assert p.observed == "ROCmFlashAttentionBackend"
    assert p.confidence.value == "reported"


def test_observed_backend_falls_back_to_the_log():
    p = derive_path("TRITON_ATTN", {}, LOG_TRITON)
    assert p.observed == "Triton Attention"     # exact, not a substring check
    assert p.confidence.value == "reported"


def test_a_qualified_log_line_still_confirms():
    """`Triton Attention (V1)` and `TritonAttentionBackend` are one backend.
    Revision 2's bare prefix rule called this a disagreement, so CONFIRMED was
    rarely reachable in practice."""
    p = derive_path("TRITON_ATTN", {"backend_observed": "TritonAttentionBackend"},
                    "INFO Using Triton Attention (V1) backend.\n")
    assert p.confidence.value == "confirmed"


def test_the_default_selector_line_is_ignored():
    """vLLM prints this on every run. Without the negative lookahead it becomes
    a second 'name', the observation goes ambiguous, and the fallback check is
    skipped on the very run the gate depends on."""
    p = derive_path("TRITON_ATTN", {"backend_observed": "TritonAttentionBackend"},
                    "Using the default attention backend selector\n" + LOG_TRITON)
    assert p.confidence.value == "confirmed"


def test_two_agreeing_signals_are_confirmed():
    p = derive_path("TRITON_ATTN", {"backend_observed": "TritonAttentionBackend"}, LOG_TRITON)
    assert p.confidence.value == "confirmed"
    assert p.is_silent_fallback is False


def test_conflicting_signals_are_unknown_not_a_guess():
    """When introspection and logs disagree, that disagreement is the finding.
    Picking one silently would hide it."""
    p = derive_path("ROCM_ATTN", {"backend_observed": "ROCmFlashAttentionBackend"}, LOG_TRITON)
    assert p.confidence.value == "unknown"
    assert p.observed is None
    assert len(p.signals) == 2


def test_a_prefix_lookalike_does_not_count_as_agreement():
    """`Flash Attention` is not `ROCmFlashAttentionBackend`. A substring rule
    would wrongly confirm these as the same backend."""
    p = derive_path("ROCM_ATTN", {"backend_observed": "ROCmFlashAttentionBackend"}, LOG_FLASH)
    assert p.confidence.value == "unknown"


def test_several_distinct_log_lines_do_not_yield_a_guess():
    """vLLM prints several backend lines per run; choosing one would be a guess."""
    p = derive_path("ROCM_ATTN", {}, LOG_TRITON + LOG_FLASH)
    assert p.observed is None
    assert p.confidence.value == "unknown"


def test_no_signals_at_all_is_unknown():
    p = derive_path("ROCM_ATTN", {}, "")
    assert p.observed is None
    assert p.confidence.value == "unknown"
    assert p.is_silent_fallback is False
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_vllm_producer_parse.py -v`

- [ ] **Step 3: Implement `silentpath/producers/vllm_worker.py`**

```python
"""Run one cell under one backend in a fresh process.

Usage: python -m silentpath.producers.vllm_worker <job_json> <out_json>

Timing covers generate() only — never engine init or model load, which on a
subprocess-per-cell design would otherwise dominate the measurement.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from gate.g1_worker import introspect_backend
from silentpath.telemetry import repeat_timed


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    result = {
        "backend_requested": os.environ.get("VLLM_ATTENTION_BACKEND", ""),
        "backend_observed": None, "ok": False, "error": None,
        "text": "", "token_ids": [], "chosen_logprobs": [],
        "wall_seconds": 0.0, "device_seconds": None, "wall_stdev": None,
        "samples": 0, "peak_memory_bytes": None,
    }
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(model=job["model"], seed=job.get("seed", 0), enforce_eager=True,
                  max_model_len=job.get("max_model_len", 2048),
                  gpu_memory_utilization=job.get("gpu_memory_utilization", 0.85))
        params = SamplingParams(temperature=0.0, max_tokens=job.get("max_tokens", 64),
                                logprobs=1, seed=job.get("seed", 0))

        sample, out = repeat_timed(
            lambda: llm.generate([job["prompt"]], params)[0].outputs[0],
            repetitions=max(1, int(job.get("repetitions", 3))))

        token_ids = list(out.token_ids)
        chosen: list[float] = []
        if out.logprobs:
            for tok, step in zip(token_ids, out.logprobs):
                e = step.get(tok)
                chosen.append(float(getattr(e, "logprob", e)) if e is not None else float("nan"))

        result.update(ok=True, text=out.text, token_ids=token_ids, chosen_logprobs=chosen,
                      wall_seconds=sample.wall_seconds, device_seconds=sample.device_seconds,
                      wall_stdev=sample.wall_stdev, samples=sample.samples,
                      peak_memory_bytes=sample.peak_memory_bytes,
                      backend_observed=introspect_backend(llm))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement `silentpath/producers/vllm_subprocess.py`**

```python
"""Producer that runs each cell in a fresh vLLM process.

A fresh process per cell is not an optimisation choice: vLLM selects its
attention backend at engine construction, so an in-process loop would measure
the first backend repeatedly.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from silentpath.backends import names_agree, observed_backends_in
from silentpath.record import Confidence, CostSample, ObservedPath, Output, RunRecord
from silentpath.telemetry import Stopwatch


def derive_path(requested: str, worker_result: dict, stderr: str) -> ObservedPath:
    """Combine introspection and log signals into one observation.

    Two agreeing signals are CONFIRMED; one is REPORTED; two that disagree are
    UNKNOWN, because a disagreement is itself information and silently picking
    a winner would destroy it. Agreement is decided by `names_agree`, which is a
    qualifier-stripped prefix match — strict enough to reject `Flash Attention`
    against `ROCmFlashAttentionBackend`, loose enough to accept
    `Triton Attention (V1)` against `TritonAttentionBackend`.
    """
    signals: dict[str, str] = {}
    if worker_result.get("backend_observed"):
        signals["introspection"] = str(worker_result["backend_observed"])

    log_names = observed_backends_in(stderr)
    if len(log_names) == 1:
        signals["log"] = log_names[0]
    elif len(log_names) > 1:
        signals["log_ambiguous"] = " | ".join(log_names)

    resolvable = {k: v for k, v in signals.items() if k in ("introspection", "log")}

    if not resolvable:
        return ObservedPath(requested=requested, observed=None,
                            confidence=Confidence.UNKNOWN, signals=signals)
    if len(resolvable) == 1:
        return ObservedPath(requested=requested, observed=next(iter(resolvable.values())),
                            confidence=Confidence.REPORTED, signals=signals)

    agree = names_agree(signals["introspection"], signals["log"])
    return ObservedPath(
        requested=requested,
        observed=signals["introspection"] if agree else None,
        confidence=Confidence.CONFIRMED if agree else Confidence.UNKNOWN,
        signals=signals)


class VllmSubprocessProducer:
    def __init__(self, timeout_seconds: int = 1800, artifact_dir: Path | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.artifact_dir = artifact_dir

    def run(self, cell) -> RunRecord:
        job = {**cell.config, "prompt": cell.prompt}
        job.pop("backend", None)

        with tempfile.TemporaryDirectory() as td:
            job_path, out_path = Path(td) / "job.json", Path(td) / "out.json"
            job_path.write_text(json.dumps(job, default=str), encoding="utf-8")
            env = {**os.environ, "VLLM_ATTENTION_BACKEND": cell.backend}
            # Time the whole subprocess: engine init and model load dominate a
            # cell and are what the credit is actually billed for. This is also
            # charged when the cell fails, which is when it matters most.
            with Stopwatch() as cell_timer:
                proc = subprocess.run(
                    [sys.executable, "-m", "silentpath.producers.vllm_worker",
                     str(job_path), str(out_path)],
                    env=env, capture_output=True, text=True, timeout=self.timeout_seconds)
            result = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

        if self.artifact_dir:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            (self.artifact_dir / f"{cell.backend}.{cell.workload_id}.stderr.txt").write_text(
                proc.stderr, encoding="utf-8")

        return RunRecord.build(
            config=cell.config, workload_id=cell.workload_id,
            path=derive_path(cell.backend, result, proc.stderr),
            cost=CostSample(wall_seconds=result.get("wall_seconds") or 0.0,
                            cell_wall_seconds=cell_timer.seconds,
                            device_seconds=result.get("device_seconds"),
                            samples=max(1, int(result.get("samples") or 1)),
                            wall_stdev=result.get("wall_stdev"),
                            peak_memory_bytes=result.get("peak_memory_bytes")),
            output=Output(text=result.get("text", ""),
                          token_ids=result.get("token_ids", []),
                          chosen_logprobs=result.get("chosen_logprobs", []),
                          decision=None),
            ok=bool(result.get("ok")),
            error=result.get("error") or (None if result else f"worker exited {proc.returncode}"))
```

- [ ] **Step 5: Verify** — `pytest tests/test_vllm_producer_parse.py -v` → all green

```bash
git add silentpath/producers tests/test_vllm_producer_parse.py
git commit -m "feat(producer): add subprocess vLLM producer with prefix-matched path observation"
```

---

## Task 14: CLI

**Files:** Create `silentpath/cli.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_cli.py`

```python
from silentpath.cli import build_parser


def test_run_command_parses_required_arguments():
    a = build_parser().parse_args(
        ["run", "--matrix", "configs/smoke.yaml", "--out", "runs/x", "--cap-hours", "2.5"])
    assert a.command == "run"
    assert a.cap_hours == 2.5


def test_report_command_parses():
    assert build_parser().parse_args(["report", "--out", "runs/x"]).command == "report"


def test_run_defaults_to_fake_producer_off():
    a = build_parser().parse_args(["run", "--matrix", "m.yaml", "--out", "o", "--cap-hours", "1"])
    assert a.fake is False


def test_fake_flag_enables_gpu_free_dry_run():
    a = build_parser().parse_args(
        ["run", "--matrix", "m.yaml", "--out", "o", "--cap-hours", "1", "--fake"])
    assert a.fake is True


def test_estimate_seconds_has_a_conservative_default():
    """It bounds the first cell, before any measurement exists."""
    a = build_parser().parse_args(["run", "--matrix", "m.yaml", "--out", "o",
                                   "--cap-hours", "1"])
    assert a.estimate_seconds >= 300.0
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_cli.py -v`

- [ ] **Step 3: Implement `silentpath/cli.py`**

```python
"""Command line entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from silentpath.budget import BudgetExceeded, GpuBudget
from silentpath.matrix import expand, load_matrix
from silentpath.report import findings_only, render_divergence_table
from silentpath.runner import billable_seconds, run_matrix
from silentpath.store import RecordStore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="silentpath")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="execute a configuration matrix")
    r.add_argument("--matrix", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--cap-hours", required=True, type=float)
    r.add_argument("--estimate-seconds", type=float, default=600.0,
                   help="expected cost of one cell, used to bound the first cell "
                        "before any has been measured")
    r.add_argument("--fake", action="store_true",
                   help="use the fake producer; no GPU, for wiring checks")

    rep = sub.add_parser("report", help="render the divergence and cost table")
    rep.add_argument("--out", required=True, type=Path)
    rep.add_argument("--findings-only", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except BudgetExceeded as exc:
        # Exhausting the cap is an expected outcome, not a crash. Completed
        # cells are already in the store, so a later call resumes.
        print(f"[silentpath] stopped: {exc}")
        return 2


def _run(argv: list[str] | None) -> int:
    args = build_parser().parse_args(argv)
    store = RecordStore(args.out)

    if args.command == "run":
        cells = expand(load_matrix(args.matrix.read_text(encoding="utf-8")))
        if args.fake:
            from silentpath.producers.fake import FakeProducer
            producer = FakeProducer()
        else:
            from silentpath.producers.vllm_subprocess import VllmSubprocessProducer
            producer = VllmSubprocessProducer(artifact_dir=args.out / "stderr")

        # Charge what previous invocations already spent. A fresh budget per
        # invocation re-arms the full cap on every resume: two runs under a
        # declared 0.75h cap spent 1.0h.
        budget = GpuBudget(cap_hours=args.cap_hours)
        budget.seed(sum(billable_seconds(r) for r in store.load()))
        print(f"[silentpath] prior spend {budget.spent_hours:.2f}h of {args.cap_hours:.2f}h")

        n = run_matrix(cells, producer, store, budget,
                       estimate_seconds=args.estimate_seconds)
        print(f"[silentpath] executed {n} new cell(s); {len(store.load())} total; "
              f"spent {budget.spent_hours:.2f}h (${budget.spent_usd:.2f})")
        return 0

    rows = render_divergence_table(store.load())
    if getattr(args, "findings_only", False):
        rows = findings_only(rows)
    print(json.dumps(rows, indent=2))
    inadmissible = sum(1 for r in rows if not r["cost_ratio_admissible"])
    print(f"\n[silentpath] {len(findings_only(rows))} finding(s) of {len(rows)} comparison(s)")
    if inadmissible:
        print(f"[silentpath] {inadmissible} row(s) have too few repetitions for a speed claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify tests, then the whole wiring with no GPU**

```bash
pytest tests/test_cli.py -v
python -m silentpath.cli run --matrix configs/smoke.yaml --out runs/dryrun --cap-hours 1 --fake
python -m silentpath.cli report --out runs/dryrun
```

Expected: `test_cli.py` all green; 4 cells executed; the report prints 2 comparisons with **0 findings** — the fake producer returns the same answer for every backend, so anything else means the comparator is broken.

```bash
git add silentpath/cli.py tests/test_cli.py
git commit -m "feat(cli): add run and report commands"
```

- [ ] **Step 5: Smoke-run the real sweep** *(Linux/ROCm only, after G1 and G2 returned)*

```bash
python -m silentpath.cli run --matrix configs/smoke.yaml --out runs/smoke --cap-hours 1.0
python -m silentpath.cli report --out runs/smoke
```

Sanity-check before believing anything: `silent_fallback` must **not** be true on every row. If it is, `BACKEND_ALIASES` was never populated from real stderr in Task 3 Step 9.

---

## Task 15: D1 — the MCP server

Maps to program theme 1 (serving, MCP, tool-calling, GPU resource consumption).

**Files:** Create `silentpath/mcp_server.py`, `tests/test_mcp_tools.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_mcp_tools.py`

Test the tool *functions*, not the MCP transport.

```python
from silentpath.mcp_server import summarise_store, which_path
from silentpath.record import Confidence, CostSample, ObservedPath, Output, RunRecord
from silentpath.store import RecordStore


def make(backend, observed, text="42.00", seconds=1.0, workload="w1", logprobs=None,
         samples=3):
    return RunRecord.build(
        config={"backend": backend}, workload_id=workload,
        path=ObservedPath(requested=backend, observed=observed,
                          confidence=Confidence.CONFIRMED, signals={"log": observed}),
        cost=CostSample(wall_seconds=seconds, device_seconds=seconds, samples=samples,
                        wall_stdev=0.01 if samples > 1 else None),
        output=Output(text=text, token_ids=[1], chosen_logprobs=logprobs or [-0.5],
                      decision=text))


def test_which_path_reports_a_genuine_silent_fallback(tmp_store_dir):
    RecordStore(tmp_store_dir).append(make("ROCM_ATTN", "TritonAttentionBackend"))
    assert which_path(str(tmp_store_dir))["silent_fallbacks"][0]["requested"] == "ROCM_ATTN"


def test_which_path_reports_none_for_a_correct_backend_class_name(tmp_store_dir):
    """The realistic case. A substring rule would report a fallback here and
    make every real sweep a false positive."""
    RecordStore(tmp_store_dir).append(make("ROCM_ATTN", "ROCmFlashAttentionBackend"))
    assert which_path(str(tmp_store_dir))["silent_fallbacks"] == []


def test_summary_counts_findings_separately_from_comparisons(tmp_store_dir):
    s = RecordStore(tmp_store_dir)
    s.append(make("ROCM_ATTN", "ROCmFlashAttentionBackend", text="42.00"))
    s.append(make("TRITON_ATTN", "TritonAttentionBackend", text="47.00"))
    out = summarise_store(str(tmp_store_dir))
    assert out["comparisons"] == 1
    assert out["findings"] == 1


def test_summary_does_not_count_genuine_numeric_noise_as_a_finding(tmp_store_dir):
    """Fixture must actually reach NUMERIC — identical logprobs would make this
    vacuous, since it would only ever exercise IDENTICAL."""
    s = RecordStore(tmp_store_dir)
    s.append(make("ROCM_ATTN", "ROCmFlashAttentionBackend", text="42.00", logprobs=[-0.5]))
    s.append(make("TRITON_ATTN", "TritonAttentionBackend", text="42.00", logprobs=[-0.50000001]))
    out = summarise_store(str(tmp_store_dir))
    assert out["comparisons"] == 1
    assert out["findings"] == 0


def test_summary_reports_worst_cost_ratio_slow_record_first(tmp_store_dir):
    s = RecordStore(tmp_store_dir)
    s.append(make("TRITON_ATTN", "TritonAttentionBackend", seconds=5.5))
    s.append(make("ROCM_ATTN", "ROCmFlashAttentionBackend", seconds=1.0))
    assert summarise_store(str(tmp_store_dir))["worst_cost_ratio"] == 5.5


def test_summary_reports_worst_cost_ratio_fast_record_first(tmp_store_dir):
    s = RecordStore(tmp_store_dir)
    s.append(make("ROCM_ATTN", "ROCmFlashAttentionBackend", seconds=1.0))
    s.append(make("TRITON_ATTN", "TritonAttentionBackend", seconds=5.5))
    assert summarise_store(str(tmp_store_dir))["worst_cost_ratio"] == 5.5


def test_single_shot_rows_are_excluded_from_worst_cost_ratio(tmp_store_dir):
    """A timing with one repetition is not an admissible speed claim. Revision 2
    implemented this gate but no test asserted it, so removing it left the suite
    green."""
    s = RecordStore(tmp_store_dir)
    s.append(make("ROCM_ATTN", "ROCmFlashAttentionBackend", seconds=1.0, samples=1))
    s.append(make("TRITON_ATTN", "TritonAttentionBackend", seconds=5.5, samples=1))
    out = summarise_store(str(tmp_store_dir))
    assert out["worst_cost_ratio"] is None
    assert out["inadmissible_speed_rows"] == 1


def test_unresolved_paths_are_counted_separately_from_fallbacks(tmp_store_dir):
    """'We could not tell' is a different claim from 'it fell back'."""
    RecordStore(tmp_store_dir).append(make("ROCM_ATTN", "SomeBrandNewBackendV9"))
    out = summarise_store(str(tmp_store_dir))
    assert out["silent_fallbacks"] == 0
    assert out["unresolved_paths"] == 1


def test_empty_store_summarises_without_error(tmp_store_dir):
    out = summarise_store(str(tmp_store_dir))
    assert out["records"] == 0
    assert out["findings"] == 0
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_mcp_tools.py -v`

- [ ] **Step 3: Implement `silentpath/mcp_server.py`**

```python
"""D1 — expose the probe as MCP tools so an agent can ask what a path costs
before committing to it.

The tool functions are plain callables and are unit-tested directly; the MCP
decoration is a thin wrapper so the transport is never in the way of a test.
"""
from __future__ import annotations

from typing import Any

from silentpath.report import findings_only, render_divergence_table
from silentpath.store import RecordStore


def which_path(store_dir: str) -> dict[str, Any]:
    """Report which backend was requested versus actually observed."""
    records = RecordStore(store_dir).load()
    return {
        "records": len(records),
        "paths": [{
            "workload": r.workload_id,
            "requested": r.path.requested,
            "observed": r.path.observed,
            "confidence": r.path.confidence.value,
            "device_seconds": r.cost.device_seconds or r.cost.wall_seconds,
            "samples": r.cost.samples,
        } for r in records],
        "silent_fallbacks": [
            {"workload": r.workload_id, "requested": r.path.requested,
             "observed": r.path.observed}
            for r in records if r.path.is_silent_fallback],
        # Reported separately, never folded into silent_fallbacks: "we could not
        # tell" is a different claim from "it fell back", and conflating them is
        # how a false headline gets published.
        "unresolved": [
            {"workload": r.workload_id, "requested": r.path.requested,
             "observed": r.path.observed, "confidence": r.path.confidence.value}
            for r in records if r.path.is_unresolved],
    }


def summarise_store(store_dir: str) -> dict[str, Any]:
    """Headline numbers: how many comparisons, how many are real findings."""
    records = RecordStore(store_dir).load()
    rows = render_divergence_table(records)
    ratios = [r["cost_ratio"] for r in rows if r["cost_ratio_admissible"]]
    return {
        "records": len(records),
        "comparisons": len(rows),
        "findings": len(findings_only(rows)),
        "worst_cost_ratio": max(ratios) if ratios else None,
        "inadmissible_speed_rows": sum(1 for r in rows if not r["cost_ratio_admissible"]),
        "silent_fallbacks": sum(1 for r in records if r.path.is_silent_fallback),
        "unresolved_paths": sum(1 for r in records if r.path.is_unresolved),
    }


def build_server():  # pragma: no cover - transport wiring
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("silentpath")
    server.tool()(which_path)
    server.tool()(summarise_store)
    return server


if __name__ == "__main__":  # pragma: no cover
    build_server().run()
```

- [ ] **Step 4: Verify, and confirm the suite is not vacuous**

```bash
pytest tests/test_mcp_tools.py -v      # all green
pytest                                  # full suite green
pytest --collect-only | tail -1         # record the real total; do not trust a hard-coded number
```

Repeat the Task 5 sabotage check now that all three consumers exist: make `is_finding` also return True for `NUMERIC` and confirm **three** tests fail across `test_compare.py`, `test_report.py`, `test_mcp_tools.py`. Revert.

```bash
git add silentpath/mcp_server.py tests/test_mcp_tools.py
git commit -m "feat(d1): expose path and divergence summary as MCP tools"
```

---

## Task 16: D1 packaging and submission

The five XP checkboxes are worth roughly 800 XP and are the most commonly forfeited points in this program. This task exists so they are never skipped.

**Files:** Create `README.md`, `LICENSE`; modify `docs/STATUS.md`

- [ ] **Step 1: Write `README.md`**

Must contain: the one-sentence thesis; the G1 verdict with date and model; a quickstart that runs `--fake` with no GPU; the real sweep command; a results table; an explicit statement of what is *not* claimed (numeric divergence is not a finding, and a `cost_ratio` with `samples < 2` is not a speed result); and the licence.

Lead with the finding, not the architecture. If G1 returned DECISION-DIVERGENT, the first line is the divergent example.

- [ ] **Step 2: Verify the repository is genuinely reproducible**

```bash
cd "$(mktemp -d)" && git clone "H:/augsepthacks/amd" fresh && cd fresh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m silentpath.cli run --matrix configs/smoke.yaml --out runs/x --cap-hours 1 --fake
python -m silentpath.cli report --out runs/x
```

Every step must pass in the clone. A quickstart that only works in the original working directory is the most common way a submission loses its "complete project" credit.

- [ ] **Step 3: Add an OSI licence**

```bash
curl -sL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
```

The open-source line item is 200 XP and requires an actual licence file.

- [ ] **Step 4: Record the three milestones** (capped at 3 per project, 100 XP each)

```bash
git tag -a d1-m1-seam     -m "RunRecord seam and comparator complete"
git tag -a d1-m2-producer -m "vLLM subprocess producer with path observation"
git tag -a d1-m3-mcp      -m "MCP tool surface and CLI"
```

- [ ] **Step 5: Submit, with every checkbox ticked**

Omitting these silently forfeits ~700 XP:

- [ ] Demo video recorded and linked (150 XP)
- [ ] Marked as a complete project (200 XP)
- [ ] Marked as open-source, with LICENSE present (200 XP)
- [ ] Tagged as built with AMD technologies (250 XP)
- [ ] All three milestones logged (300 XP)
- [ ] GitHub repo connected (5 XP)

- [ ] **Step 6: Update `docs/STATUS.md` and commit**

Move D1 to DONE, record the submission date, and note the date the XP actually appears — that measures the reporting lag empirically, which every later deliverable's scheduling depends on.

```bash
git add README.md LICENSE docs/STATUS.md
git commit -m "docs(d1): add README, licence and submission record"
```

---

## Expected test counts

Per task, verified by execution against revision 1 where unchanged. Do not trust cumulative totals — run `pytest --collect-only | tail -1`.

| Task | File | Tests |
|---|---|---|
| 2 | `test_g2_parse.py` | 3 |
| 3 | `test_backends.py` | 15 (5 are parametrised cases of one function) |
| 3 | `test_g1_compare.py` | 8 |
| 4 | `test_record.py` | 12 |
| 5 | `test_compare.py` | 9 |
| 6 | `test_store.py` | 6 |
| 7 | `test_budget.py` | 9 |
| 8 | `test_matrix.py` | 5 |
| 9 | `test_fake_producer.py` | 7 |
| 10 | `test_runner.py` | 10 |
| 11 | `test_report.py` | 7 |
| 12 | `test_telemetry.py` | 6 |
| 13 | `test_vllm_producer_parse.py` | 9 |
| 14 | `test_cli.py` | 5 |
| 15 | `test_mcp_tools.py` | 9 |

## What is deliberately not in this plan

- **D2–D6.** Each gets its own plan. D2's premise depends entirely on Task 3's verdict, so writing it now would be writing fiction.
- **Cross-vendor (NVIDIA) records.** The seam makes this nearly free later — same `RunRecord` schema, different producer — and the council rated it the most defensible artifact. Deferred, not dropped; it belongs in the D2 plan once there is a finding to compare across vendors.
- **rocprofiler-sdk per-kernel counters.** A documented extension, not a dependency.
- **Any public website, hosted dataset, or preprint.** Overruled unanimously by the council: no XP, real time cost.

## Execution notes

- Run `pytest` after every task. The suite must be green before each commit.
- Tasks 1, 4–13, 15, 16 and Task 14 Steps 1–4 need no GPU and run on Windows. Only Tasks 2, 3 and Task 14 Step 5 need Linux with ROCm.
- Task 2 runs before Task 3. Task 3 is the ⛔ gate **for the D2 plan only** — Tasks 4–16 run in parallel with it.
- **Task 3 Step 9 is mandatory.** Without real backend aliases, `which_path` reports a fallback on every run and D1 publishes a false positive.
- Do not claim the $100 credit until Task 14 Step 5 runs unattended end to end.
- Commit with a bare `git commit` so the global git config applies. Never pass `-c user.email`.
