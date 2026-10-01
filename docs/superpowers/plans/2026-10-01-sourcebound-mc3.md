# SOURCEBOUND (Mini-Challenge 3) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SOURCEBOUND RAG submission specified in `docs/MINI_CHALLENGE_3_SPEC.md` (the "spec"). It answers ten hidden questions over a hostile mixed-format corpus with the value only and an **exact** citation set, inside the MC3 container contract. It has:
- a stdlib-only `/app/app.py` client;
- a resident GPU worker;
- deterministic grounding, supersession and necessity gates;
- a harness-faithful evaluation;
- a registry-side (`crane`) release.

**Architecture:** Everything that can run without a GPU is written test-first and tested locally with pytest and a scripted `FakeEngine`:
- walk, parsers, killable parser pool, index, hybrid retrieval and identifier bridge;
- reader prompt, gates and pipeline;
- unix-socket/TCP protocol, worker, supervisor and thin client;
- evaluation tooling and corpus generator;
- layer builder.

The only GPU-only module is `sourcebound/engine_qwen.py`. GPU sessions are driven from this Windows machine through `tools/amd-gpu/remote-mc3.js`, which is built on the Playwright tooling documented in `GPUWEBSKILL.md`. The release image is assembled in the registry by appending layers to the mandated base with `crane` from a GitHub Actions workflow. The notebook then rehearses that exact image.

**Tech Stack:**
- **GPU side:** Python 3.14, torch 2.13.0+rocm10.0.0 (preinstalled, never reinstalled), transformers 5.17, PyMuPDF, openpyxl.
- **Local:** Python 3.10+ (verified with 3.12), pytest, numpy, Pillow, PyMuPDF, openpyxl.
- **Remote driver:** Node 22 + Playwright (`tools/amd-gpu`).
- **Release:** crane (go-containerregistry), GitHub Actions, Docker Hub.

---

## How this plan was verified (2026-10-01)

Every code block in Tasks 1–14 was written to a scratch copy of this repository and run before the plan was written.

**Results:**
- **New suite:** 76 tests, all in `tests/test_sb_*.py`; 75 passed and 1 skipped on Windows / Python 3.12. The skip is the real `chmod 000` test, which runs on Linux CI as a non-root user.
- **Whole repository:** with the new code overlaid on the current `master` in a throwaway worktree, **237 passed, 1 skipped** (162 existing + 75 new).
- **Scripts:** `node --check` passes on `remote-mc3.js`; `bash -n` passes on `publish.sh`, `rehearse.sh`, `contract_check.sh` and `worker_ctl.sh`.
- **Plan matches code:** every source file is checked to appear in this plan verbatim.

**What the runs caught.** Running the code caught six defects. Each one is fixed below and pinned by a test:

| # | Found by | Defect | Fix |
|---|---|---|---|
| 1 | `test_withdrawn_by_name_points_to_current_sibling` | A current datasheet whose header says "Revision 1 is withdrawn" was itself marked WITHDRAWN | Heading lines that describe *another* revision are ignored (`_OTHER_DOC`) |
| 2 | `test_shape_keeps_value_drops_label_and_unit` | `"REV-C2".` kept a trailing quote | `shape()` strips wrappers until stable |
| 3 | `test_answer_must_appear_in_its_evidence` | **Near-miss leak:** the answer-in-evidence fallback searched the whole file, so `4.3.1` (another CSV row) passed against an ORR-1847 quote | The fallback is the *segment containing the quote*, never the whole file |
| 4 | contract tests on Windows | `socket.AF_UNIX` does not exist on Windows CPython | The protocol compares with `getattr(socket, "AF_UNIX", None)`; Windows uses localhost TCP |
| 5 | `test_pipeline_retries_past_withdrawn_and_completes_qualifier` | Every image in the top 8 was attached to the VLM call (wasted vision tokens) | Only images ranked in the top 4 are attached (max 3) |
| 6 | PyMuPDF probe | An owner-password-only PDF opens with `is_encrypted == False` | `metadata["encryption"]` is also checked |

**Revision 2 (independent plan review).** These findings were confirmed and fixed in the code above:

| # | Finding | Fix |
|---|---|---|
| R1 | `build_app()` reads `mc3/requirements.txt`, but it was created in Task 14, so Task 12's test and the Task 13 CI job would fail | `mc3/requirements.txt` moves to Task 1 |
| R2 | `pkill -f sourcebound.worker` inside `gpu.js sh` (`bash -lc`) kills its own shell; `[exit -15]` was read as success | Patterns are `'[s]ourcebound...'`; the exit check catches negative codes |
| R3 | `publish.sh` downloaded the whole model and wrote every tar before appending, about 35–39 GB on a runner with about 25–29 GB free | `push-weights`: download, tar, append and delete one ≤ 4 GiB group at a time (`group_files` is unit-tested) |
| R4 | The supervisor was set as `ENTRYPOINT`, so `docker run <image> python3 /app/app.py --index ...` would hang (spec §10 says CMD) | `CMD` everywhere; `publish.sh` stops if the base has an ENTRYPOINT; `check_image.py` checks `Cmd` |
| R5a | The transcript cache was keyed by image sha1 only, so the second reader in E4 reused the first reader's OCR | Key = `<engine name>:<sha1>`; `worker-start` clears `/tmp/sb-index` |
| R5b | Every index after the first got only what was left of the startup budget, so suites and scale timing ran with about 120 s | Only the first index after worker start is charged to the startup budget (tested). Scale timing starts a fresh worker and checks `index_stats.transcribed` |
| R7 | (re-review) The `[s]ourcebound.worker` pattern still matched the `nohup python -m sourcebound.worker` text in the same `bash -lc` line | Start and stop moved to `eval_mc3/worker_ctl.sh`; no `gpu.js sh` command contains the process name |
| R8 | (re-review) `QwenEngine.name` was the `mc3-reader` symlink name, the same for every reader | The name is the resolved model directory |
| R6 | Advisory items: gitignored gates JSON; ephemeral model gone in a new session; Git Bash-only escaping; `--index` exit code; push confirmation; registry reachability from the pod | `git add -f`; re-download step; Git Bash note; `--index` exits non-zero only if the worker never became ready; ask before the pushing command; `rehearse local` fallback |

**Not verifiable locally:**
- `engine_qwen.py`: compiled only, exercised in Task 15;
- `remote-mc3.js` against a live pod;
- `publish.sh` and `check_image.py` against a registry;
- `rehearse.sh`.

Those are the GPU and release tasks, and each has explicit expected output and a fallback.

---

## Before you start (read once)

1. Read `GPUWEBSKILL.md` §0, §3, §5.3 and §8. `node tools/amd-gpu/gpu.js status` must work. If it fails with `NOT_SIGNED_IN`, ask the user to run `! cd tools/amd-gpu && node login.js`. **Never type the password yourself.**
2. Facts this plan relies on:
   - **GPU:** W7900D `gfx1100` (RDNA3), 48 GiB. **BF16 only, no FP8.**
   - **Storage:** `/workspace` is persistent, 25 GB, and is the Jupyter root (it held the MC2 Qwen3-VL-4B, 8.3 GB). `/` is large but ephemeral. `/opt/venv` resets every session.
   - **Environment:** `HF_HOME=/workspace/.cache/huggingface` and `HF_ENDPOINT=https://hf-mirror.com` are preset; downloads run at about 90 MB/s.
   - **Overheads:** a pod launch takes about 80 s; each `gpu.js` or `remote-mc3.js` call adds about 10 s of browser overhead.
3. **Quota rule:** 3 h of pod time per day, counted while the pod exists. **Every GPU task ends with `node tools/amd-gpu/remote-mc3.js end`.** Plan at most 150 min per session. Record `quota_remaining_seconds` from `gpu.js status` at the start and end of each GPU task in its results note.
4. **Hard contract** (spec Further Notes A):
   - **Invocations:**
     - `python3 /app/app.py --index /app/corpus` — once, inside the 600 s startup budget;
     - `python3 /app/app.py --corpus /app/corpus --query-id <id> --query "<q>"` — a new process each time, under 30 s.
   - **Output:** `/app/output/<id>_output.json`, always with string `answer` and list `citations`.
   - **Limits:** 10 min for the whole run; VRAM 1–48 GiB, continuously sampled; image ≤ 60 GiB uncompressed on the mandated base (`rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`).
   - **No network at evaluation.**
5. **Never** commit or print the image reference (challenge PDF: "do not advertise where it lives"). It lives only in the GitHub secret `MC3_IMAGE` and in your local shell variable `MC3_IMAGE`.
6. Run commands from the repo root `H:\augsepthacks\amd`. The Bash tool is Git Bash; PowerShell also works for `python` and `node`. Work on a branch: `git switch -c feat/sourcebound-mc3`.

## Time and quota budget

| Session | Tasks | Pod minutes (target) |
|---|---|---|
| Local, no GPU | 1–14 | 0 |
| GPU 1 | 15: gates, dependency lock, models, smoke (E1 on the kit) | ≤ 120 |
| GPU 2 | 16–17: E1 on development corpora, E2 dense, E3 bridge | ≤ 150 |
| GPU 3 | 18: E4 release readers (8B, 9B) | ≤ 150 |
| GPU 4 | 19–20: fixes from failure analysis, freeze, holdout once, scale timing | ≤ 150 |
| CI + GPU 5 | 21: publish with crane, check, rehearsal on the notebook | ≤ 90 |

## File structure

```
sourcebound/                 package (copied to /app/sourcebound in the image)
  __init__.py
  normalize.py               grader normalization, answer shaping, qualifier completion, number alignment
  terms.py                   identifier-aware tokens; identifier extraction for the bridge
  records.py                 Segment, FileRecord, WITHDRAWN/SUPERSEDED status, document families
  walk.py                    defensive corpus walk, magic-byte type detection, ledger
  parsers.py                 pdf / docx / xlsx / csv / text / code / image parsers (run in children)
  pool.py                    ParserPool: per-file timeout, kill-and-replace children
  bm25.py                    BM25
  index.py                   Index (BM25 + identifiers + vectors), build_index, save/load, transcription phase
  retrieve.py                hybrid search (RRF), file ranking, identifier bridge
  reader.py                  evidence pack, reader prompt, warrant prompt, JSON reply parser
  gates.py                   grounding, answer-in-evidence, supersession, necessity -> Verdict
  pipeline.py                one question under a deadline (retrieve -> read -> gates -> warrant)
  engine.py                  engine interface + FakeEngine
  engine_qwen.py             GPU engine (Qwen VLM + Qwen3-Embedding)          [GPU only]
  protocol.py                NDJSON over unix socket / TCP (stdlib only)
  worker.py                  resident worker: index / query / status
  supervisor.py              container entrypoint: restarts worker, never exits
mc3/app.py                   /app/app.py thin client (stdlib + protocol)
mc3/requirements.txt         top-level deps; mc3/requirements.lock is frozen on the GPU box (Task 15)
eval_mc3/kit/                starter-kit corpus + sample-questions.json (from mc3-starter-kit.zip)
eval_mc3/score.py            grader-faithful scoring (exact citation sets)
eval_mc3/run_eval.py         harness-faithful runner (index + process per question)
eval_mc3/run_suite.py        runs every corpus of a split, applies hazards, aggregates
eval_mc3/vram.py             rocm-smi peak sampler
eval_mc3/generate_corpus.py  seeded synthetic corpora with every trap + oracle evidence
eval_mc3/gates.py            GPU session-1 gates
release/build_layers.py      app / deps / weights layer tarballs
release/publish.sh           crane append + mutate (CI)
release/check_image.py       base-prefix, uncompressed size, config, secrets
release/rehearse.sh          notebook rehearsal of the pushed image
release/contract_check.sh    CPU contract run inside the CI container
docker/mc3-contract.Dockerfile   CPU contract image for CI (not the submission)
docker/Dockerfile.mc3        reference-only Dockerfile (equivalent filesystem)
.github/workflows/mc3-contract.yml   unit tests + grader-isolation run on every push
.github/workflows/mc3-release.yml    manual release
tools/amd-gpu/remote-mc3.js  GPU workflow
tests/conftest.py            (modified) kit fixtures
tests/sb_helpers.py          pool test target + docx writer
tests/fixtures/kit_fake.py   FakeEngine script for the kit (test fixture only)
tests/test_sb_*.py           the new tests
```

---

### Task 1: Scaffold, starter kit and fixtures

Spec: §1 (modules), Testing Decisions (prior art: ROADREAD fake-engine tests).

**Files:**
- Create: `sourcebound/__init__.py`, `eval_mc3/__init__.py`, `release/__init__.py`, `tests/fixtures/__init__.py`, `mc3/requirements.txt`, `eval_mc3/kit/` (from the zip)
- Modify: `tests/conftest.py`, `pyproject.toml`, `.gitignore`

- [ ] **Step 1: Unpack the starter kit into the repo** (the corpus is about 130 KB; the empty `archive/` is recreated by the fixture because git cannot track it)

```bash
python - <<'EOF'
import zipfile, shutil, pathlib
z = zipfile.ZipFile("mc3-starter-kit.zip"); z.extractall("_kit_tmp")
dst = pathlib.Path("eval_mc3/kit"); dst.mkdir(parents=True, exist_ok=True)
shutil.copytree("_kit_tmp/mc3-starter-kit/mc3-corpus", dst / "mc3-corpus", dirs_exist_ok=True)
shutil.copy("_kit_tmp/mc3-starter-kit/sample-questions.json", dst)
shutil.rmtree("_kit_tmp")
EOF
ls eval_mc3/kit eval_mc3/kit/mc3-corpus
```

Expected: `mc3-corpus  sample-questions.json` and `archive engineering logs planning specs support vendor`.

- [ ] **Step 2: Create the package markers**

`sourcebound/__init__.py`:

````python
"""SOURCEBOUND: retrieval-augmented answering with exact citations (AMD Mini-Challenge 3).

See docs/MINI_CHALLENGE_3_SPEC.md. Modules that the thin client imports (protocol) are stdlib only.
"""
````

Create empty `eval_mc3/__init__.py`, `release/__init__.py` and `tests/fixtures/__init__.py`.

Create `mc3/requirements.txt` now. The release layer builder (Task 12) packs it as `/app/requirements.txt` until Task 15 freezes the lock.

````text
# SOURCEBOUND runtime dependencies (top level). `node tools/amd-gpu/remote-mc3.js setup` resolves these on the
# GPU notebook (same base image as the release), purges anything that would shadow the base's ROCm torch,
# numpy or Pillow, and freezes the exact set into mc3/requirements.lock. The release installs the LOCK with
# --no-deps. NEVER add torch, torchvision, torchaudio, triton, numpy or pillow here.
transformers==5.17.0
pymupdf
openpyxl
````

- [ ] **Step 3: Replace `tests/conftest.py`.** This is a superset of the current file: `tmp_store_dir` is kept for the SILENTPATH tests.

````python
"""Shared fixtures. Nothing in tests/ requires a GPU."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parent.parent / "eval_mc3" / "kit"


@pytest.fixture
def tmp_store_dir(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture
def kit_corpus(tmp_path) -> Path:
    """A private copy of the MC3 sample corpus, with the empty archive/ directory a zip cannot carry."""
    dst = tmp_path / "corpus"
    shutil.copytree(KIT / "mc3-corpus", dst)
    (dst / "archive").mkdir(exist_ok=True)
    return dst


@pytest.fixture
def kit_questions() -> list[dict]:
    return json.loads((KIT / "sample-questions.json").read_text(encoding="utf-8"))["queries"]
````

- [ ] **Step 4: Edit `pyproject.toml`.** Make `include = ["roadread*", "silentpath*", "gate*"]` read `include = ["roadread*", "silentpath*", "gate*", "sourcebound*"]`, and make the `dev` extra read `dev = ["pytest>=8", "pymupdf>=1.24", "openpyxl>=3.1", "numpy>=1.26"]`. Then append to `.gitignore`:

```
eval_mc3/data/
layers/
stage/
_kit_tmp/
```

- [ ] **Step 5: Install the local dependencies and run the existing suite**

Run: `pip install pytest pymupdf openpyxl numpy pillow && python -m pytest -q`
Expected: `162 passed` (nothing new yet; the fixtures are unused).

- [ ] **Step 6: Commit**

```bash
git add sourcebound/__init__.py eval_mc3/__init__.py release/__init__.py tests/fixtures/__init__.py mc3/requirements.txt eval_mc3/kit tests/conftest.py pyproject.toml .gitignore
git commit -m "chore(mc3): scaffold sourcebound package and starter-kit fixtures" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: Grader normalization and answer shaping

Spec: §7 (answer shaping), Further Notes A (normalization)

**Files:**
- Create: `sourcebound/normalize.py`
- Test: `tests/test_sb_normalize.py`

- [ ] **Step 1: Write the failing test** — `tests/test_sb_normalize.py`:

````python
import pytest

from sourcebound.normalize import align_number, complete_qualifier, contains_answer, official, shape


def test_official_matches_grader_rules():
    assert official("Q3 FY27") == official("q3fy27") == "Q3FY27"
    assert official("ORR-FAN-2214-B") == "ORRFAN2214B"
    assert official("4.3.2") == "432"


@pytest.mark.parametrize("raw,want", [
    ("94 °C", "94"), ("94°C", "94"), ("94 C", "94"), ("180 seconds", "180"), ("180s", "180"),
    ("4.1 TB/s", "4.1"), ("$84.50", "84.50"), ("`E7731`", "E7731"), ('"REV-C2".', "REV-C2"),
    ("pin B14", "B14"), ("Version: 4.3.2", "4.3.2"), ("4.3.2", "4.3.2"), ("12V-2x6", "12V-2x6"),
    ("Q3 FY27", "Q3 FY27"), ("PIN-7", "PIN-7"), ("10,000", "10,000"),
])
def test_shape_keeps_value_drops_label_and_unit(raw, want):
    assert shape(raw) == want


def test_contains_answer_respects_token_boundaries():
    assert contains_answer("94", "Maximum junction temperature .......... 94 C")
    assert not contains_answer("94", "batch 1940 accepted")
    assert contains_answer("4.3.2", "status: closed | fixed_in: 4.3.2")
    assert not contains_answer("4.3", "fixed_in: 4.3.2")
    assert contains_answer("Q3 FY27", "enters customer sampling in Q3 FY27.")
    assert contains_answer("Q3FY27", "in Q3 FY27")


def test_complete_qualifier_expands_only_truncated_identifiers():
    assert complete_qualifier("Q3", "TQ-60 enters customer sampling in Q3 FY27.") == "Q3 FY27"
    assert complete_qualifier("C2", "BOARD REVISION: REV-C2") == "REV-C2"
    assert complete_qualifier("REV-C2", "BOARD REVISION: REV-C2") == "REV-C2"
    assert complete_qualifier("40", "MODEL: TQ-40") == "40"            # numbers are never expanded
    assert complete_qualifier("B14", "B14: THERM_ALERT#") == "B14"     # already standalone
    assert complete_qualifier("4.3.2", "Meridian 4.3.2") == "4.3.2"


def test_align_number_prefers_the_printed_form():
    assert align_number("84.50", "unit_cost_usd: 84.5") == "84.5"
    assert align_number("94", "94 C") == "94"
    assert align_number("E7731", "E7731") == "E7731"
````

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest -q tests/test_sb_normalize.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'sourcebound.normalize'`

- [ ] **Step 3: Implement**

`sourcebound/normalize.py`:

````python
"""Official answer normalization and deterministic answer shaping (spec §7, Further Notes A).

Nothing here may know about a particular corpus: no answer tables, no sample-specific rules.
"""
from __future__ import annotations

import re
import unicodedata

_DROP = re.compile(r"[\s\-.·_]")
_PUNCT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                        "—": "-", "−": "-", " ": " ", "…": "..."})


def official(s: str) -> str:
    """The grader's normalization: uppercase, drop all whitespace and the characters - . · _"""
    return _DROP.sub("", (s or "").upper())


def plain(s: str) -> str:
    """Loose form used to match quotes against documents: NFKC, ASCII quotes/dashes, lowercase, single spaces."""
    s = unicodedata.normalize("NFKC", s or "").translate(_PUNCT)
    return " ".join(s.lower().split())


_LABEL = re.compile(r"^(?:the\s+)?(?:answer|value|pin|version|firmware(?:\s+version)?|release|part(?:\s+number|\s+no\.?)?"
                    r"|error\s+code|code)(?:\s*[:#=]\s*|\s+is\s+|\s+)(?=\S)", re.I)
_UNITS = {"c", "°c", "°f", "f", "k", "°", "deg", "degc", "degree", "degrees", "degrees c", "s", "sec", "secs", "second",
          "seconds", "ms", "min", "mins", "minute", "minutes", "h", "hr", "hrs", "hour", "hours", "d", "day", "days",
          "week", "weeks", "w", "kw", "mw", "v", "mv", "a", "ma", "hz", "khz", "mhz", "ghz", "b", "kb", "mb", "gb",
          "tb", "kib", "mib", "gib", "tib", "gb/s", "tb/s", "mb/s", "gbps", "mbps", "%", "mm", "cm", "m", "in", "kg",
          "g", "lb", "lbs", "usd", "eur", "units", "unit", "pcs", "rpm", "nm", "ns", "us", "µs"}
_NUM_UNIT = re.compile(r"^([-+]?\d[\d,]*(?:\.\d+)?)\s*(.*)$")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w]|\.\d)")


def shape(answer: str) -> str:
    """Reduce a model's answer to the value only: no wrapper punctuation, label or unit."""
    a = unicodedata.normalize("NFKC", answer or "").translate(_PUNCT).strip()
    prev = None
    while prev != a:
        prev = a
        a = a.strip("`'\" ").rstrip(".,;:").strip()
    a = _LABEL.sub("", a, count=1).strip()
    a = re.sub(r"^(?:\$|usd\s*|eur\s*|€)\s*(?=\d)", "", a, flags=re.I)
    m = _NUM_UNIT.match(a)
    if m and m.group(2) and m.group(2).strip().lower().rstrip(".") in _UNITS:
        a = m.group(1)
    return " ".join(a.split())


def answer_pattern(answer: str) -> re.Pattern | None:
    """Regex that finds `answer` in text whatever its separators, never inside a longer token (4.3 ≠ 4.3.2)."""
    core = official(answer)
    if not core:
        return None
    body = r"[\s\-.·_]*".join(re.escape(ch) for ch in core)
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9]|[.\-_][A-Za-z0-9])", re.I)


def contains_answer(answer: str, text: str) -> bool:
    p = answer_pattern(answer)
    return bool(p and p.search(unicodedata.normalize("NFKC", text or "")))


_QUARTER = re.compile(r"^Q([1-4])$", re.I)
_JOINED = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+")


def complete_qualifier(answer: str, quote: str) -> str:
    """Expand a truncated identifier to the full token printed in the evidence: Q3 -> Q3 FY27, C2 -> REV-C2.

    Only answers containing a letter are expanded (numbers never are), and only when the answer never occurs
    on its own in the quote."""
    if not answer or not quote or not re.search(r"[A-Za-z]", answer):
        return answer
    m = _QUARTER.match(answer.strip())
    if m:
        q = re.search(rf"\bQ{m.group(1)}\s*(?:FY|CY)\s*'?\d{{2,4}}\b", quote, re.I)
        return " ".join(q.group(0).split()) if q else answer
    alone = re.compile(r"(?<![A-Za-z0-9_\-])" + re.escape(answer) + r"(?![A-Za-z0-9_\-])", re.I)
    if alone.search(quote):
        return answer
    key = official(answer)
    for tok in _JOINED.findall(quote):
        parts = [official(p) for p in re.split(r"[-_]", tok)]
        for i in range(len(parts)):
            for j in range(i + 1, len(parts) + 1):
                if "".join(parts[i:j]) == key and (i > 0 or j < len(parts)):
                    return tok
    return answer


def align_number(answer: str, quote: str) -> str:
    """If the answer is a number written differently from the evidence (84.50 vs 84.5), use the evidence's form."""
    try:
        val = float(answer.replace(",", ""))
    except ValueError:
        return answer
    for lit in _NUMBER.findall(quote or ""):
        try:
            if float(lit.replace(",", "")) == val:
                return lit
        except ValueError:
            continue
    return answer
````

- [ ] **Step 4: Run it and watch it pass**

Run: `python -m pytest -q tests/test_sb_normalize.py`
Expected: `20 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_sb_normalize.py sourcebound/normalize.py
git commit -m "feat(mc3): grader normalization and value-only answer shaping" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: Identifier-aware terms and the supersession model

Spec: §4 (status flags, families), §5 (tokenizer, bridge identifiers)

**Files:**
- Create: `sourcebound/terms.py`
- Create: `sourcebound/records.py`
- Test: `tests/test_sb_terms_records.py`

Note the r2 head used in the test: it contains "Revision 1 is withdrawn." on purpose. That is defect #1 from the verification table.

- [ ] **Step 1: Write the failing test** — `tests/test_sb_terms_records.py`:

````python
from sourcebound.records import FileRecord, assign_status, family_of, revision_of
from sourcebound.terms import identifiers, terms


def test_terms_keep_codes_whole_and_split():
    t = terms("incident logged against ORR-1847 (E7731) on 4.3.2")
    for want in ("orr-1847", "orr", "1847", "orr1847", "e7731", "4.3.2"):
        assert want in t
    assert "the" not in terms("what is the value", query=True)


def test_identifiers_are_letter_led_codes_with_digits():
    ids = identifiers("ERROR E7731: throttle; incident logged against ORR-1847; fan ORR-FAN-2214-B; die 0 at 91C")
    assert ids == ["E7731", "ORR-1847", "ORR-FAN-2214-B"]
    assert identifiers("2026-09-02T03:14:41Z meridian[2211]") == []


def test_revision_and_family():
    assert revision_of("specs/tq40_datasheet_r2.pdf")[0] == "2"
    assert revision_of("spec_revB.pdf")[0] == "B"
    assert revision_of("support/rma_parts.xlsx") == ("", ())
    assert family_of("specs/tq40_datasheet_r1_WITHDRAWN.pdf") == family_of("specs/tq40_datasheet_r2.pdf")
    assert family_of("logs/a_2026-09-02.log") != family_of("logs/a_2026-09-03.log")


def _recs(*names):
    return {n: FileRecord(rel=n, path=n, ftype="pdf") for n in names}


def test_withdrawn_by_name_points_to_current_sibling():
    r = _recs("specs/tq40_datasheet_r1_WITHDRAWN.pdf", "specs/tq40_datasheet_r2.pdf")
    assign_status(r, {"specs/tq40_datasheet_r2.pdf": "Datasheet, revision 2 - supersedes revision 1\n...\nRevision 1 is withdrawn."})
    assert r["specs/tq40_datasheet_r1_WITHDRAWN.pdf"].status == "WITHDRAWN"
    assert r["specs/tq40_datasheet_r1_WITHDRAWN.pdf"].superseded_by == "specs/tq40_datasheet_r2.pdf"
    assert r["specs/tq40_datasheet_r2.pdf"].status == "CURRENT"


def test_withdrawn_by_heading_and_older_revision_superseded():
    r = _recs("a/spec_v1.pdf", "a/spec_v2.pdf", "a/spec_v3.pdf")
    assign_status(r, {"a/spec_v3.pdf": "Spec v3\nWITHDRAWN - do not use", "a/spec_v1.pdf": "Spec v1", "a/spec_v2.pdf": "Spec v2"})
    assert r["a/spec_v3.pdf"].status == "WITHDRAWN" and r["a/spec_v3.pdf"].superseded_by == "a/spec_v2.pdf"
    assert r["a/spec_v1.pdf"].status == "SUPERSEDED" and r["a/spec_v1.pdf"].superseded_by == "a/spec_v2.pdf"
    assert r["a/spec_v2.pdf"].status == "CURRENT"
````

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest -q tests/test_sb_terms_records.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'sourcebound.records'`

- [ ] **Step 3: Implement**

`sourcebound/terms.py`:

````python
"""Identifier-aware tokenization for lexical search, and identifier extraction for the bridge hop (spec §5)."""
from __future__ import annotations

import re

from .normalize import official

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_.#/][A-Za-z0-9]+)*")
_SPLIT = re.compile(r"[-_.#/]")
STOP = frozenset("""a an and are as at be by does do for from has have how in is it its of on or that the this to was
were what when where which who why with whose into than then there these those under over per via""".split())


def terms(text: str, query: bool = False) -> list[str]:
    """Whole compound tokens plus their parts and a de-punctuated form: ORR-1847 -> orr-1847, orr, 1847, orr1847."""
    out: list[str] = []
    for m in _TOKEN.finditer(text or ""):
        t = m.group(0).lower()
        if query and t in STOP:
            continue
        out.append(t)
        parts = [p for p in _SPLIT.split(t) if p]
        if len(parts) > 1:
            out.extend(p for p in parts if not (query and p in STOP))
            out.append("".join(parts))
    return out


_ID = re.compile(r"(?<![A-Za-z0-9_\-])[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*")


def identifiers(text: str) -> list[str]:
    """Code-like tokens that start with a letter and contain a digit: ORR-1847, E7731, ORR-FAN-2214-B, TQ-40."""
    seen, out = set(), []
    for m in _ID.finditer(text or ""):
        tok = m.group(0).strip("-_")
        if len(tok) < 3 or not re.search(r"\d", tok):
            continue
        k = id_key(tok)
        if k not in seen:
            seen.add(k)
            out.append(tok)
    return out


def id_key(tok: str) -> str:
    return official(tok)
````

`sourcebound/records.py`:

````python
"""Segments, file records, and the supersession model: status flags and document families (spec §4)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath


@dataclass
class Segment:
    file: str      # corpus-relative POSIX path
    locator: str   # "page 3", "Parts!4", "lines 1-24", "image", "row 7"
    kind: str      # prose | row | code | log | transcript
    text: str


@dataclass
class FileRecord:
    rel: str
    path: str
    ftype: str     # pdf | docx | xlsx | csv | text | code | image
    size: int = 0
    status: str = "CURRENT"   # CURRENT | WITHDRAWN | SUPERSEDED
    family: str = ""
    revision: str = ""
    superseded_by: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


_NAME_WITHDRAWN = re.compile(r"(?:^|[^a-z])(withdrawn|superseded|obsolete|deprecated)(?:[^a-z]|$)", re.I)
_HEAD_WITHDRAWN = re.compile(r"\b(withdrawn|superseded by|obsolete|deprecated|do not use|do not design)\b", re.I)
_REV = re.compile(r"(?:^|[_\-\s.])(r|rev|revision|v|ver|version)[_\-\s]?([0-9]{1,3}|[a-z]\d?)(?=$|[_\-\s.])", re.I)
_STATUS_WORDS = re.compile(r"(?:^|[_\-\s])(withdrawn|superseded|obsolete|deprecated|draft|final|current|latest)(?=$|[_\-\s])", re.I)


# "supersedes revision 1" and "Revision 1 is withdrawn" describe ANOTHER document, not this one
_OTHER_DOC = re.compile(r"\bsupersedes\b|\b(?:revision|rev\.?|r|version|v)\s*[0-9a-z]{1,3}\s+(?:is|was|has been)\s+"
                        r"(?:withdrawn|superseded|obsolete|deprecated)", re.I)


def head_lines(text: str, n: int = 3) -> str:
    return "\n".join([ln for ln in (text or "").splitlines() if ln.strip()][:n])


def withdrawn_heading(text: str) -> bool:
    return any(_HEAD_WITHDRAWN.search(ln) and not _OTHER_DOC.search(ln) for ln in head_lines(text).splitlines())


def revision_of(name: str) -> tuple[str, tuple]:
    """('2', (0, 2)) for tq40_datasheet_r2; ('B', (1, 66, 0)) for spec_revB; ('', ()) when absent."""
    m = None
    for m in _REV.finditer(PurePosixPath(name).stem):
        pass
    if not m:
        return "", ()
    tok = m.group(2).upper()
    if tok.isdigit():
        return tok, (0, int(tok))
    return tok, (1, ord(tok[0]), int(tok[1:] or 0))


def family_of(rel: str) -> str:
    p = PurePosixPath(rel)
    stem = _REV.sub("", p.stem)
    stem = _STATUS_WORDS.sub("", stem)
    stem = re.sub(r"[_\-\s]+", "_", stem).strip("_").lower()
    return f"{p.parent.as_posix()}/{stem}{p.suffix.lower()}"


def assign_status(records: dict[str, FileRecord], heads: dict[str, str]) -> None:
    """WITHDRAWN from the file name or the first three lines; SUPERSEDED for an older revision in a family."""
    for rel, rec in records.items():
        name_hit = _NAME_WITHDRAWN.search(PurePosixPath(rel).stem.replace("_", " "))
        if name_hit or withdrawn_heading(heads.get(rel, "")):
            rec.status = "WITHDRAWN"
        rec.family = family_of(rel)
        rec.revision = revision_of(rel)[0]
    fams: dict[str, list[FileRecord]] = {}
    for rec in records.values():
        fams.setdefault(rec.family, []).append(rec)
    for members in fams.values():
        if len(members) < 2:
            continue
        live = [r for r in members if r.status != "WITHDRAWN" and revision_of(r.rel)[1]]
        if not live:
            live = [r for r in members if r.status != "WITHDRAWN"]
        if not live:
            continue
        current = max(live, key=lambda r: revision_of(r.rel)[1] or (-1,))
        for r in members:
            if r is current:
                continue
            older = revision_of(r.rel)[1] and revision_of(current.rel)[1] and revision_of(r.rel)[1] < revision_of(current.rel)[1]
            if r.status == "WITHDRAWN" or older:
                r.status = "WITHDRAWN" if r.status == "WITHDRAWN" else "SUPERSEDED"
                r.superseded_by = current.rel
````

- [ ] **Step 4: Run it and watch it pass**

Run: `python -m pytest -q tests/test_sb_terms_records.py`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_sb_terms_records.py sourcebound/terms.py sourcebound/records.py
git commit -m "feat(mc3): identifier tokens, document families and withdrawal status" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: Defensive corpus walk

Spec: §3, user stories 10–14

**Files:**
- Create: `sourcebound/walk.py`
- Test: `tests/test_sb_walk.py`

`opener` is the injection point for the Windows permission test. The real `chmod 000` case (file **and** directory) runs on Linux CI.

- [ ] **Step 1: Write the failing test** — `tests/test_sb_walk.py`:

````python
import os
import sys

import pytest

from sourcebound.walk import OLE, detect, walk


def test_kit_hazards_are_skipped_and_everything_else_found(kit_corpus):
    files, ledger = walk(kit_corpus)
    rels = {f.rel for f in files}
    assert "vendor/telemetry_capture.dat" not in rels
    assert {"specs/tq40_datasheet_r2.pdf", "planning/roadmap_fy27.docx", "support/rma_parts.xlsx",
            "support/bug_database.csv", "logs/prod_inference_2026-09-02.log", "engineering/ingest_service.py",
            "specs/backplane_pinout.png", "support/asset_label.jpg"} <= rels
    assert any(e["path"] == "vendor/telemetry_capture.dat" and "unknown type" in e["reason"] for e in ledger)


def _deny(name):
    def opener(path, mode="r", *a, **k):
        if os.path.basename(path) == name:
            raise PermissionError(13, "Permission denied", path)
        return open(path, mode, *a, **k)
    return opener


@pytest.mark.parametrize("victim", ["internal_audit.txt", "asset_label.jpg", "backplane_pinout.png"])
def test_unreadable_file_never_stops_the_walk(kit_corpus, victim):
    files, ledger = walk(kit_corpus, opener=_deny(victim))
    rels = {f.rel for f in files}
    assert all(os.path.basename(r) != victim for r in rels)
    assert len(rels) == 11                         # 13 files - unknown .dat - the denied one
    assert any(victim in e["path"] and "PermissionError" in e["reason"] for e in ledger)


@pytest.mark.skipif(sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="chmod 000 only blocks a non-root POSIX user")
def test_real_chmod_000_file_and_directory(kit_corpus):
    (kit_corpus / "vendor" / "internal_audit.txt").chmod(0)
    locked = kit_corpus / "zz_locked"
    locked.mkdir()
    (locked / "x.txt").write_text("hidden")
    locked.chmod(0)
    try:
        files, ledger = walk(kit_corpus)
        assert "vendor/internal_audit.txt" not in {f.rel for f in files}
        assert any(e["path"] == "zz_locked/" for e in ledger)
    finally:
        locked.chmod(0o755)


def test_detect_magic_bytes():
    assert detect(".pdf", b"%PDF-1.7") == ("pdf", "")
    assert detect(".pdf", b"hello")[0] is None
    assert detect(".xlsx", OLE)[0] is None                 # encrypted OOXML is an OLE container
    assert detect(".docx", b"PK\x03\x04") == ("docx", "")
    assert detect(".txt", b"abc\x00def")[0] is None
    assert detect(".md", b"# notes") == ("text", "")
    assert detect(".dat", bytes(range(16)))[0] is None
    assert detect("", b"plain")[0] is None


def test_empty_corpus_and_empty_file(tmp_path):
    (tmp_path / "empty_dir").mkdir()
    (tmp_path / "zero.txt").write_bytes(b"")
    files, ledger = walk(tmp_path)
    assert files == [] and ledger == [{"path": "zero.txt", "reason": "empty file"}]
````

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest -q tests/test_sb_walk.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'sourcebound.walk'`

- [ ] **Step 3: Implement**

`sourcebound/walk.py`:

````python
"""Defensive corpus walk (spec §3). One bad file or directory never stops the walk.

Every skipped path lands in the ledger with a reason, so a run can be audited afterwards.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

from .records import FileRecord

KNOWN = {".pdf": "pdf", ".docx": "docx", ".docm": "docx", ".xlsx": "xlsx", ".xlsm": "xlsx", ".csv": "csv",
         ".tsv": "csv", ".txt": "text", ".log": "text", ".py": "code", ".png": "image", ".jpg": "image",
         ".jpeg": "image", ".tif": "image", ".tiff": "image", ".bmp": "image", ".gif": "image", ".webp": "image"}
TEXT_ALLOW = {".md": "text", ".rst": "text", ".json": "text", ".yaml": "text", ".yml": "text", ".toml": "text",
              ".ini": "text", ".cfg": "text", ".conf": "text", ".html": "text", ".htm": "text", ".xml": "text",
              ".sh": "code", ".js": "code", ".ts": "code", ".c": "code", ".h": "code", ".cpp": "code",
              ".java": "code", ".go": "code", ".rs": "code", ".sql": "code"}
MAX_BYTES = int(os.environ.get("SB_MAX_FILE_BYTES", str(64 * 2**20)))
OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def detect(ext: str, head: bytes) -> tuple[str | None, str]:
    """(ftype, "") or (None, reason). The extension proposes, the magic bytes confirm."""
    if head.startswith(OLE):
        return None, "OLE container (encrypted Office file or legacy format)"
    kind = KNOWN.get(ext)
    if kind == "pdf":
        return ("pdf", "") if head.startswith(b"%PDF") else (None, "extension .pdf but not a PDF")
    if kind in ("docx", "xlsx"):
        return (kind, "") if head.startswith(b"PK") else (None, f"extension {ext} but not an OOXML zip")
    if kind == "image":
        return "image", ""
    if kind in ("csv", "text", "code") or ext in TEXT_ALLOW:
        if b"\x00" in head:
            return None, "binary content in a text-type file"
        if kind is None:
            sample = head.decode("utf-8", errors="replace")
            printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in sample)
            if sample and printable / len(sample) < 0.95:
                return None, "allow-listed extension but not printable text"
            return TEXT_ALLOW[ext], ""
        return kind, ""
    return None, f"unknown type {ext or '(no extension)'}"


def zip_problem(path: str) -> str:
    """Reason to refuse an OOXML zip (encrypted members, zip bomb), or ''."""
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        if any(i.flag_bits & 0x1 for i in infos):
            return "encrypted zip member"
        if sum(i.file_size for i in infos) > 512 * 2**20:
            return "uncompressed size over 512 MiB"
    return ""


def walk(root: Path, opener=open) -> tuple[list[FileRecord], list[dict]]:
    """Return readable, typed files under root (sorted, symlinks not followed) and a ledger of skips."""
    root = Path(root)
    files: list[FileRecord] = []
    ledger: list[dict] = []

    def skip(rel: str, reason: str) -> None:
        ledger.append({"path": rel, "reason": reason})

    def visit(d: Path) -> None:
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError as e:
            skip(_rel(root, d) + "/", f"cannot list directory: {type(e).__name__}")
            return
        for e in entries:
            rel = _rel(root, Path(e.path))
            try:
                if e.is_symlink():
                    skip(rel, "symlink not followed")
                elif e.is_dir(follow_symlinks=False):
                    visit(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    rec = _probe(e, rel, opener, skip)
                    if rec:
                        files.append(rec)
                else:
                    skip(rel, "not a regular file")
            except Exception as ex:  # noqa: BLE001 - one entry must never stop the walk
                skip(rel, f"{type(ex).__name__}: {ex}")

    visit(root)
    return files, ledger


def _probe(e: os.DirEntry, rel: str, opener, skip) -> FileRecord | None:
    size = e.stat(follow_symlinks=False).st_size
    if size == 0:
        skip(rel, "empty file")
        return None
    if size > MAX_BYTES:
        skip(rel, f"larger than {MAX_BYTES} bytes")
        return None
    try:
        with opener(e.path, "rb") as f:
            head = f.read(4096)
    except OSError as ex:
        skip(rel, f"unreadable: {type(ex).__name__}")
        return None
    ftype, reason = detect(Path(e.name).suffix.lower(), head)
    if not ftype:
        skip(rel, reason)
        return None
    return FileRecord(rel=rel, path=e.path, ftype=ftype, size=size)


def _rel(root: Path, p: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()
````

- [ ] **Step 4: Run it and watch it pass**

Run: `python -m pytest -q tests/test_sb_walk.py`
Expected: `6 passed, 1 skipped` on Windows or as root; `7 passed` on Linux as a non-root user (the CI unit job)

- [ ] **Step 5: Commit**

```bash
git add tests/test_sb_walk.py sourcebound/walk.py
git commit -m "feat(mc3): defensive corpus walk with skip ledger" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: Per-type parsers

Spec: §4, user stories 16–26

**Files:**
- Create: `sourcebound/parsers.py`
- Create: `tests/sb_helpers.py`
- Test: `tests/test_sb_parsers.py`

Create `tests/sb_helpers.py` before Step 1. The test imports its `write_docx`, and Task 6 uses its pool target:

````python
"""Helpers that ParserPool children import by dotted path (they must live in an importable module)."""
import os
import time


def slow_or_fast(path, rel, ftype):
    if "hang" in rel:
        time.sleep(60)
    if "crash" in rel:
        os._exit(3)
    if "raise" in rel:
        raise ValueError("bad file")
    return {"segments": [], "head": rel, "ocr_pages": [], "media": []}


def write_docx(path, paragraphs, table=None):
    """Minimal OOXML writer: paragraphs, then one table (first row = header)."""
    import zipfile
    from xml.sax.saxutils import escape

    def p(t):
        return f"<w:p><w:r><w:t xml:space=\"preserve\">{escape(t)}</w:t></w:r></w:p>"
    body = "".join(p(t) for t in paragraphs)
    if table:
        rows = "".join("<w:tr>" + "".join(f"<w:tc>{p(c)}</w:tc>" for c in r) + "</w:tr>" for r in table)
        body += f"<w:tbl>{rows}</w:tbl>"
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           f"<w:body>{body}</w:body></w:document>")
    ct = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", doc)
````

- [ ] **Step 1: Write the failing test** — `tests/test_sb_parsers.py`:

````python
import pytest

from sourcebound.parsers import Encrypted, parse_file
from tests.sb_helpers import write_docx


def texts(res):
    return "\n".join(s["text"] for s in res["segments"])


def test_pdf_pages_and_head(kit_corpus):
    r = parse_file(str(kit_corpus / "specs/tq40_datasheet_r2.pdf"), "specs/tq40_datasheet_r2.pdf", "pdf")
    assert "Maximum junction temperature .......... 94 C" in texts(r)
    assert r["segments"][0]["locator"] == "page 1"
    assert "supersedes revision 1" in r["head"]


def test_password_pdf_is_never_read(kit_corpus):
    with pytest.raises(Encrypted):
        parse_file(str(kit_corpus / "vendor/supplier_agreement_ENCRYPTED.pdf"), "x.pdf", "pdf")


def test_owner_password_only_pdf_is_also_refused(tmp_path):
    fitz = pytest.importorskip("fitz")
    d = fitz.open()
    d.new_page().insert_text((72, 72), "unit price 6412")
    d.save(tmp_path / "owner.pdf", encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="")
    assert fitz.open(tmp_path / "owner.pdf").is_encrypted is False     # why the metadata check exists
    with pytest.raises(Encrypted):
        parse_file(str(tmp_path / "owner.pdf"), "owner.pdf", "pdf")


def test_docx_paragraphs_and_tables(kit_corpus, tmp_path):
    r = parse_file(str(kit_corpus / "planning/roadmap_fy27.docx"), "planning/roadmap_fy27.docx", "docx")
    assert "TQ-60 enters customer sampling in Q3 FY27." in texts(r)
    write_docx(tmp_path / "t.docx", ["Milestones"], [["product", "milestone", "quarter"], ["KV-9", "tape-out", "Q1 FY28"]])
    r = parse_file(str(tmp_path / "t.docx"), "t.docx", "docx")
    assert "table 1 | product: KV-9 | milestone: tape-out | quarter: Q1 FY28" in texts(r)


def test_xlsx_every_sheet_with_headers_on_every_row(kit_corpus):
    r = parse_file(str(kit_corpus / "support/rma_parts.xlsx"), "support/rma_parts.xlsx", "xlsx")
    t = texts(r)
    assert "part_number: ORR-FAN-2214-B | description: Fan assembly, dual-rotor, field replaceable | compatible_with: TQ-40" in t
    assert "unit_cost_usd: 84.5" in t
    assert "Lead times are supplier-quoted" in t                    # the second sheet
    assert {s["kind"] for s in r["segments"]} >= {"row"}


def test_csv_rows_carry_headers(kit_corpus):
    r = parse_file(str(kit_corpus / "support/bug_database.csv"), "support/bug_database.csv", "csv")
    assert any(s["text"].startswith("ticket: ORR-1847 |") and "fixed_in: 4.3.2" in s["text"] for s in r["segments"])


def test_semicolon_latin1_csv(tmp_path):
    (tmp_path / "t.csv").write_bytes("part;cost\nGRÜN-1;12,5\n".encode("cp1252"))
    r = parse_file(str(tmp_path / "t.csv"), "t.csv", "csv")
    assert texts(r) == "part: GRÜN-1 | cost: 12,5"


def test_log_and_code(kit_corpus, tmp_path):
    r = parse_file(str(kit_corpus / "logs/prod_inference_2026-09-02.log"), "l.log", "text")
    assert "ERROR E7731: thermal throttle engaged" in texts(r)
    r = parse_file(str(kit_corpus / "engineering/ingest_service.py"), "i.py", "code")
    assert "DEFAULT_BATCH_TIMEOUT_S = 180" in texts(r) and "Raised from 60" in texts(r)
    big = "\n".join([f"# constant {i}\nC{i} = {i}" for i in range(200)]) + "\n\ndef f():\n    return 1\n"
    (tmp_path / "big.py").write_text(big)
    r = parse_file(str(tmp_path / "big.py"), "big.py", "code")
    assert any(s["locator"] == "module" and "C199 = 199" in s["text"] for s in r["segments"])
    assert any("def f()" in s["text"] for s in r["segments"])


def test_long_log_is_windowed(tmp_path):
    lines = [f"2026-09-02T03:{i // 60:02d}:{i % 60:02d}Z node INFO tick {i}" for i in range(100)]
    (tmp_path / "a.log").write_text("\n".join(lines))
    r = parse_file(str(tmp_path / "a.log"), "a.log", "text")
    assert len(r["segments"]) > 3 and all(s["kind"] == "log" for s in r["segments"])
    assert "tick 99" in texts(r)


def test_image_is_validated_not_read(kit_corpus):
    r = parse_file(str(kit_corpus / "support/asset_label.jpg"), "support/asset_label.jpg", "image")
    assert r["image"] is True and r["segments"] == []
````

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest -q tests/test_sb_parsers.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'sourcebound.parsers'` (create `tests/sb_helpers.py` first, as shown, because the test imports its docx writer)

- [ ] **Step 3: Implement**

`sourcebound/parsers.py`:

````python
"""Per-type parsers (spec §4). They run inside ParserPool children, so they may hang or crash safely.

parse_file(path, rel, ftype) -> {"segments": [dict], "head": str, "ocr_pages": [int], "media": [str]}
Raises Encrypted for any encrypted container: such files are never indexed.
"""
from __future__ import annotations

import ast
import csv
import io
import re
import xml.etree.ElementTree as ET
import zipfile

from .walk import zip_problem

CHUNK = 1800          # characters per prose segment
WHOLE_TEXT = 2000     # text/log files up to this size are one segment
WHOLE_CODE = 3000
MAX_PAGES = 300
MAX_ROWS = 20000


class Encrypted(Exception):
    pass


def seg(rel: str, locator: str, kind: str, text: str) -> dict:
    return {"file": rel, "locator": locator, "kind": kind, "text": text.strip()}


def read_text(path: str) -> str:
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def split_text(text: str, limit: int = CHUNK) -> list[str]:
    """Split on blank lines, then lines, keeping chunks under `limit` characters."""
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], ""
    for p in paras:
        pieces = [p] if len(p) <= limit else [p[i:i + limit] for i in range(0, len(p), limit)]
        for piece in pieces:
            if buf and len(buf) + len(piece) + 2 > limit:
                out.append(buf)
                buf = ""
            buf = f"{buf}\n\n{piece}" if buf else piece
    if buf:
        out.append(buf)
    return out


def rows_to_segments(rel: str, rows: list[list[str]], prefix: str, locator_fmt: str) -> list[dict]:
    """Header row + data rows -> one 'h: v | h: v' segment per row (header names repeated on every row)."""
    rows = [[(c or "").strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    hi = next((i for i, r in enumerate(rows) if sum(1 for c in r if c) >= 2), None)
    if hi is None:
        return [seg(rel, locator_fmt.format(1), "prose", prefix + "\n".join(" ".join(c for c in r if c) for r in rows))]
    out = []
    if hi > 0:
        out.append(seg(rel, locator_fmt.format(1), "prose", prefix + "\n".join(" ".join(c for c in r if c) for r in rows[:hi])))
    header = [h or f"col{j + 1}" for j, h in enumerate(rows[hi])]
    for i, r in enumerate(rows[hi + 1:], start=hi + 2):
        cells = [f"{header[j] if j < len(header) else f'col{j + 1}'}: {c}" for j, c in enumerate(r) if c]
        if cells:
            out.append(seg(rel, locator_fmt.format(i), "row", prefix + " | ".join(cells)))
    return out


# ---------------------------------------------------------------- PDF
def parse_pdf(path: str, rel: str) -> dict:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    try:
        # needs_pass: user password required. is_encrypted / metadata["encryption"]: owner-password-only PDFs
        # open without a password and report is_encrypted False, so the metadata check is required too.
        if doc.needs_pass or doc.is_encrypted or (doc.metadata or {}).get("encryption"):
            raise Encrypted("encrypted PDF")
        segs, ocr, head = [], [], ""
        for i, page in enumerate(doc):
            if i >= MAX_PAGES:
                break
            text = page.get_text("text", sort=True)
            if i == 0:
                head = text
            if len(text.strip()) < 20:
                if page.get_images() or page.get_drawings():
                    ocr.append(i)
                continue
            for j, chunk in enumerate(split_text(text)):
                segs.append(seg(rel, f"page {i + 1}" + (f" part {j + 1}" if j else ""), "prose", chunk))
        return {"segments": segs, "head": head, "ocr_pages": ocr, "media": []}
    finally:
        doc.close()


# ---------------------------------------------------------------- DOCX (stdlib only)
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _para_text(p: ET.Element) -> str:
    out = []
    for el in p.iter():
        if el.tag == W + "t" and el.text:
            out.append(el.text)
        elif el.tag == W + "tab":
            out.append("\t")
        elif el.tag in (W + "br", W + "cr"):
            out.append("\n")
    return "".join(out)


def _table_rows(tbl: ET.Element) -> list[list[str]]:
    rows = []
    for tr in tbl.findall(W + "tr"):
        rows.append([" ".join(_para_text(p) for p in tc.iter(W + "p")).strip() for tc in tr.findall(W + "tc")])
    return rows


def parse_docx(path: str, rel: str) -> dict:
    problem = zip_problem(path)
    if problem:
        raise Encrypted(problem)
    segs, paras, head = [], [], ""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        body = ET.fromstring(z.read("word/document.xml")).find(W + "body")
        tcount = 0

        def flush() -> None:
            if paras:
                for k, chunk in enumerate(split_text("\n".join(paras))):
                    segs.append(seg(rel, f"text {len(segs) + 1}", "prose", chunk))
                paras.clear()

        for el in list(body) if body is not None else []:
            if el.tag == W + "tbl":
                flush()
                tcount += 1
                segs.extend(rows_to_segments(rel, _table_rows(el), f"table {tcount} | ", f"table {tcount} row {{}}"))
            elif el.tag in (W + "p", W + "sdt"):
                for p in ([el] if el.tag == W + "p" else el.iter(W + "p")):
                    t = _para_text(p)
                    if t.strip():
                        paras.append(t)
                        if not head:
                            head = t
        flush()
        for part in sorted(n for n in names if re.match(r"word/(header|footer|footnotes|endnotes)\d*\.xml$", n)):
            text = "\n".join(t for t in (_para_text(p) for p in ET.fromstring(z.read(part)).iter(W + "p")) if t.strip())
            if text.strip():
                segs.append(seg(rel, part.split("/")[-1].removesuffix(".xml"), "prose", text))
        media = [n for n in names if n.startswith("word/media/") and n.lower().endswith((".png", ".jpg", ".jpeg"))]
    head = "\n".join(s["text"] for s in segs[:2]) or head
    return {"segments": segs, "head": head, "ocr_pages": [], "media": media}


# ---------------------------------------------------------------- XLSX
def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def parse_xlsx(path: str, rel: str) -> dict:
    import openpyxl

    problem = zip_problem(path)
    if problem:
        raise Encrypted(problem)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    segs = []
    try:
        for ws in wb.worksheets:
            rows = []
            for i, r in enumerate(ws.iter_rows(values_only=True)):
                if i >= MAX_ROWS:
                    break
                rows.append([_cell(v) for v in r])
            segs.extend(rows_to_segments(rel, rows, f"sheet {ws.title} | ", f"{ws.title}!{{}}"))
    finally:
        wb.close()
    return {"segments": segs, "head": "\n".join(s["text"] for s in segs[:2]), "ocr_pages": [], "media": []}


# ---------------------------------------------------------------- CSV
def parse_csv(path: str, rel: str) -> dict:
    text = read_text(path)
    try:
        dialect = csv.Sniffer().sniff(text[:65536], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = []
    for i, r in enumerate(csv.reader(io.StringIO(text), dialect)):
        if i > MAX_ROWS:
            break
        rows.append(r)
    segs = rows_to_segments(rel, rows, "", "row {}")
    return {"segments": segs, "head": text[:500], "ocr_pages": [], "media": []}


# ---------------------------------------------------------------- text, logs, code
def parse_text(path: str, rel: str) -> dict:
    text = read_text(path)
    if len(text) <= WHOLE_TEXT:
        segs = [seg(rel, "all", "prose", text)] if text.strip() else []
    elif rel.lower().endswith(".log") or _looks_like_log(text):
        lines = text.splitlines()
        segs = [seg(rel, f"lines {i + 1}-{min(i + 24, len(lines))}", "log", "\n".join(lines[i:i + 24]))
                for i in range(0, len(lines), 18) if "\n".join(lines[i:i + 24]).strip()]
    else:
        segs = [seg(rel, f"part {k + 1}", "prose", c) for k, c in enumerate(split_text(text))]
    return {"segments": segs, "head": text[:500], "ocr_pages": [], "media": []}


def _looks_like_log(text: str) -> bool:
    lines = [ln for ln in text.splitlines()[:50] if ln.strip()]
    return bool(lines) and sum(bool(re.match(r"^\[?\d{4}-\d{2}-\d{2}", ln)) for ln in lines) / len(lines) > 0.6


def parse_code(path: str, rel: str) -> dict:
    text = read_text(path)
    if len(text) <= WHOLE_CODE:
        return {"segments": [seg(rel, "all", "code", text)] if text.strip() else [], "head": text[:500],
                "ocr_pages": [], "media": []}
    lines = text.splitlines()
    segs = []
    try:
        tree = ast.parse(text) if rel.endswith(".py") else None
    except SyntaxError:
        tree = None
    if tree is None:
        segs = [seg(rel, f"lines {i + 1}-{min(i + 60, len(lines))}", "code", "\n".join(lines[i:i + 60]))
                for i in range(0, len(lines), 50)]
    else:
        spans = [(n.lineno, n.end_lineno) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        inside = {ln for a, b in spans for ln in range(a, (b or a) + 1)}
        module = [lines[i - 1] for i in range(1, len(lines) + 1) if i not in inside]
        segs.append(seg(rel, "module", "code", "\n".join(module)))
        for a, b in spans:
            # include decorators/comments directly above the definition
            start = a
            while start > 1 and lines[start - 2].strip().startswith(("#", "@")):
                start -= 1
            segs.append(seg(rel, f"lines {start}-{b}", "code", "\n".join(lines[start - 1:b])))
    return {"segments": [s for s in segs if s["text"]], "head": text[:500], "ocr_pages": [], "media": []}


# ---------------------------------------------------------------- images
def parse_image(path: str, rel: str) -> dict:
    from PIL import Image

    with Image.open(path) as im:
        im.verify()          # raises on truncated or corrupt files
    return {"segments": [], "head": "", "ocr_pages": [], "media": [], "image": True}


PARSERS = {"pdf": parse_pdf, "docx": parse_docx, "xlsx": parse_xlsx, "csv": parse_csv, "text": parse_text,
           "code": parse_code, "image": parse_image}


def parse_file(path: str, rel: str, ftype: str) -> dict:
    return PARSERS[ftype](path, rel)
````

- [ ] **Step 4: Run it and watch it pass**

Run: `python -m pytest -q tests/test_sb_parsers.py`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_sb_parsers.py sourcebound/parsers.py tests/sb_helpers.py
git commit -m "feat(mc3): pdf/docx/xlsx/csv/text/code/image parsers; encrypted files never read" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: Killable parser pool

Spec: §3 (per-file timeout), user story 15

**Files:**
- Create: `sourcebound/pool.py`
- Test: `tests/test_sb_pool.py`

- [ ] **Step 1: Write the failing test** — `tests/test_sb_pool.py`:

````python
import time

from sourcebound.pool import ParserPool


def test_hung_crashed_and_raising_jobs_do_not_block_the_rest():
    pool = ParserPool("tests.sb_helpers:slow_or_fast", workers=2, timeout_s=3)
    jobs = [(k, ("p", k, "text")) for k in ("a", "hang", "b", "crash", "c", "raise", "d")]
    t = time.monotonic()
    res = pool.run(jobs)
    assert time.monotonic() - t < 30
    assert res["hang"][0] == "err" and "timeout" in res["hang"][1]
    assert res["crash"] == ("err", "parser process died")
    assert res["raise"] == ("err", "ValueError: bad file")
    for k in "abcd":
        assert res[k] == ("ok", {"segments": [], "head": k, "ocr_pages": [], "media": []})


def test_deadline_marks_remaining_jobs():
    pool = ParserPool("tests.sb_helpers:slow_or_fast", workers=1, timeout_s=30)
    res = pool.run([("hang", ("p", "hang", "text")), ("b", ("p", "b", "text"))], deadline=time.monotonic() + 4)
    assert res["hang"][0] == "err" and res["b"][0] == "err"
````

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest -q tests/test_sb_pool.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'sourcebound.pool'`

- [ ] **Step 3: Implement**

`sourcebound/pool.py`:

````python
"""A small process pool whose stuck workers can be killed one at a time (spec §3, per-file timeout).

concurrent.futures cannot cancel a running task, so a hung PDF would hold a worker forever. Here each child
serves jobs over a Pipe; a child that overruns its timeout is terminated and replaced, and the job is recorded
as a failure. Children are started with "spawn" so they never inherit the GPU worker's HIP state.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import time
from multiprocessing.connection import wait


def _resolve(fn_path: str):
    mod, name = fn_path.split(":")
    return getattr(importlib.import_module(mod), name)


def _child(conn, fn_path: str) -> None:
    fn = _resolve(fn_path)
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        if job is None:
            return
        try:
            conn.send(("ok", fn(*job)))
        except BaseException as e:  # noqa: BLE001 - report every failure to the parent
            conn.send(("err", f"{type(e).__name__}: {e}"))


class ParserPool:
    def __init__(self, fn_path: str = "sourcebound.parsers:parse_file", workers: int = 4, timeout_s: float = 20.0):
        self.fn_path, self.workers, self.timeout_s = fn_path, max(1, workers), timeout_s
        self.ctx = mp.get_context("spawn")

    def _spawn(self) -> dict:
        parent, child = self.ctx.Pipe()
        p = self.ctx.Process(target=_child, args=(child, self.fn_path), daemon=True)
        p.start()
        child.close()
        return {"p": p, "c": parent, "job": None, "t": 0.0}

    def _replace(self, k: dict) -> None:
        try:
            k["p"].terminate()
            k["p"].join(2)
            k["c"].close()
        except Exception:
            pass
        k.update(self._spawn())

    def run(self, jobs: list[tuple[str, tuple]], deadline: float | None = None) -> dict[str, tuple[str, object]]:
        """jobs: [(key, args)]. Returns {key: ("ok", result) | ("err", reason)}. deadline is time.monotonic()."""
        results: dict[str, tuple[str, object]] = {}
        pending = list(jobs)
        kids = [self._spawn() for _ in range(min(self.workers, len(pending)))]
        try:
            while pending or any(k["job"] is not None for k in kids):
                if deadline is not None and time.monotonic() > deadline:
                    for key, _ in pending:
                        results[key] = ("err", "index deadline reached before parsing")
                    pending.clear()
                    for k in kids:
                        if k["job"] is not None:
                            results[k["job"]] = ("err", "index deadline reached during parsing")
                            self._replace(k)
                            k["job"] = None
                    break
                for k in kids:
                    if k["job"] is None and pending:
                        key, args = pending.pop(0)
                        try:
                            k["c"].send(args)
                        except (BrokenPipeError, OSError):
                            self._replace(k)
                            k["c"].send(args)
                        k["job"], k["t"] = key, time.monotonic()
                busy = [k["c"] for k in kids if k["job"] is not None]
                ready = wait(busy, timeout=0.1) if busy else []
                for k in kids:
                    if k["job"] is None:
                        continue
                    if k["c"] in ready:
                        try:
                            results[k["job"]] = k["c"].recv()
                        except (EOFError, OSError):
                            results[k["job"]] = ("err", "parser process died")
                            self._replace(k)
                        k["job"] = None
                    elif time.monotonic() - k["t"] > self.timeout_s:
                        results[k["job"]] = ("err", f"parse timeout after {self.timeout_s:.0f}s")
                        self._replace(k)
                        k["job"] = None
        finally:
            for k in kids:
                try:
                    k["c"].send(None)
                except Exception:
                    pass
            for k in kids:
                k["p"].join(1)
                if k["p"].is_alive():
                    k["p"].terminate()
        return results
````

- [ ] **Step 4: Run it and watch it pass**

Run: `python -m pytest -q tests/test_sb_pool.py`
Expected: `2 passed` (in about 10 s: it includes a 3 s timeout and a 4 s deadline)

- [ ] **Step 5: Commit**

```bash
git add tests/test_sb_pool.py sourcebound/pool.py
git commit -m "feat(mc3): parser pool that kills and replaces hung children" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 7: Index, FakeEngine and hybrid retrieval with the identifier bridge

Spec: §4–§5, §9 (transcription phase), Testing seam 2

**Files:**
- Create: `sourcebound/bm25.py`
- Create: `sourcebound/engine.py`
- Create: `sourcebound/index.py`
- Create: `sourcebound/retrieve.py`
- Create: `tests/fixtures/kit_fake.py`
- Test: `tests/test_sb_index_retrieve.py`

Create the FakeEngine script for the kit first. It is a **test fixture only** and is never imported by `sourcebound/`. Its replies include realistic model mistakes: citing the withdrawn datasheet, over-citing, and a near-miss unit cost.

````python
"""FakeEngine script for the starter-kit corpus. TEST FIXTURE ONLY: never imported by sourcebound/.

The replies imitate what a reader model plausibly returns, including its mistakes (citing the withdrawn
datasheet, over-citing the log, offering a near-miss unit cost), so the contract test proves that the
deterministic gates, not the script, produce the right answers and citation sets.
"""
import json
import sys
from pathlib import Path

TRANSCRIPTS = {
    "backplane_pinout.png": "TQ-40 backplane connector - pin assignment\nB11: GND\nB12: SDA\nB13: SCL\nB14: THERM_ALERT#\n"
                            "B15: PRSNT#\nB16: GND\nTHERM_ALERT# is asserted low when the die exceeds the warning threshold.\n"
                            "IMAGE: connector pin assignment diagram",
    "asset_label.jpg": "Orrery Systems\nMODEL: TQ-40\nBOARD REVISION: REV-C2\nSN: OS4-118823-77\nMADE IN MALAYSIA\n"
                       "IMAGE: photo of an asset label",
}


def _r(answer, *ev, status="answered"):
    return {"status": status, "answer": answer, "answer_type": "extracted",
            "evidence": [{"file": f, "quote": q, "role": role} for f, q, role in ev], "lookup": []}


R1 = "specs/tq40_datasheet_r1_WITHDRAWN.pdf"
R2 = "specs/tq40_datasheet_r2.pdf"
CSV = "support/bug_database.csv"
LOG = "logs/prod_inference_2026-09-02.log"
ROW = "ticket: ORR-1847 | component: power | severity: S2 | summary: Die temperature sensor reports high under sustained small-batch load | status: closed | fixed_in: 4.3.2"

REPLIES = [
    ("maximum junction temperature", [_r("105", (R1, "Maximum junction temperature .......... 105 C", "value")),
                                      _r("94 C", (R2, "Maximum junction temperature .......... 94 C", "value"))]),
    ("customer sampling", _r("Q3", ("planning/roadmap_fy27.docx", "TQ-60 enters customer sampling in Q3 FY27.", "value"))),
    ("fan assembly", _r("ORR-FAN-2214-B", ("support/rma_parts.xlsx", "part_number: ORR-FAN-2214-B | description: Fan assembly, dual-rotor, field replaceable | compatible_with: TQ-40", "value"))),
    ("fixed ticket ORR-1847", _r("4.3.2", (CSV, ROW, "value"), (LOG, "incident logged against ORR-1847", "link"))),
    ("error code", _r("E7731", (LOG, "ERROR E7731: thermal throttle engaged on die 0", "value"))),
    ("batch timeout", _r("180 seconds", ("engineering/ingest_service.py", "DEFAULT_BATCH_TIMEOUT_S = 180", "value"))),
    ("THERM_ALERT#", _r("B14", ("specs/backplane_pinout.png", "B14: THERM_ALERT#", "value"))),
    ("board revision", _r("C2", ("support/asset_label.jpg", "BOARD REVISION: REV-C2", "value"))),
    ("production log shows", _r("4.3.2", (CSV, ROW, "value"), (LOG, "incident logged against ORR-1847", "link"),
                                ("engineering/meridian_release_notes.txt", "4.3.1  Fixes a rare hang", "link"))),
    ("unit price", [_r("84.5", ("support/rma_parts.xlsx", "unit_cost_usd: 84.5", "value")), _r("", status="not_found")]),
]


def warrant(question, prompt):
    return "NO" if "unit price" in question.lower() else "YES"


def dump(dest: Path) -> None:
    """Write the JSON files the worker reads with SB_ENGINE=fake (used by the CI contract job)."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "replies.json").write_text(json.dumps(REPLIES), encoding="utf-8")
    (dest / "transcripts.json").write_text(json.dumps(TRANSCRIPTS), encoding="utf-8")


if __name__ == "__main__":
    dump(Path(sys.argv[1]))
````

- [ ] **Step 1: Write the failing test** — `tests/test_sb_index_retrieve.py`:

````python
import time

from sourcebound.engine import FakeEngine
from sourcebound.index import IMAGE_STUB, Index, build_index
from sourcebound.retrieve import bridge, rank_files, search
from tests.fixtures.kit_fake import TRANSCRIPTS


def _build(corpus, tmp_path, transcripts=TRANSCRIPTS):
    eng = FakeEngine(transcripts=transcripts)
    return build_index(corpus, eng, tmp_path / "idx", time.monotonic() + 300, workers=2, log=lambda *a: None), eng


def test_kit_index_skips_hazards_and_marks_supersession(kit_corpus, tmp_path):
    idx, _ = _build(kit_corpus, tmp_path)
    skipped = {e["path"] for e in idx.ledger}
    assert {"vendor/telemetry_capture.dat", "vendor/supplier_agreement_ENCRYPTED.pdf"} <= skipped
    assert "vendor/supplier_agreement_ENCRYPTED.pdf" not in idx.files
    assert idx.files["specs/tq40_datasheet_r1_WITHDRAWN.pdf"].status == "WITHDRAWN"
    assert idx.files["specs/tq40_datasheet_r1_WITHDRAWN.pdf"].superseded_by == "specs/tq40_datasheet_r2.pdf"
    assert "B14: THERM_ALERT#" in idx.file_text("specs/backplane_pinout.png")
    assert idx.vectors is not None and idx.vectors.shape[0] == len(idx.segments)


def test_index_roundtrip(kit_corpus, tmp_path):
    idx, _ = _build(kit_corpus, tmp_path)
    again = Index.load(tmp_path / "idx")
    assert again.stats == idx.stats and len(again.segments) == len(idx.segments)
    assert again.files["support/asset_label.jpg"].extra["image_path"].endswith("asset_label.jpg")
    assert again.fingerprint == idx.fingerprint


def test_untranscribed_image_gets_a_filename_stub(kit_corpus, tmp_path):
    idx, _ = _build(kit_corpus, tmp_path, transcripts={})
    assert idx.file_text("support/asset_label.jpg") == IMAGE_STUB.format(name="asset_label.jpg")


def test_retrieval_finds_each_single_hop_file(kit_corpus, tmp_path, kit_questions):
    idx, eng = _build(kit_corpus, tmp_path)
    for q in kit_questions:
        if not q["expected_citations"] or len(q["expected_citations"]) > 1:
            continue
        top = [r.rel for r in rank_files(idx, search(idx, eng, q["query"]))[:8]]
        assert q["expected_citations"][0] in top, (q["query"], top)


def test_bridge_follows_ticket_from_log_but_not_from_question(kit_corpus, tmp_path):
    idx, eng = _build(kit_corpus, tmp_path)
    q9 = "The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?"
    ranked = rank_files(idx, search(idx, eng, q9))
    links = bridge(idx, q9, ranked)
    assert any(b.ident == "ORR-1847" and idx.segments[b.seg].file == "support/bug_database.csv"
               and b.from_file == "logs/prod_inference_2026-09-02.log" for b in links)
    q4 = "Which firmware version fixed ticket ORR-1847?"
    assert not any(b.ident == "ORR-1847" for b in bridge(idx, q4, rank_files(idx, search(idx, eng, q4))))
````

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest -q tests/test_sb_index_retrieve.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'sourcebound.engine'` (create `tests/fixtures/kit_fake.py` first, as shown)

- [ ] **Step 3: Implement**

`sourcebound/bm25.py`:

````python
"""Okapi BM25 over pre-tokenized documents. Small and dependency-free; rebuilt from segments on load."""
from __future__ import annotations

import math
from collections import Counter, defaultdict


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / self.n) if self.n else 1.0
        self.post: dict[str, dict[int, int]] = defaultdict(dict)
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                self.post[t][i] = c
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def search(self, q: list[str], k: int = 50) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(q):
            p = self.post.get(t)
            if not p:
                continue
            idf = self.idf[t]
            for i, tf in p.items():
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
````

`sourcebound/engine.py`:

````python
"""Engine interface and the deterministic FakeEngine used by every GPU-free test (spec §1, Testing seam 2).

An engine provides:
    transcribe(paths) -> list[str]                    text printed in each image
    embed(texts, kind="document"|"query") -> ndarray  L2-normalized float32 rows
    generate(task, prompt, images=(), max_new_tokens=200, question="") -> str
        task "read" returns the reader JSON; task "warrant" returns YES or NO.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

NOT_FOUND = json.dumps({"status": "not_found", "answer": "", "evidence": []})


class FakeEngine:
    name = "fake"

    def __init__(self, replies=None, transcripts=None, warrant="YES", dim: int = 256):
        # replies: [(question substring, reply)], reply = str | dict | list of those (consumed one per call)
        self.replies = [(k, list(v) if isinstance(v, list) else v) for k, v in (replies or [])]
        self.transcripts = transcripts or {}
        self.warrant = warrant
        self.dim = dim
        self.calls: list[tuple[str, str, int]] = []

    @classmethod
    def from_env(cls) -> "FakeEngine":
        def load(var):
            p = os.environ.get(var)
            return json.loads(Path(p).read_text(encoding="utf-8")) if p else None
        return cls(replies=[tuple(x) for x in (load("SB_FAKE_REPLIES") or [])], transcripts=load("SB_FAKE_TRANSCRIPTS"))

    def transcribe(self, paths):
        return [self.transcripts.get(Path(p).name, "") for p in paths]

    def embed(self, texts, kind="document"):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[r, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)

    def generate(self, task, prompt, images=(), max_new_tokens=200, question=""):
        self.calls.append((task, question, len(images)))
        if task == "warrant":
            return self.warrant(question, prompt) if callable(self.warrant) else self.warrant
        for key, reply in self.replies:
            if key.lower() in question.lower():
                if isinstance(reply, list):
                    reply = reply.pop(0) if len(reply) > 1 else reply[0]
                return reply if isinstance(reply, str) else json.dumps(reply)
        return NOT_FOUND

    def warm_up(self):
        return None
````

`sourcebound/index.py`:

````python
"""The corpus index: segments, file records, BM25, identifier index, dense vectors; build, save, load (spec §4-5)."""
from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .pool import ParserPool
from .records import FileRecord, Segment, assign_status
from .terms import id_key, identifiers, terms
from .walk import walk

IMAGE_STUB = "image file {name} (no transcript available)"


@dataclass
class Index:
    root: str
    files: dict[str, FileRecord]
    segments: list[Segment]
    ledger: list[dict] = field(default_factory=list)
    vectors: np.ndarray | None = None
    fingerprint: str = ""
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_file: dict[str, list[int]] = {}
        for i, s in enumerate(self.segments):
            self.by_file.setdefault(s.file, []).append(i)
        docs = [terms(s.text) + terms(s.file.replace("/", " ")) for s in self.segments]
        self.bm25 = BM25(docs)
        self.ids: dict[str, set[int]] = {}
        for i, s in enumerate(self.segments):
            for tok in identifiers(s.text):
                self.ids.setdefault(id_key(tok), set()).add(i)

    def file_text(self, rel: str) -> str:
        return "\n".join(self.segments[i].text for i in self.by_file.get(rel, []))

    # ------------------------------------------------------------ persistence
    def save(self, d: Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "index.json.tmp"
        tmp.write_text(json.dumps({
            "root": self.root, "fingerprint": self.fingerprint, "stats": self.stats, "ledger": self.ledger,
            "files": {k: v.to_dict() for k, v in self.files.items()},
            "segments": [asdict(s) for s in self.segments]}, ensure_ascii=False), encoding="utf-8")
        if self.vectors is not None:
            np.save(d / "vectors.npy", self.vectors)
        elif (d / "vectors.npy").exists():
            (d / "vectors.npy").unlink()
        os.replace(tmp, d / "index.json")

    @classmethod
    def load(cls, d: Path) -> "Index | None":
        d = Path(d)
        try:
            data = json.loads((d / "index.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        vec = np.load(d / "vectors.npy") if (d / "vectors.npy").exists() else None
        return cls(root=data["root"], files={k: FileRecord(**v) for k, v in data["files"].items()},
                   segments=[Segment(**s) for s in data["segments"]], ledger=data["ledger"], vectors=vec,
                   fingerprint=data["fingerprint"], stats=data.get("stats", {}))


def fingerprint(root: Path) -> str:
    h = hashlib.sha1()
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        dirnames.sort()
        for n in sorted(filenames):
            try:
                st = os.stat(os.path.join(dirpath, n), follow_symlinks=False)
                h.update(f"{os.path.relpath(os.path.join(dirpath, n), root)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
            except OSError:
                h.update(f"{n}|unreadable\n".encode())
    return h.hexdigest()


def _sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_index(root: Path, engine, index_dir: Path, deadline: float, workers: int = 4,
                parse_timeout_s: float = 20.0, opener=open, log=print) -> Index:
    """Walk, parse, transcribe, embed, persist. `deadline` is a time.monotonic() value; optional work stops early."""
    t0 = time.monotonic()
    root, index_dir = Path(root), Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    recs, ledger = walk(root, opener=opener)
    records = {r.rel: r for r in recs}
    log(f"walk: {len(records)} files, {len(ledger)} skipped")

    # 1. parse text-bearing files in killable children
    jobs = [(r.rel, (r.path, r.rel, r.ftype)) for r in recs]
    results = ParserPool(workers=workers, timeout_s=parse_timeout_s).run(jobs, deadline=deadline - 90) if jobs else {}
    segments: list[Segment] = []
    heads: dict[str, str] = {}
    ocr_jobs: list[tuple[str, str, str]] = []      # (rel, kind, locator)
    for rel, (status, res) in sorted(results.items()):
        if status != "ok":
            ledger.append({"path": rel, "reason": str(res)})
            records.pop(rel, None)
            continue
        heads[rel] = res.get("head", "")
        segments.extend(Segment(**s) for s in res["segments"] if s["text"])
        if res.get("image"):
            ocr_jobs.append((rel, "image", "image"))
        ocr_jobs += [(rel, "page", str(p)) for p in res.get("ocr_pages", [])]
        ocr_jobs += [(rel, "media", m) for m in res.get("media", [])]
    log(f"parse: {len(segments)} segments in {time.monotonic() - t0:.1f}s")

    # 2. transcribe images, then text-less PDF pages, then embedded document images, while time remains
    order = {"image": 0, "page": 1, "media": 2}
    ocr_jobs.sort(key=lambda j: (order[j[1]], j[0], j[2]))
    cache_path = index_dir / "transcripts.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    render_dir = index_dir / "renders"
    render_dir.mkdir(exist_ok=True)
    pending: list[tuple[str, str, str, str, str]] = []   # (rel, kind, locator, image_path, cache_key)
    for rel, kind, loc in ocr_jobs:
        try:
            img_path = _materialize(records[rel], kind, loc, render_dir)
            key = f"{getattr(engine, 'name', '?')}:{_sha1(img_path)}"
            pending.append((rel, kind, loc, img_path, key))
        except Exception as e:  # noqa: BLE001
            ledger.append({"path": f"{rel}#{loc}", "reason": f"render failed: {type(e).__name__}: {e}"})
    transcribed = 0
    batch = int(os.environ.get("SB_OCR_BATCH", "4"))
    for i in range(0, len(pending), batch):
        chunk = pending[i:i + batch]
        todo = [p for p in chunk if p[4] not in cache]
        if todo and time.monotonic() < deadline - 45:
            texts = engine.transcribe([p[3] for p in todo])
            for p, text in zip(todo, texts):
                cache[p[4]] = text
        for rel, kind, loc, img_path, key in chunk:
            text = cache.get(key, "")
            if text.strip():
                transcribed += 1
                segments.append(Segment(rel, "image" if kind == "image" else f"{kind} {loc}", "transcript", text))
            elif kind == "image":
                segments.append(Segment(rel, "image", "transcript", IMAGE_STUB.format(name=Path(rel).name)))
    for rel, kind, loc, img_path, key in pending:
        if kind == "image":
            records[rel].extra["image_path"] = records[rel].path
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    log(f"transcribe: {transcribed}/{len(pending)} in {time.monotonic() - t0:.1f}s")

    # 3. supersession, dense vectors, persist
    assign_status(records, heads)
    segments = [s for s in segments if s.file in records]
    vectors = None
    if segments and time.monotonic() < deadline - 10:
        vectors = engine.embed([f"{s.file}\n{s.text}"[:2000] for s in segments], kind="document")
    idx = Index(root=str(root), files=records, segments=segments, ledger=ledger, vectors=vectors,
                fingerprint=fingerprint(root),
                stats={"files": len(records), "skipped": len(ledger), "segments": len(segments),
                       "transcribed": transcribed, "seconds": round(time.monotonic() - t0, 2)})
    idx.save(index_dir)
    log(f"index: {idx.stats}")
    return idx


def _materialize(rec: FileRecord, kind: str, loc: str, render_dir: Path) -> str:
    """Path of an image file to transcribe: the file itself, a rendered PDF page, or an extracted DOCX image."""
    if kind == "image":
        return rec.path
    safe = hashlib.sha1(f"{rec.rel}#{loc}".encode()).hexdigest()[:16]
    if kind == "page":
        import fitz
        out = render_dir / f"{safe}.png"
        with fitz.open(rec.path) as doc:
            doc[int(loc)].get_pixmap(dpi=150).save(str(out))
        return str(out)
    out = render_dir / f"{safe}{Path(loc).suffix.lower()}"
    with zipfile.ZipFile(rec.path) as z:
        out.write_bytes(z.read(loc))
    return str(out)
````

`sourcebound/retrieve.py`:

````python
"""Hybrid retrieval, file ranking and the identifier bridge hop (spec §5)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .normalize import official
from .terms import id_key, identifiers, terms

RRF_K = 60


@dataclass
class Ranked:
    rel: str
    score: float
    segs: list[int] = field(default_factory=list)


@dataclass
class Bridge:
    seg: int
    ident: str
    from_file: str


def rrf(*lists: list[tuple[int, float]], k: int = RRF_K) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for lst in lists:
        for rank, (i, _) in enumerate(lst):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


def search(index, engine, question: str, k: int = 50) -> list[tuple[int, float]]:
    lex = index.bm25.search(terms(question, query=True), k)
    dense: list[tuple[int, float]] = []
    if os.environ.get("SB_DENSE", "1") == "1" and index.vectors is not None and len(index.vectors):
        q = np.asarray(engine.embed([question], kind="query")[0], dtype=np.float32)
        sims = index.vectors @ q
        top = np.argsort(-sims)[:k]
        dense = [(int(i), float(sims[i])) for i in top]
    return rrf(lex, dense)


def rank_files(index, fused: list[tuple[int, float]], demote: float = 0.5) -> list[Ranked]:
    per: dict[str, list[tuple[int, float]]] = {}
    for i, s in fused:
        per.setdefault(index.segments[i].file, []).append((i, s))
    out = []
    for rel, hits in per.items():
        hits.sort(key=lambda x: -x[1])
        score = hits[0][1] + (0.1 * hits[1][1] if len(hits) > 1 else 0.0)
        rec = index.files.get(rel)
        if rec and rec.status != "CURRENT" and rec.superseded_by in index.files:
            score *= demote
        out.append(Ranked(rel, score, [i for i, _ in hits]))
    return sorted(out, key=lambda r: (-r.score, r.rel))


def bridge(index, question: str, ranked: list[Ranked], top_files: int = 6, segs_per_file: int = 3,
           max_files_per_id: int = 3, max_links: int = 6, extra_ids: list[str] | None = None) -> list[Bridge]:
    """Follow identifiers that appear in the top evidence but not in the question into other files."""
    qkey = official(question)
    cands: list[tuple[str, str]] = [(t, "") for t in (extra_ids or [])]
    for r in ranked[:top_files]:
        for i in r.segs[:segs_per_file]:
            cands += [(t, r.rel) for t in identifiers(index.segments[i].text)]
    out: list[Bridge] = []
    done: set[str] = set()
    for tok, src in cands:
        key = id_key(tok)
        if key in done or (src and key in qkey):
            continue
        done.add(key)
        hits = sorted(index.ids.get(key, ()))
        files = {index.segments[i].file for i in hits}
        if not hits or len(files) > max_files_per_id:
            continue
        for i in hits:
            f = index.segments[i].file
            if f == src:
                continue
            if any(b.seg == i for b in out):
                continue
            out.append(Bridge(i, tok, src))
            if len(out) >= max_links:
                return out
    return out
````

- [ ] **Step 4: Run it and watch it pass**

Run: `python -m pytest -q tests/test_sb_index_retrieve.py`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_sb_index_retrieve.py sourcebound/bm25.py sourcebound/engine.py sourcebound/index.py sourcebound/retrieve.py tests/fixtures/kit_fake.py
git commit -m "feat(mc3): index with transcripts, BM25+dense RRF retrieval and identifier bridge" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 8: Reader, gates and the per-question pipeline

Spec: §6–§7, user stories 29–41

**Files:**
- Create: `sourcebound/reader.py`
- Create: `sourcebound/gates.py`
- Create: `sourcebound/pipeline.py`
- Test: `tests/test_sb_gates_pipeline.py`

These are the scoring rules. Read `judge()` closely. The citation set is computed here from verified quotes; it is never copied from the model.

- [ ] **Step 1: Write the failing test** — `tests/test_sb_gates_pipeline.py`:

````python
import time

import pytest

from sourcebound import pipeline
from sourcebound.engine import FakeEngine
from sourcebound.gates import grounded, judge
from sourcebound.index import build_index
from sourcebound.reader import Pack, parse_reply
from tests.fixtures import kit_fake as kf


@pytest.fixture
def kit_index(kit_corpus, tmp_path):
    return build_index(kit_corpus, FakeEngine(transcripts=kf.TRANSCRIPTS), tmp_path / "idx", time.monotonic() + 300,
                       workers=2, log=lambda *a: None)


def reply(answer, *ev, status="answered"):
    return parse_reply(__import__("json").dumps(kf._r(answer, *ev, status=status)))


PACK = Pack(text="")


def test_grounding_is_strict_but_tolerates_spacing():
    doc = "Maximum junction temperature .......... 94 C\nBoard power (TBP) ..................... 410 W"
    assert grounded("maximum junction temperature .......... 94 C", doc)
    assert grounded("Maximum  junction temperature .......... 94 C", doc)
    assert not grounded("Maximum junction temperature .......... 105 C", doc)
    assert not grounded("94 C", "Board power 410 W")


def test_fabricated_quote_is_refused(kit_index):
    v = judge(kit_index, PACK, "q", reply("99", (kf.R2, "Maximum junction temperature 99 C", "value")))
    assert not v.answer and v.retry


def test_withdrawn_value_is_rejected_with_exclusion(kit_index):
    v = judge(kit_index, PACK, "q", reply("105", (kf.R1, "Maximum junction temperature .......... 105 C", "value")))
    assert not v.answer and v.exclude == [kf.R1] and kf.R2 in v.retry


def test_answer_must_appear_in_its_evidence(kit_index):
    v = judge(kit_index, PACK, "q", reply("4.3.1", (kf.CSV, kf.ROW, "value")))
    assert not v.answer and "does not appear" in v.retry


def test_link_dropped_when_question_supplies_the_identifier(kit_index):
    r = reply("4.3.2", (kf.CSV, kf.ROW, "value"), (kf.LOG, "incident logged against ORR-1847", "link"))
    assert judge(kit_index, PACK, "Which firmware version fixed ticket ORR-1847?", r).citations == [kf.CSV]
    v = judge(kit_index, PACK, "The production log shows an incident. Which release fixed it?", r)
    assert v.answer == "4.3.2" and v.citations == [kf.CSV, kf.LOG]


def test_topical_neighbour_is_never_cited(kit_index):
    r = reply("4.3.2", (kf.CSV, kf.ROW, "value"), ("engineering/meridian_release_notes.txt", "4.3.1  Fixes a rare hang", "link"))
    assert judge(kit_index, PACK, "Which release fixed the incident?", r).citations == [kf.CSV]


def test_unparseable_and_not_found():
    assert parse_reply("no json here") is None
    assert parse_reply('```json\n{"status": "answered", "answer": "B14", "evidence": []}\n```')["answer"] == "B14"
    assert parse_reply('<think>x</think>{"answer": "", "evidence": []}')["status"] == "not_found"


def test_pipeline_retries_past_withdrawn_and_completes_qualifier(kit_index):
    eng = FakeEngine(replies=kf.REPLIES, warrant=kf.warrant)
    out = pipeline.answer(kit_index, eng, "What is the maximum junction temperature of the TQ-40?", 20)
    assert (out["answer"], out["citations"]) == ("94", [kf.R2])
    out = pipeline.answer(kit_index, eng, "What board revision is printed on the asset label?", 20)
    assert (out["answer"], out["citations"]) == ("REV-C2", ["support/asset_label.jpg"])
    assert any(n == 1 for task, q, n in eng.calls if "asset label" in q and task == "read")   # image attached


def test_warrant_no_leads_to_refusal(kit_index):
    eng = FakeEngine(replies=kf.REPLIES, warrant=kf.warrant)
    out = pipeline.answer(kit_index, eng, "What is the unit price of the TQ-40 at 10,000 unit volume?", 20)
    assert (out["answer"], out["citations"]) == ("", [])


def test_need_lookup_runs_one_more_hop(kit_index):
    first = {"status": "need_lookup", "answer": "", "evidence": [], "lookup": ["ORR-1847"]}
    second = kf._r("4.3.2", (kf.CSV, kf.ROW, "value"), (kf.LOG, "incident logged against ORR-1847", "link"))
    eng = FakeEngine(replies=[("which defect fix", [first, second])])
    out = pipeline.answer(kit_index, eng, "For the logged incident, which defect fix release applies?", 20)
    assert out["answer"] == "4.3.2" and out["citations"] == [kf.CSV, kf.LOG]


def test_deadline_short_circuits_to_refusal(kit_index):
    out = pipeline.answer(kit_index, FakeEngine(replies=kf.REPLIES), "What error code is logged?", 0.5)
    assert out["answer"] == "" and out["citations"] == []
````

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest -q tests/test_sb_gates_pipeline.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'sourcebound.pipeline'`

- [ ] **Step 3: Implement**

`sourcebound/reader.py`:

````python
"""Evidence pack, reader prompt and reply parsing (spec §6)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .retrieve import Bridge, Ranked

WHOLE_FILE_CHARS = 6000
PACK_CHARS = 36000


@dataclass
class Pack:
    text: str
    images: list[str] = field(default_factory=list)
    fids: dict[str, str] = field(default_factory=dict)          # "F1" -> rel
    bridged_only: set[int] = field(default_factory=set)         # segments that entered only through a bridge
    bridges: list[Bridge] = field(default_factory=list)


def build_pack(index, ranked: list[Ranked], bridges: list[Bridge], exclude: set[str] = frozenset(),
               max_files: int = 8, max_images: int = 3, image_rank: int = 4, budget: int = PACK_CHARS) -> Pack:
    order: list[str] = []
    direct: set[int] = set()
    for r in ranked:
        if r.rel in exclude:
            continue
        if len(order) < max_files:
            order.append(r.rel)
        direct.update(r.segs)
    linked: dict[str, list[Bridge]] = {}
    for b in bridges:
        f = index.segments[b.seg].file
        if f in exclude:
            continue
        linked.setdefault(f, []).append(b)
        if f not in order:
            order.append(f)
    pack = Pack(text="", bridges=[b for bs in linked.values() for b in bs])
    blocks, used = [], 0
    for n, rel in enumerate(order, start=1):
        fid = f"F{n}"
        rec = index.files[rel]
        head = f"[{fid}] path={rel} type={rec.ftype} status={rec.status}"
        if rec.superseded_by:
            head += f" superseded_by={rec.superseded_by}"
        if rec.ftype == "image" and n <= image_rank and len(pack.images) < max_images:
            pack.images.append(rec.extra.get("image_path", rec.path))
            head += f" (image #{len(pack.images)} attached; transcript below)"
        for b in linked.get(rel, []):
            src = next((k for k, v in pack.fids.items() if v == b.from_file), b.from_file or "the question")
            head += f"\n(linked by identifier {b.ident} found in {src})"
        whole = index.file_text(rel)
        if len(whole) <= WHOLE_FILE_CHARS:
            body = whole
        else:
            picks = next((r.segs for r in ranked if r.rel == rel), [])[:4] + [b.seg for b in linked.get(rel, [])]
            keep = sorted({j for i in picks for j in (i - 1, i, i + 1)
                           if 0 <= j < len(index.segments) and index.segments[j].file == rel})
            body = "\n...\n".join(index.segments[j].text for j in keep)
        block = f"{head}\n{body.strip()}\n"
        if used + len(block) > budget and blocks:
            break
        pack.fids[fid] = rel
        blocks.append(block)
        used += len(block)
    for b in pack.bridges:
        if b.seg not in direct:
            pack.bridged_only.add(b.seg)
    pack.text = "\n".join(blocks)
    return pack


INSTRUCTIONS = """You answer a question using ONLY the documents below. They describe fictional products, so anything you know from elsewhere is wrong.

Rules:
- The answer is the value only: a number, code, part number, version, quarter or name. No sentence and no unit.
- Copy the value exactly as printed, including qualifiers that identify it (for example the fiscal year after a quarter, or the prefix of a revision).
- The entity and the attribute must match the question exactly (product, model, component, column). A value for a different product or a different attribute is not an answer.
- A document whose status is WITHDRAWN or SUPERSEDED is not authoritative; use the current document.
- In source code the assigned value is the answer; comments can mention old values.
- When an identifier (ticket, error code, part number) found in one document leads to the value in another document, quote both: the value quote with role "value" and the identifier quote with role "link".
- Quotes must be copied character for character from the documents.
- If the documents do not state the answer, reply with status "not_found".
- If a document names an identifier whose details are missing from these documents, reply with status "need_lookup" and list it in "lookup".

Reply with one JSON object and nothing else:
{"status": "answered", "answer": "...", "answer_type": "extracted", "evidence": [{"file": "F1", "quote": "...", "role": "value"}], "lookup": []}
status is one of answered, not_found, need_lookup; answer_type is extracted or derived; role is value or link."""


def reader_prompt(pack: Pack, question: str, feedback: str = "") -> str:
    fb = f"\nA previous attempt was rejected: {feedback}\n" if feedback else ""
    return f"{INSTRUCTIONS}\n\nDOCUMENTS\n{pack.text}\n{fb}\nQUESTION: {question}\nJSON:"


WARRANT = """Question: {question}
Proposed answer: {answer}
Evidence:
{evidence}

Does the evidence state exactly the attribute the question asks for, for exactly the entity the question asks about, with the proposed answer as its value? Reply YES or NO."""


def warrant_prompt(question: str, answer: str, quotes: list[str]) -> str:
    return WARRANT.format(question=question, answer=answer, evidence="\n".join(f"- {q}" for q in quotes))


def _first_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def parse_reply(text: str) -> dict | None:
    """First balanced JSON object in the reply, coerced to the reader schema; None if absent or invalid."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"<\|[^|]*\|>", "", text)
    obj = _first_object(text)
    if obj is None:
        return None
    try:
        d = json.loads(obj)
    except ValueError:
        return None
    if not isinstance(d, dict):
        return None
    status = str(d.get("status", "")).strip().lower()
    if status not in ("answered", "not_found", "need_lookup"):
        status = "answered" if str(d.get("answer", "")).strip() else "not_found"
    ev = []
    for e in d.get("evidence") or []:
        if isinstance(e, dict) and str(e.get("quote", "")).strip():
            role = str(e.get("role", "value")).lower()
            ev.append({"file": str(e.get("file", "")).strip(), "quote": str(e["quote"]),
                       "role": role if role in ("value", "link") else "value"})
    lookup = [str(x) for x in (d.get("lookup") or []) if str(x).strip()][:4]
    atype = str(d.get("answer_type", "extracted")).lower()
    return {"status": status, "answer": str(d.get("answer", "") or "").strip(), "evidence": ev, "lookup": lookup,
            "answer_type": atype if atype in ("extracted", "derived") else "extracted"}
````

`sourcebound/gates.py`:

````python
"""Deterministic gates: grounding, answer-in-evidence, supersession, necessity (spec §7).

The model proposes an answer and quotes; these rules decide the answer and the citation set.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .normalize import align_number, complete_qualifier, contains_answer, official, plain, shape
from .terms import id_key, identifiers

ALLOW_DERIVED = os.environ.get("SB_ALLOW_DERIVED", "0") == "1"


@dataclass
class Evidence:
    rel: str
    quote: str
    role: str


@dataclass
class Verdict:
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    reason: str = ""
    retry: str = ""                        # feedback for one more reader attempt
    exclude: list[str] = field(default_factory=list)
    values: list[Evidence] = field(default_factory=list)
    links: list[Evidence] = field(default_factory=list)


def grounded(quote: str, text: str, min_ratio: float = 0.9) -> bool:
    q, t = plain(quote), plain(text)
    if not q or not t:
        return False
    if q in t:
        return True
    parts = [p.strip() for p in re.split(r"\.\.\.|\|", q) if len(p.strip()) >= 4]
    if len(parts) > 1:
        return all(grounded(p, text, min_ratio) for p in parts)
    if len(q) < 12:
        return False
    m = SequenceMatcher(None, t, q, autojunk=False).find_longest_match(0, len(t), 0, len(q))
    start = max(0, m.a - m.b)
    window = t[start:start + len(q) + 8]
    return SequenceMatcher(None, window, q, autojunk=False).ratio() >= min_ratio


def segment_context(index, ev: Evidence) -> str:
    """Text of the segments of ev.rel that contain ev.quote (the row, page or line window it came from)."""
    q = plain(ev.quote)
    return "\n".join(index.segments[i].text for i in index.by_file.get(ev.rel, []) if q in plain(index.segments[i].text))


def resolve(ref: str, pack, index) -> str | None:
    ref = (ref or "").strip().strip("[]")
    if ref in pack.fids:
        return pack.fids[ref]
    cand = ref.replace("\\", "/")
    root = index.root.replace("\\", "/").rstrip("/") + "/"
    for prefix in (root, "/app/corpus/", "./"):
        if cand.startswith(prefix):
            cand = cand[len(prefix):]
    return cand if cand in index.files else None


def judge(index, pack, question: str, reply: dict | None) -> Verdict:
    if reply is None:
        return Verdict(reason="unparseable reply", retry="Reply with one valid JSON object only.")
    if reply["status"] != "answered" or not reply["answer"]:
        return Verdict(reason=reply["status"] or "not_found")
    ev = []
    for e in reply["evidence"]:
        rel = resolve(e["file"], pack, index)
        if rel and grounded(e["quote"], index.file_text(rel)):
            ev.append(Evidence(rel, e["quote"], e["role"]))
    raw = reply["answer"]
    values = [e for e in ev if e.role == "value"] or [e for e in ev if contains_answer(shape(raw), e.quote)]
    links = [e for e in ev if e.role == "link" and e not in values]
    if not values:
        return Verdict(reason="no grounded value quote",
                       retry="Quote the exact text that states the answer, copied character for character, with its file id.")
    answer = shape(raw)
    answer = align_number(complete_qualifier(answer, values[0].quote), values[0].quote)
    derived = reply["answer_type"] == "derived" and ALLOW_DERIVED
    if not derived:
        # The answer must be in the quote, or in the segment (row, page, window) that contains the quote.
        # Never the whole file: a near-miss row elsewhere in the same table would then pass.
        hits = [v for v in values if contains_answer(answer, v.quote)] or \
               [v for v in values if contains_answer(answer, segment_context(index, v))]
        if not hits:
            return Verdict(reason="answer not found in its evidence",
                           retry=f"The answer {answer!r} does not appear in the quoted text. Answer with the value exactly as printed.")
        values = hits + [v for v in values if v not in hits]
    primary = values[0]
    rec = index.files[primary.rel]
    if rec.status != "CURRENT" and rec.superseded_by in index.files:
        return Verdict(reason=f"value came from a {rec.status} document", exclude=[primary.rel],
                       retry=f"{primary.rel} is {rec.status}; use {rec.superseded_by} instead.")
    cites = [primary.rel]
    if derived:
        cites += [v.rel for v in values[1:] if v.rel not in cites][:1]
    qkey = official(question)
    value_ctx = primary.quote + "\n" + segment_context(index, primary)
    ctx_keys = {id_key(t) for t in identifiers(value_ctx)}
    for l in links:
        if l.rel in cites:
            continue
        shared = [t for t in identifiers(l.quote) if id_key(t) in ctx_keys and id_key(t) not in qkey]
        if shared:
            cites.append(l.rel)
    # a value segment that was reachable only through an identifier bridge implies its source file is a link
    for b in getattr(pack, "bridges", []):
        if (b.seg in pack.bridged_only and index.segments[b.seg].file == primary.rel and b.from_file
                and b.from_file not in cites and id_key(b.ident) in ctx_keys and id_key(b.ident) not in qkey
                and plain(primary.quote) in plain(index.segments[b.seg].text)):
            cites.append(b.from_file)
    return Verdict(answer=answer, citations=cites, reason="ok", values=values, links=links)
````

`sourcebound/pipeline.py`:

````python
"""One question, end to end, under a deadline: retrieve -> bridge -> read -> gates -> warrant (spec §5-7)."""
from __future__ import annotations

import os
import time

from .gates import judge
from .reader import build_pack, parse_reply, reader_prompt, warrant_prompt
from .retrieve import bridge, rank_files, search

MIN_CALL_S = float(os.environ.get("SB_MIN_CALL_S", "6"))
WARRANT = os.environ.get("SB_WARRANT", "1") == "1"
MAX_NEW = int(os.environ.get("SB_MAX_NEW_TOKENS", "200"))
BRIDGE = os.environ.get("SB_BRIDGE", "1") == "1"


def refusal(diag: dict, reason: str) -> dict:
    diag["reason"] = reason
    return {"answer": "", "citations": [], "confidence": 0.0, "diag": diag}


def answer(index, engine, question: str, deadline_s: float = 24.0) -> dict:
    t0 = time.monotonic()
    end = t0 + deadline_s
    left = lambda: end - time.monotonic()  # noqa: E731
    diag: dict = {"attempts": []}
    fused = search(index, engine, question)
    ranked = rank_files(index, fused)
    bridges = bridge(index, question, ranked) if BRIDGE else []
    diag["ranked"] = [r.rel for r in ranked[:8]]
    diag["bridges"] = [(b.ident, index.segments[b.seg].file, b.from_file) for b in bridges]
    exclude: set[str] = set()
    feedback, looked_up = "", False
    for attempt in range(3):
        if left() < MIN_CALL_S:
            diag["attempts"].append("skipped: deadline")
            break
        pack = build_pack(index, ranked, bridges, exclude)
        raw = engine.generate("read", reader_prompt(pack, question, feedback), pack.images, MAX_NEW, question=question)
        reply = parse_reply(raw)
        diag["attempts"].append({"fids": pack.fids, "raw": raw[:600]})
        if reply and reply["status"] == "need_lookup" and reply["lookup"] and not looked_up:
            looked_up = True
            bridges = bridges + bridge(index, question, [], extra_ids=reply["lookup"])
            continue
        v = judge(index, pack, question, reply)
        diag["attempts"][-1]["verdict"] = v.reason
        if v.answer:
            if WARRANT and left() > 2.0:
                quotes = [e.quote for e in v.values[:2] + v.links[:2]]
                ok = engine.generate("warrant", warrant_prompt(question, v.answer, quotes), (), 3, question=question)
                diag["warrant"] = ok.strip()[:20]
                if ok.strip().upper().startswith("NO"):
                    feedback = (f"The answer {v.answer!r} does not state the requested attribute of the requested "
                                "entity. Look again; if no document states it, reply not_found.")
                    continue
            conf = 0.9 if len(v.citations) == 1 else 0.8
            diag["seconds"] = round(time.monotonic() - t0, 3)
            return {"answer": v.answer, "citations": v.citations, "confidence": conf, "diag": diag}
        if v.reason in ("not_found", "need_lookup"):
            return refusal(diag, v.reason)
        exclude |= set(v.exclude)
        feedback = v.retry
        if not feedback:
            break
    diag["seconds"] = round(time.monotonic() - t0, 3)
    return refusal(diag, "gates failed")
````

- [ ] **Step 4: Run it and watch it pass**

Run: `python -m pytest -q tests/test_sb_gates_pipeline.py`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_sb_gates_pipeline.py sourcebound/reader.py sourcebound/gates.py sourcebound/pipeline.py
git commit -m "feat(mc3): evidence pack, quote-grounding, supersession and necessity gates, warrant check" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 9: Protocol, worker, supervisor and the /app/app.py client (the contract seam)

Spec: §2, user stories 1–5, 42–45

**Files:**
- Create: `sourcebound/protocol.py`
- Create: `sourcebound/worker.py`
- Create: `sourcebound/supervisor.py`
- Create: `mc3/app.py`
- Create: `eval_mc3/score.py`
- Test: `tests/test_sb_contract.py`

This is the highest seam: real `app.py` processes, a real worker thread, the grader's scoring. `SB_NO_SPAWN=1` stops tests from leaving a background supervisor behind.

- [ ] **Step 1: Write the failing test** — `tests/test_sb_contract.py`:

````python
"""The highest seam: the harness contract. `mc3/app.py` runs as separate processes against a real worker
(FakeEngine), exactly as the grader execs it, and the JSON files are scored with the grader's rules."""
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from eval_mc3.score import score_one
from sourcebound import protocol
from sourcebound.engine import FakeEngine
from sourcebound.worker import Worker, serve
from tests.fixtures import kit_fake as kf

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "mc3" / "app.py"


def free_addr(tmp_path) -> str:
    if hasattr(socket, "AF_UNIX"):
        return f"unix:{tmp_path / 'sb.sock'}"
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"tcp:127.0.0.1:{port}"


@pytest.fixture
def running_worker(tmp_path):
    addr = free_addr(tmp_path)
    eng = FakeEngine(replies=kf.REPLIES, transcripts=kf.TRANSCRIPTS, warrant=kf.warrant)
    w = Worker(eng, tmp_path / "index", parse_workers=2)
    stop = threading.Event()
    t = threading.Thread(target=serve, args=(w, addr, None, stop), daemon=True)
    t.start()
    for _ in range(50):
        try:
            protocol.request({"op": "status"}, timeout=1, addr=addr)
            break
        except OSError:
            time.sleep(0.1)
    yield addr, w, eng
    stop.set()
    t.join(5)


def app(args, addr, out_dir, timeout=120):
    env = {**os.environ, "SB_ADDR": addr, "SB_OUTPUT_DIR": str(out_dir), "PYTHONPATH": str(ROOT),
           "SB_READY_WAIT_S": "5", "SB_NO_SPAWN": "1"}
    t = time.monotonic()
    r = subprocess.run([sys.executable, str(APP), *args], env=env, capture_output=True, text=True, timeout=timeout)
    return r, time.monotonic() - t


def test_sample_kit_scores_200_through_the_contract(running_worker, kit_corpus, kit_questions, tmp_path):
    addr, w, _ = running_worker
    out = tmp_path / "out"
    r, _ = app(["--index", str(kit_corpus)], addr, out)
    assert r.returncode == 0, r.stderr
    assert "telemetry_capture.dat" in r.stderr                      # skip ledger is reported
    results = []
    for i, q in enumerate(kit_questions, start=1):
        qid = f"query_{i:02d}"
        r, dt = app(["--corpus", str(kit_corpus), "--query-id", qid, "--query", q["query"]], addr, out)
        assert r.returncode == 0, r.stderr
        pred = json.loads((out / f"{qid}_output.json").read_text(encoding="utf-8"))
        assert set(pred) == {"answer", "citations", "confidence"}
        results.append((q["n"], pred, score_one(pred, q, str(kit_corpus))))
    failed = [(n, p, s["kind"]) for n, p, s in results if not s["strict"]]
    assert failed == []


def test_unreadable_file_does_not_cost_other_files(running_worker, kit_corpus, tmp_path):
    addr, w, _ = running_worker
    real_open = open

    def deny(path, mode="r", *a, **k):
        if os.path.basename(path) == "asset_label.jpg":
            raise PermissionError(13, "denied", path)
        return real_open(path, mode, *a, **k)
    w.opener = deny
    r, _ = app(["--index", str(kit_corpus)], addr, tmp_path / "out")
    assert r.returncode == 0
    assert "support/asset_label.jpg" not in w.index.files and "support/rma_parts.xlsx" in w.index.files


def test_worker_absent_writes_valid_refusal_fast(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "query_07_output.json").write_text('{"answer": "STALE", "citations": ["x"]}')
    env_addr = free_addr(tmp_path)
    r, dt = app(["--corpus", str(tmp_path), "--query-id", "query_07", "--query", "anything?"], env_addr, out)
    assert r.returncode == 0 and dt < 15
    assert json.loads((out / "query_07_output.json").read_text()) == {"answer": "", "citations": [], "confidence": 0.0}


def test_query_id_is_used_verbatim(running_worker, kit_corpus, tmp_path):
    addr, _, _ = running_worker
    app(["--index", str(kit_corpus)], addr, tmp_path / "out")
    app(["--corpus", str(kit_corpus), "--query-id", "Q-weird_9", "--query", "What error code is logged when the thermal throttle engages?"],
        addr, tmp_path / "out")
    assert json.loads((tmp_path / "out" / "Q-weird_9_output.json").read_text())["answer"] == "E7731"


def test_bad_request_does_not_kill_worker(running_worker):
    addr, _, _ = running_worker
    with protocol.connect(addr) as s:
        s.sendall(b"not json\n")
        assert protocol.recv_line(s)["ok"] is False
    assert protocol.request({"op": "status"}, timeout=2, addr=addr)["ok"] is True


def test_client_imports_no_heavy_modules():
    code = ("import sys, runpy; sys.argv=['app.py','--help']\n"
            "try:\n    runpy.run_path(r'%s', run_name='not_main')\nexcept SystemExit:\n    pass\n"
            "print(sorted(m for m in ('torch','transformers','PIL','numpy','fitz','openpyxl') if m in sys.modules))") % APP
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert out.stdout.strip() == "[]", out.stderr


def test_only_the_first_index_is_charged_to_the_startup_budget(tmp_path, monkeypatch):
    import sourcebound.worker as wk
    seen = []

    def fake_build(corpus, eng, d, deadline, **k):
        seen.append(deadline - time.monotonic())
        return type("I", (), {"stats": {}, "ledger": []})()
    monkeypatch.setattr(wk, "build_index", fake_build)
    w = Worker(FakeEngine(), tmp_path / "idx", started=time.time() - 1000)     # the container started long ago
    w.handle({"op": "index", "corpus": str(tmp_path)})
    w.handle({"op": "index", "corpus": str(tmp_path)})
    assert seen[0] == pytest.approx(wk.MIN_INDEX_S, abs=2)                    # first: what is left of startup
    assert seen[1] == pytest.approx(wk.INDEX_BUDGET_S, abs=2)                 # later (eval suites): a full budget
````

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest -q tests/test_sb_contract.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval_mc3.score'` / `'sourcebound.protocol'`

- [ ] **Step 3: Implement**

`sourcebound/protocol.py`:

````python
"""Newline-delimited JSON between the thin client and the worker. Standard library only.

Address: SB_ADDR = "unix:/path" or "tcp:host:port". Default: a unix socket on POSIX, localhost TCP on Windows
(Windows CPython has no AF_UNIX).
"""
from __future__ import annotations

import json
import os
import socket

MAX_LINE = 8 * 2**20
DEFAULT_ADDR = "unix:/tmp/sourcebound.sock" if hasattr(socket, "AF_UNIX") else "tcp:127.0.0.1:47823"


def address() -> str:
    return os.environ.get("SB_ADDR", DEFAULT_ADDR)


def _parse(addr: str):
    kind, _, rest = addr.partition(":")
    if kind == "unix":
        return socket.AF_UNIX, rest
    host, _, port = rest.rpartition(":")
    return socket.AF_INET, (host, int(port))


def connect(addr: str | None = None, timeout: float = 2.0) -> socket.socket:
    fam, target = _parse(addr or address())
    s = socket.socket(fam, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(target)
    except BaseException:
        s.close()
        raise
    return s


def listen(addr: str | None = None) -> socket.socket:
    fam, target = _parse(addr or address())
    s = socket.socket(fam, socket.SOCK_STREAM)
    if fam == getattr(socket, "AF_UNIX", None):
        try:
            os.unlink(target)
        except FileNotFoundError:
            pass
    else:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(target)
    s.listen(16)
    return s


def send_line(sock: socket.socket, obj: dict) -> None:
    sock.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def recv_line(sock: socket.socket) -> dict:
    buf = bytearray()
    while not buf.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("connection closed before a full message")
        buf += chunk
        if len(buf) > MAX_LINE:
            raise ValueError("message too large")
    return json.loads(buf.decode("utf-8"))


def request(obj: dict, timeout: float, addr: str | None = None) -> dict:
    with connect(addr, timeout=min(timeout, 2.0)) as s:
        s.settimeout(timeout)
        send_line(s, obj)
        return recv_line(s)
````

`sourcebound/worker.py`:

````python
"""Resident worker: owns the engine and the index, serves index / query / status requests (spec §2)."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import traceback
from pathlib import Path

from . import pipeline, protocol
from .index import Index, build_index

INDEX_BUDGET_S = float(os.environ.get("SB_INDEX_BUDGET_S", "540"))
MIN_INDEX_S = 120.0


class Worker:
    def __init__(self, engine, index_dir: Path, started: float | None = None, opener=open,
                 parse_workers: int = int(os.environ.get("SB_PARSE_WORKERS", "4"))):
        self.engine, self.index_dir = engine, Path(index_dir)
        self.started = started or time.time()
        self.opener, self.parse_workers = opener, parse_workers
        self.index = Index.load(self.index_dir)
        self.indexing = False
        self.indexed_once = False        # only the first index is charged to the container's startup budget
        self.index_lock = threading.Lock()
        self.engine_lock = threading.Lock()

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "status":
            return {"ok": True, "engine": getattr(self.engine, "name", "?"), "indexing": self.indexing,
                    "indexed": self.index is not None, "stats": self.index.stats if self.index else {}}
        if op == "index":
            return self._index(req)
        if op == "query":
            return self._query(req)
        return {"ok": False, "error": f"unknown op {op!r}"}

    def _index(self, req: dict) -> dict:
        if self.indexed_once:
            remaining = INDEX_BUDGET_S
        else:
            remaining = self.started + INDEX_BUDGET_S - time.time()
        if req.get("deadline_epoch"):
            remaining = min(remaining, float(req["deadline_epoch"]) - time.time())
        deadline = time.monotonic() + max(MIN_INDEX_S, remaining)
        with self.index_lock:
            self.indexing = True
            try:
                with self.engine_lock:
                    idx = build_index(Path(req["corpus"]), self.engine, self.index_dir, deadline,
                                      workers=self.parse_workers, opener=self.opener)
                self.index = idx
                self.indexed_once = True
            finally:
                self.indexing = False
        return {"ok": True, "stats": idx.stats, "skipped": idx.ledger}

    def _query(self, req: dict) -> dict:
        def refuse(reason):
            return {"ok": True, "answer": "", "citations": [], "confidence": 0.0, "diag": {"reason": reason}}
        if self.indexing:
            return refuse("indexing in progress")
        if self.index is None:
            self.index = Index.load(self.index_dir)
        if self.index is None:
            return refuse("no index")
        with self.engine_lock:
            res = pipeline.answer(self.index, self.engine, str(req.get("query", "")), float(req.get("deadline_s", 24)))
        return {"ok": True, **res}


def _serve_conn(worker: Worker, conn: socket.socket) -> None:
    with conn:
        conn.settimeout(900)
        try:
            resp = worker.handle(protocol.recv_line(conn))
        except Exception:  # noqa: BLE001 - a bad request must never kill the worker
            resp = {"ok": False, "error": traceback.format_exc(limit=3)}
        try:
            protocol.send_line(conn, resp)
        except OSError:
            pass


def serve(worker: Worker, addr: str | None = None, ready_file: str | None = None,
          stop: threading.Event | None = None) -> None:
    srv = protocol.listen(addr)
    srv.settimeout(0.5)
    if ready_file:
        Path(ready_file).write_text(json.dumps({"pid": os.getpid(), "addr": addr or protocol.address(),
                                                "startup_s": round(time.time() - worker.started, 2)}))
    try:
        while not (stop and stop.is_set()):
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            threading.Thread(target=_serve_conn, args=(worker, conn), daemon=True).start()
    finally:
        srv.close()


def make_engine():
    if os.environ.get("SB_ENGINE", "qwen") == "fake":
        from .engine import FakeEngine
        return FakeEngine.from_env()
    from .engine_qwen import QwenEngine
    return QwenEngine(os.environ.get("SB_READER", "/models/reader"), os.environ.get("SB_EMBEDDER", "/models/embedder"))


def main() -> None:
    started = float(os.environ.get("SB_STARTED_EPOCH", time.time()))
    engine = make_engine()
    engine.warm_up()
    worker = Worker(engine, Path(os.environ.get("SB_INDEX_DIR", "/app/index")), started=started)
    print(f"sourcebound worker ready in {time.time() - started:.1f}s engine={getattr(engine, 'name', '?')}", flush=True)
    serve(worker, ready_file=os.environ.get("SB_READY_FILE", "/tmp/sourcebound.ready"))


if __name__ == "__main__":
    main()
````

`sourcebound/supervisor.py`:

````python
"""Container CMD: keep a worker running and never exit (spec §2, user stories 5 and 45)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

MAX_RESTARTS = int(os.environ.get("SB_MAX_RESTARTS", "5"))


def main() -> None:
    os.environ.setdefault("SB_STARTED_EPOCH", str(time.time()))
    log_path = os.environ.get("SB_WORKER_LOG", "/tmp/sourcebound-worker.log")
    child: list[subprocess.Popen] = []

    def stop(signum, frame):
        for c in child:
            c.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    restarts = 0
    while True:
        with open(log_path, "a", encoding="utf-8") as log:
            p = subprocess.Popen([sys.executable, "-m", "sourcebound.worker"], stdout=log, stderr=subprocess.STDOUT)
            child[:] = [p]
            rc = p.wait()
        print(f"sourcebound supervisor: worker exited rc={rc}", flush=True)
        restarts += 1
        if restarts > MAX_RESTARTS:
            print("sourcebound supervisor: restart limit reached; staying alive", flush=True)
            while True:
                time.sleep(3600)
        time.sleep(min(30, 2 ** restarts))


if __name__ == "__main__":
    main()
````

`mc3/app.py`:

````python
#!/usr/bin/env python3
"""MC3 harness entry point (spec §2). Thin client: standard library + sourcebound.protocol only.

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

Every query writes /app/output/<query-id>_output.json with "answer" and "citations", whatever happens.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]          # /app in the image; repo root in development
from sourcebound import protocol  # noqa: E402

OUTPUT_DIR = Path(os.environ.get("SB_OUTPUT_DIR", "/app/output"))
QUERY_BUDGET_S = float(os.environ.get("SB_QUERY_BUDGET_S", "25"))
READY_WAIT_S = float(os.environ.get("SB_READY_WAIT_S", "560"))
REFUSAL = {"answer": "", "citations": [], "confidence": 0.0}


def write_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def clean(resp: dict, corpus: Path) -> dict:
    """Enforce the output schema. An answer without citations, or citations without an answer, is a refusal."""
    answer = resp.get("answer")
    answer = answer.strip() if isinstance(answer, str) else ""
    cites, root = [], corpus.as_posix().rstrip("/") + "/"
    for c in resp.get("citations") or []:
        if isinstance(c, str) and c.strip():
            c = c.strip().replace("\\", "/")
            c = c[len(root):] if c.startswith(root) else c
            while c.startswith("./"):
                c = c[2:]
            if c not in cites:
                cites.append(c)
    if not answer or not cites:
        return dict(REFUSAL)
    conf = resp.get("confidence")
    conf = max(0.0, min(1.0, float(conf))) if isinstance(conf, (int, float)) else 0.5
    return {"answer": answer, "citations": cites, "confidence": conf}


def alive() -> bool:
    try:
        return bool(protocol.request({"op": "status"}, timeout=2.0).get("ok"))
    except Exception:
        return False


def spawn_supervisor() -> None:
    if os.environ.get("SB_NO_SPAWN") == "1":          # tests: never leave a background worker behind
        return
    kw = {"start_new_session": True} if os.name == "posix" else {}
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(HERE), str(HERE.parent), os.environ.get("PYTHONPATH", "")])}
    subprocess.Popen([sys.executable, "-m", "sourcebound.supervisor"], cwd=str(HERE), env=env,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)


def wait_ready(seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if alive():
            return True
        time.sleep(1.0)
    return False


def do_index(corpus: Path) -> int:
    if not alive():
        print("sourcebound: worker not running; starting supervisor", file=sys.stderr)
        spawn_supervisor()
        if not wait_ready(READY_WAIT_S):
            print("sourcebound: worker never became ready", file=sys.stderr)
            return 1
    try:
        resp = protocol.request({"op": "index", "corpus": str(corpus.resolve())}, timeout=READY_WAIT_S)
    except Exception as e:  # noqa: BLE001 - the worker is up; queries will answer or refuse on their own
        resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps({"ok": resp.get("ok"), "stats": resp.get("stats"), "skipped": resp.get("skipped"),
                      "error": resp.get("error")}, ensure_ascii=False), file=sys.stderr)
    return 0                                       # non-zero only when the worker never became ready


def do_query(corpus: Path, qid: str, query: str, t0: float) -> int:
    out = OUTPUT_DIR / f"{qid}_output.json"
    try:
        out.unlink()
    except FileNotFoundError:
        pass
    result = dict(REFUSAL)
    try:
        budget = QUERY_BUDGET_S - (time.monotonic() - t0)
        resp = protocol.request({"op": "query", "corpus": str(corpus), "query_id": qid, "query": query,
                                 "deadline_s": max(1.0, budget - 1.0)}, timeout=max(1.0, budget))
        if resp.get("ok"):
            result = clean(resp, corpus)
            log = os.environ.get("SB_DIAG_LOG")
            if log:                                    # development only: keep the pipeline's trace per question
                with open(log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"qid": qid, "query": query, "result": result, "diag": resp.get("diag"),
                                        "client_s": round(time.monotonic() - t0, 3)}, ensure_ascii=False) + "\n")
        else:
            print(f"sourcebound: worker error: {resp.get('error')}", file=sys.stderr)
    except (ConnectionRefusedError, FileNotFoundError) as e:
        print(f"sourcebound: worker unreachable ({e}); starting it for later questions", file=sys.stderr)
        try:
            spawn_supervisor()
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001 - timeouts and bad replies become a refusal, never a crash
        print(f"sourcebound: {type(e).__name__}: {e}", file=sys.stderr)
    write_atomic(out, result)
    return 0


def main() -> int:
    t0 = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    a = ap.parse_args()
    if a.index is not None:
        return do_index(a.index)
    if a.corpus is None or a.query is None or not a.query_id:
        ap.error("a query needs --corpus, --query-id and --query")
    return do_query(a.corpus, a.query_id, a.query, t0)


if __name__ == "__main__":
    sys.exit(main())
````

`eval_mc3/score.py`:

````python
"""Grader-faithful scoring: official answer normalization + exact citation sets (spec Testing Decisions)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sourcebound.normalize import official  # noqa: E402

CORPUS_PREFIXES = ("/app/corpus/",)


def norm_cite(c: str, corpus: str = "") -> str:
    c = c.replace("\\", "/")
    for p in (*CORPUS_PREFIXES, corpus.replace("\\", "/").rstrip("/") + "/" if corpus else None):
        if p and c.startswith(p):
            c = c[len(p):]
    while c.startswith("./"):
        c = c[2:]
    return c


def score_one(pred: dict | None, q: dict, corpus: str = "") -> dict:
    """pred = parsed output JSON (None if missing/malformed); q = a sample-questions.json entry."""
    if not isinstance(pred, dict) or not isinstance(pred.get("answer"), str) or not isinstance(pred.get("citations"), list):
        return {"answer_ok": False, "cite_ok": False, "strict": False, "kind": "malformed"}
    want_a, want_c = q.get("expected_answer", ""), {norm_cite(c) for c in q.get("expected_citations", [])}
    got_c = {norm_cite(c, corpus) for c in pred["citations"] if isinstance(c, str)}
    if want_a == "":
        a_ok = pred["answer"] == ""
    else:
        a_ok = official(pred["answer"]) == official(want_a)      # aliases deliberately ignored (spec G7)
    c_ok = got_c == want_c
    if a_ok and c_ok:
        kind = "ok"
    elif want_a == "" and pred["answer"]:
        kind = "false_answer"
    elif want_a and not pred["answer"]:
        kind = "false_refusal"
    elif not a_ok:
        kind = "wrong_answer"
    elif got_c > want_c:
        kind = "over_citation"
    elif got_c < want_c:
        kind = "under_citation"
    else:
        kind = "wrong_citation"
    return {"answer_ok": a_ok, "cite_ok": c_ok, "strict": a_ok and c_ok, "kind": kind}
````

- [ ] **Step 4: Run it and watch it pass**

Run: `python -m pytest -q tests/test_sb_contract.py`
Expected: `7 passed`. `test_sample_kit_scores_200_through_the_contract` runs `--index` and all ten kit questions as separate `app.py` processes and asserts **10/10 strict**

- [ ] **Step 5: Commit**

```bash
git add tests/test_sb_contract.py sourcebound/protocol.py sourcebound/worker.py sourcebound/supervisor.py mc3/app.py eval_mc3/score.py
git commit -m "feat(mc3): resident worker, supervisor and stdlib thin client; kit scores 10/10 via the contract" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 10: Evaluation tooling and the synthetic corpus generator

Spec: Testing Decisions (evaluation corpora, metrics)

**Files:**
- Create: `eval_mc3/vram.py`
- Create: `eval_mc3/run_eval.py`
- Create: `eval_mc3/run_suite.py`
- Create: `eval_mc3/generate_corpus.py`
- Test: `tests/test_sb_generated.py`

The generator reuses no sample answer. Each corpus has 16 questions covering every trap; the encrypted, unreadable and sibling-attribute questions are refusals. The oracle evidence lets the gates be tested on fresh names and values without a GPU.

- [ ] **Step 1: Write the failing test** — `tests/test_sb_generated.py`:

````python
"""The generator must produce self-consistent corpora, and the pipeline (with a FakeEngine replaying the oracle
evidence) must score every generated question through the gates. This exercises supersession, qualifiers, units,
second sheets, near-miss rows, both chain types and the necessity rule on fresh names and values."""
import json
import os
import time

import pytest

from eval_mc3.generate_corpus import generate
from eval_mc3.score import score_one
from sourcebound import pipeline
from sourcebound.engine import FakeEngine
from sourcebound.index import build_index


@pytest.fixture(scope="module", params=[11, 12])
def gen(request, tmp_path_factory):
    out = tmp_path_factory.mktemp(f"g{request.param}")
    generate(out, request.param)
    return out


def oracle_replies(qs):
    reps = []
    for q in qs:
        if q["expected_answer"]:
            reps.append((q["query"], {"status": "answered", "answer": q["expected_answer"], "answer_type": "extracted",
                                      "evidence": q["oracle"], "lookup": []}))
        else:
            reps.append((q["query"], {"status": "not_found", "answer": "", "evidence": []}))
    return reps


def test_generated_corpus_is_self_consistent(gen):
    qs = json.loads((gen / "questions.json").read_text())["queries"]
    assert len(qs) == 16
    for q in qs:
        for c in q["expected_citations"]:
            assert (gen / "corpus" / c).exists(), c
    assert (gen / "corpus" / "archive").is_dir()


def test_suite_runner_end_to_end_with_oracle_worker(gen, tmp_path):
    """run_suite -> run_eval -> app.py processes -> worker (FakeEngine replaying the oracle) -> aggregate summary."""
    import socket
    import subprocess
    import sys
    import threading
    from pathlib import Path

    from sourcebound.worker import Worker, serve

    qs = json.loads((gen / "questions.json").read_text())["queries"]
    eng = FakeEngine(replies=oracle_replies(qs), transcripts=json.loads((gen / "transcripts.json").read_text()))
    if hasattr(socket, "AF_UNIX"):
        addr = f"unix:{tmp_path / 's.sock'}"
    else:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        addr = f"tcp:127.0.0.1:{s.getsockname()[1]}"
        s.close()
    stop = threading.Event()
    t = threading.Thread(target=serve, args=(Worker(eng, tmp_path / "idx", parse_workers=2), addr, None, stop), daemon=True)
    t.start()
    root = Path(__file__).resolve().parent.parent
    split = tmp_path / "split"
    split.mkdir()
    import shutil
    shutil.copytree(gen, split / "c1")
    env = {**os.environ, "SB_ADDR": addr, "SB_NO_SPAWN": "1", "PYTHONPATH": str(root),
           "SB_OUTPUT_DIR": str(tmp_path / "out")}
    r = subprocess.run([sys.executable, str(root / "eval_mc3" / "run_suite.py"), "--split", str(split), "--tag", "t",
                        "--work", str(tmp_path / "work"), "--results", str(tmp_path / "res")],
                       env=env, capture_output=True, text=True, timeout=600)
    stop.set()
    agg = json.loads(r.stdout.strip().splitlines()[-1])
    assert (agg["strict"], agg["total"], agg["violations"]) == (16, 16, 0), r.stdout + r.stderr
    assert (tmp_path / "res" / "t.diag.jsonl").exists()


def test_oracle_evidence_passes_the_gates(gen, tmp_path):
    qs = json.loads((gen / "questions.json").read_text())["queries"]
    transcripts = json.loads((gen / "transcripts.json").read_text())
    (gen / "corpus" / "vendor" / "internal_audit.txt").unlink(missing_ok=True)   # stands in for chmod 000 on Windows
    idx = build_index(gen / "corpus", FakeEngine(transcripts=transcripts), tmp_path / "idx", time.monotonic() + 300,
                      workers=2, log=lambda *a: None)
    eng = FakeEngine(replies=oracle_replies(qs))
    bad = []
    for q in qs:
        out = pipeline.answer(idx, eng, q["query"], 20)
        s = score_one(out, q)
        if not s["strict"]:
            bad.append((q["n"], q["category"], out["answer"], out["citations"], s["kind"], out["diag"].get("reason")))
    assert bad == []
````

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest -q tests/test_sb_generated.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval_mc3.generate_corpus'`

- [ ] **Step 3: Implement**

`eval_mc3/vram.py`:

````python
"""Sample used VRAM once per second with rocm-smi while an evaluation runs (no-op where rocm-smi is absent)."""
from __future__ import annotations

import json
import shutil
import subprocess
import threading


def used_bytes() -> int | None:
    if not shutil.which("rocm-smi"):
        return None
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], capture_output=True, text=True, timeout=10).stdout
        data = json.loads(out)
        return sum(int(v.get("VRAM Total Used Memory (B)", 0)) for v in data.values() if isinstance(v, dict))
    except Exception:
        return None


class VramSampler:
    def __init__(self, period_s: float = 1.0):
        self.period_s, self.peak, self._stop = period_s, None, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            b = used_bytes()
            if b is not None:
                self.peak = max(self.peak or 0, b)
            self._stop.wait(self.period_s)

    def start(self) -> None:
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        self._t.join(5)

    @property
    def peak_gib(self) -> float | None:
        return None if self.peak is None else round(self.peak / 2**30, 2)
````

`eval_mc3/run_eval.py`:

````python
"""Run the MC3 contract exactly like the harness: one --index, then one fresh process per question.

    python eval_mc3/run_eval.py --questions <questions.json> --corpus <dir> --out results/<tag>.jsonl [--skip-index]

Appends one JSON line per question (survives a quota cut-off) and prints a summary JSON as the last stdout line.
Strict score = answer AND exact citation set right, 20 points each.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_mc3.score import score_one  # noqa: E402
from eval_mc3.vram import VramSampler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--app", default=str(ROOT / "mc3" / "app.py"))
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--timeout", type=float, default=35.0)
    a = ap.parse_args()
    qs = json.loads(Path(a.questions).read_text(encoding="utf-8"))["queries"]
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outdir = Path(os.environ.get("SB_OUTPUT_DIR", str(out_path.parent / "app_output")))
    env = {**os.environ, "SB_OUTPUT_DIR": str(outdir), "SB_DIAG_LOG": str(out_path.with_suffix(".diag.jsonl"))}
    vram = VramSampler()
    vram.start()
    index_s, index_stats = None, None
    if not a.skip_index:
        t = time.monotonic()
        r = subprocess.run([sys.executable, a.app, "--index", a.corpus], env=env, capture_output=True, text=True, timeout=660)
        index_s = round(time.monotonic() - t, 2)
        print(f"index: rc={r.returncode} {index_s}s {r.stderr.strip()[-400:]}", file=sys.stderr)
        for line in reversed(r.stderr.strip().splitlines()):
            try:
                index_stats = json.loads(line).get("stats")
                break
            except (ValueError, AttributeError):
                continue
    by = defaultdict(lambda: [0, 0])
    kinds: Counter = Counter()
    lat, violations, total_t = [], 0, time.monotonic()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, q in enumerate(qs, start=1):
            qid = f"query_{i:02d}"
            t = time.monotonic()
            try:
                subprocess.run([sys.executable, a.app, "--corpus", a.corpus, "--query-id", qid, "--query", q["query"]],
                               env=env, timeout=a.timeout, capture_output=True, text=True)
            except subprocess.TimeoutExpired:
                pass
            dt = time.monotonic() - t
            try:
                pred = json.loads((outdir / f"{qid}_output.json").read_text(encoding="utf-8"))
            except Exception:
                pred = None
            s = score_one(pred, q, a.corpus)
            violations += int(dt >= 30 or s["kind"] == "malformed")
            cat = q.get("category", "all").split(",")[0].strip()
            by[cat][0] += int(s["strict"])
            by[cat][1] += 1
            kinds[s["kind"]] += 1
            lat.append(dt)
            f.write(json.dumps({"qid": qid, "query": q["query"], "expected": q.get("expected_answer"),
                                "expected_citations": q.get("expected_citations"), "pred": pred, **s,
                                "seconds": round(dt, 3)}, ensure_ascii=False) + "\n")
            f.flush()
    vram.stop()
    lat.sort()
    strict = sum(v[0] for v in by.values())
    print(json.dumps({"strict": strict, "total": len(qs), "points": 20 * strict, "max_points": 20 * len(qs),
                      "by_category": dict(by), "kinds": dict(kinds), "index_s": index_s, "index_stats": index_stats,
                      "run_s": round(time.monotonic() - total_t, 2),
                      "p50_s": round(lat[len(lat) // 2], 3) if lat else 0, "max_s": round(lat[-1], 3) if lat else 0,
                      "peak_vram_gib": vram.peak_gib, "violations": violations}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

`eval_mc3/run_suite.py`:

````python
"""Evaluate every corpus of a split, one after another, against the running worker; aggregate one summary.

    python eval_mc3/run_suite.py --split eval_mc3/data/dev --tag dev-q3vl4b-e1 [--work /tmp/sb-suite]
    python eval_mc3/run_suite.py --split eval_mc3/kit --tag kit-q3vl4b-e1

A split is a directory of c<seed>/ folders (questions.json + corpus/ + hazards.json) or the starter kit
(sample-questions.json + mc3-corpus/). Each corpus is COPIED to --work before its hazards are applied
(chmod 000, empty dirs), so the synced sources stay intact. Results: results/<tag>.jsonl (+ .diag.jsonl) and
results/<tag>.summary (last line = aggregate JSON).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT_HAZARDS = {"chmod000": ["vendor/internal_audit.txt"], "empty_dirs": ["archive"]}


def corpora(split: Path) -> list[tuple[str, Path, Path, dict]]:
    if (split / "sample-questions.json").exists():
        return [("kit", split / "sample-questions.json", split / "mc3-corpus", KIT_HAZARDS)]
    out = []
    for d in sorted(p for p in split.iterdir() if p.is_dir() and (p / "questions.json").exists()):
        hz = json.loads((d / "hazards.json").read_text()) if (d / "hazards.json").exists() else {}
        out.append((d.name, d / "questions.json", d / "corpus", hz))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "sb-suite")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    a.results.mkdir(parents=True, exist_ok=True)
    out = a.results / f"{a.tag}.jsonl"
    for p in (out, out.with_suffix(".diag.jsonl")):
        p.unlink(missing_ok=True)
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        print("note: running as root; chmod 000 only bites if the WORKER runs without DAC_OVERRIDE "
              "(remote-mc3.js worker-start nodac)", file=sys.stderr)
    parts, total = [], Counter()
    for name, questions, corpus, hz in corpora(a.split):
        work = a.work / name
        if work.exists():
            for p in work.rglob("*"):
                try:
                    p.chmod(0o755 if p.is_dir() else 0o644)
                except OSError:
                    pass
            shutil.rmtree(work)
        shutil.copytree(corpus, work)
        for d in hz.get("empty_dirs", []):
            (work / d).mkdir(parents=True, exist_ok=True)
        for f in hz.get("chmod000", []):
            if (work / f).exists():
                os.chmod(work / f, 0)
        r = subprocess.run([sys.executable, str(ROOT / "eval_mc3" / "run_eval.py"), "--questions", str(questions),
                            "--corpus", str(work), "--out", str(out)], capture_output=True, text=True)
        try:
            s = json.loads(r.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            s = {"error": (r.stderr or r.stdout)[-800:]}
        s["corpus"] = name
        parts.append(s)
        print(json.dumps(s), flush=True)
        for k in ("strict", "total", "violations"):
            total[k] += s.get(k, 0) or 0
    kinds = Counter()
    for s in parts:
        kinds.update(s.get("kinds", {}))
    agg = {"tag": a.tag, "strict": total["strict"], "total": total["total"], "points": 20 * total["strict"],
           "kinds": dict(kinds), "violations": total["violations"],
           "max_s": max((s.get("max_s") or 0 for s in parts), default=0),
           "max_index_s": max((s.get("index_s") or 0 for s in parts), default=0),
           "peak_vram_gib": max((s.get("peak_vram_gib") or 0 for s in parts), default=0),
           "corpora": [{k: s.get(k) for k in ("corpus", "strict", "total", "index_s", "index_stats", "max_s", "error")} for s in parts]}
    (a.results / f"{a.tag}.summary").write_text("\n".join(json.dumps(s) for s in parts) + "\n" + json.dumps(agg) + "\n")
    print(json.dumps(agg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

`eval_mc3/generate_corpus.py`:

````python
"""Seeded generator of MC3-style corpora with every trap of the sample kit, but new names and values.

    python eval_mc3/generate_corpus.py --out eval_mc3/data/dev --seeds 101 102 103 104
    python eval_mc3/generate_corpus.py --out eval_mc3/data/holdout --seeds 901 902

Each <out>/c<seed>/ holds: corpus/, questions.json (sample-questions.json format plus an "oracle" per question),
transcripts.json (ground-truth image text, for FakeEngine runs) and hazards.json (files to chmod 000).
Nothing from the starter-kit answers is reused. Spec: Testing Decisions, "Evaluation corpora".
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

COMPANIES = ["Halvard Dynamics", "Brightwater Silicon", "Cobalt Reach Systems", "Nimbus Forge", "Tessellate Labs",
             "Quillon Devices", "Verdant Logic", "Ostrava Compute", "Kestrel Microworks", "Lumen Arc"]
PREFIXES = ["HX", "BW", "CR", "NF", "TL", "QD", "VL", "OC", "KM", "LA"]
CODENAMES = ["Basalt", "Cinder", "Drift", "Ember", "Fjord", "Garnet", "Heron", "Ibis", "Juniper", "Kelp"]
FIRMWARE = ["Aurora", "Beacon", "Corvid", "Dynamo", "Eclipse", "Fathom", "Gossamer", "Halcyon"]
SERVICES = [("ingest", "batch timeout", "BATCH_TIMEOUT_S"), ("export", "upload timeout", "UPLOAD_TIMEOUT_S"),
            ("scheduler", "lease timeout", "LEASE_TIMEOUT_S"), ("telemetry", "flush interval", "FLUSH_INTERVAL_S")]
SIGNALS = ["THERM_ALERT#", "FAN_TACH", "PWR_GOOD", "SMB_ALERT#", "RESET_N", "PRSNT#"]
GENERIC = ["GND", "SDA", "SCL", "3V3_AUX", "GND", "12V_SENSE"]


# ------------------------------------------------------------------ writers
def write_pdf(path: Path, pages: list[list[str]], user_pw: str | None = None) -> None:
    import fitz
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        y = 72
        for ln in lines:
            page.insert_text((60, y), ln, fontsize=11, fontname="helv")
            y += 18
    if user_pw:
        doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw=user_pw + "-owner", user_pw=user_pw)
    else:
        doc.save(path)


def write_docx(path: Path, paragraphs: list[str], table: list[list[str]] | None = None) -> None:
    def p(t):
        return f'<w:p><w:r><w:t xml:space="preserve">{escape(t)}</w:t></w:r></w:p>'
    body = "".join(p(t) for t in paragraphs)
    if table:
        body += "<w:tbl>" + "".join("<w:tr>" + "".join(f"<w:tc>{p(c)}</w:tc>" for c in r) + "</w:tr>" for r in table) + "</w:tbl>"
    body += p("End of document.")
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
           f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')
    ct = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" '
          'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
            'relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", doc)


def write_xlsx(path: Path, sheets: dict[str, list[list]]) -> None:
    import openpyxl
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    wb.save(path)


def _font(size):
    from PIL import ImageFont
    return ImageFont.load_default(size=size)


def draw_pinout(path: Path, title: str, pins: list[tuple[str, str]], note: str) -> None:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (1000, 460), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 30), title, fill="black", font=_font(26))
    d.rectangle((40, 90, 960, 300), outline="black", width=3)
    for i, (pin, sig) in enumerate(pins):
        x = 60 + i * 150
        d.rectangle((x, 120, x + 118, 190), outline="black", width=2)
        d.text((x + 38, 142), pin, fill="black", font=_font(22))
        d.line((x + 59, 190, x + 59, 212), fill="black", width=2)
        d.text((x + 10, 220), sig, fill="black", font=_font(14))
    d.text((40, 330), note, fill="black", font=_font(17))
    im.save(path)


def draw_label(path: Path, lines: list[str], rng: random.Random) -> None:
    from PIL import Image, ImageDraw, ImageFilter
    im = Image.new("RGB", (760, 440), (232, 233, 228))
    lab = Image.new("RGB", (640, 320), "white")
    d = ImageDraw.Draw(lab)
    d.rectangle((0, 0, 639, 319), outline="black", width=4)
    for i, ln in enumerate(lines):
        d.text((28, 30 + i * 52), ln, fill="black", font=_font(26 if i else 28))
    lab = lab.rotate(rng.uniform(-5, 5), expand=True, fillcolor=(232, 233, 228))
    im.paste(lab, (40, 40))
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    px = im.load()
    for _ in range(4000):
        x, y = rng.randrange(im.width), rng.randrange(im.height)
        v = rng.randrange(150, 255)
        px[x, y] = (v, v, v)
    im.save(path, quality=rng.randint(55, 75))


# ------------------------------------------------------------------ world and corpus
def generate(out: Path, seed: int) -> dict:
    rng = random.Random(seed)
    k = rng.randrange(len(COMPANIES))
    company, px = COMPANIES[k], PREFIXES[k]
    base = rng.randint(3, 7) * 10
    cur, nxt, old = f"{px}-{base}", f"{px}-{base + 20}", f"{px}-{base - 20}"
    code = rng.choice(CODENAMES)
    fw = rng.choice(FIRMWARE)
    tk = f"{px}{rng.choice('DQX')}"
    yy = rng.randint(27, 29)
    corpus = out / "corpus"
    for d in ("archive", "engineering", "logs", "planning", "specs", "support", "vendor", "ops"):
        (corpus / d).mkdir(parents=True, exist_ok=True)
    q: list[dict] = []
    transcripts: dict[str, str] = {}

    def add(query, answer, cites, category, oracle):
        q.append({"n": len(q) + 1, "query": query, "expected_answer": answer, "expected_citations": cites,
                  "category": category, "oracle": oracle})

    # --- datasheets: r1 withdrawn, r2 current, three pages, bandwidth only on page 3
    tj1, tj2 = rng.randint(100, 110), rng.randint(85, 97)
    bw1, bw2 = round(rng.uniform(2.0, 3.0), 1), round(rng.uniform(3.2, 5.0), 1)
    power = rng.randint(300, 450)
    slug = cur.lower().replace("-", "")
    r1, r2 = f"specs/{slug}_datasheet_r1_WITHDRAWN.pdf", f"specs/{slug}_datasheet_r2.pdf"
    write_pdf(corpus / r1, [[f'{company} {cur} "{code}" Accelerator', "Datasheet, revision 1 - WITHDRAWN, SUPERSEDED BY REVISION 2",
                             "Preliminary silicon. Do not design to these numbers.", "Electrical and thermal",
                             f"Maximum junction temperature .......... {tj1} C", f"Board power (TBP) ..................... {power - 25} W"],
                            ["Memory", f"Peak bandwidth ........................ {bw1} TB/s"]])
    write_pdf(corpus / r2, [[f'{company} {cur} "{code}" Accelerator', "Datasheet, revision 2 - supersedes revision 1",
                             "Electrical and thermal", f"Maximum junction temperature .......... {tj2} C",
                             f"Board power (TBP) ..................... {power} W"],
                            ["Mechanical", "Form factor ........................... FHFL dual slot", "Weight ................................ 1.9 kg"],
                            ["Memory subsystem", f"Peak bandwidth ........................ {bw2} TB/s",
                             "Revision history", f"r2  Junction temperature corrected to {tj2} C.", "r1  Initial release, preliminary silicon."]])
    add(f"What is the maximum junction temperature of the {cur}?", str(tj2), [r2], "pdf, superseded-document trap",
        [{"file": r2, "quote": f"Maximum junction temperature .......... {tj2} C", "role": "value"}])
    add(f"What is the peak memory bandwidth of the {cur}, in TB/s?", str(bw2), [r2], "pdf, page 3 of a multi-page datasheet",
        [{"file": r2, "quote": f"Peak bandwidth ........................ {bw2} TB/s", "role": "value"}])

    # --- roadmap docx: paragraph + table
    qs, qv = rng.randint(1, 4), rng.randint(1, 4)
    road = f"planning/roadmap_fy{yy}.docx"
    write_docx(corpus / road, [f"{company} - Accelerator Roadmap (Internal)", "Dates are targets, not commitments.",
                               f"{cur} is in volume production. Remaining work is firmware only.",
                               f"{nxt} enters customer sampling in Q{qs} FY{yy}. It doubles memory bandwidth.",
                               f"{old} reaches end of life in Q{rng.randint(1, 4)} FY{yy}."],
               [["product", "milestone", "quarter"], [nxt, "tape-out", f"Q{rng.randint(1, 4)} FY{yy - 1}"],
                [nxt, "volume production", f"Q{qv} FY{yy + 1}"], [cur, "volume production", f"Q{rng.randint(1, 4)} FY{yy - 2}"]])
    add(f"In which quarter does the {nxt} enter customer sampling?", f"Q{qs} FY{yy}", [road], "docx, paragraph",
        [{"file": road, "quote": f"{nxt} enters customer sampling in Q{qs} FY{yy}.", "role": "value"}])
    add(f"In which quarter does the {nxt} reach volume production?", f"Q{qv} FY{yy + 1}", [road], "docx, table",
        [{"file": road, "quote": f"product: {nxt} | milestone: volume production | quarter: Q{qv} FY{yy + 1}", "role": "value"}])

    # --- parts workbook: near-miss rows + a second sheet
    fan = f"{px}-FAN-{rng.randint(1000, 9999)}-{rng.choice('ABC')}"
    fan_old = f"{px}-FAN-{rng.randint(1000, 9999)}-A"
    hsk = f"{px}-HSK-{rng.randint(1000, 9999)}-{rng.choice('ABC')}"
    cbl = f"{px}-CBL-{rng.randint(1000, 9999)}-C"
    cost = round(rng.uniform(40, 140), 2)
    lt_fan, lt_hsk, lt_cbl = rng.randint(10, 40), rng.randint(20, 60), rng.randint(5, 15)
    parts = "support/rma_parts.xlsx"
    write_xlsx(corpus / parts, {
        "parts": [["part_number", "description", "compatible_with", "unit_cost_usd"],
                  [fan, "Fan assembly, dual-rotor, field replaceable", cur, cost],
                  [hsk, "Heatsink, vapour chamber", cur, round(rng.uniform(150, 260), 2)],
                  [fan_old, "Fan assembly, single-rotor (legacy), field replaceable", old, round(rng.uniform(30, 70), 2)],
                  [cbl, "Backplane cable, 400mm", cur, 29.75]],
        "lead_times": [["part_number", "lead_time_days", "supplier"], [fan, lt_fan, "Arcwind"], [hsk, lt_hsk, "Thermacore"],
                       [cbl, lt_cbl, "Linkway"], [fan_old, rng.randint(50, 90), "Arcwind"]]})
    add(f"What is the part number of the field-replaceable fan assembly for the {cur}?", fan, [parts], "xlsx, near-miss row",
        [{"file": parts, "quote": f"part_number: {fan} | description: Fan assembly, dual-rotor, field replaceable | compatible_with: {cur}", "role": "value"}])
    add(f"What is the supplier lead time, in days, for part {fan}?", str(lt_fan), [parts], "xlsx, second sheet",
        [{"file": parts, "quote": f"sheet lead_times | part_number: {fan} | lead_time_days: {lt_fan}", "role": "value"}])

    # --- bug database (target row worded away from the incident vocabulary) + decoy release notes
    tid = f"{tk}-{rng.randint(1200, 1999)}"
    fixv = f"{rng.randint(3, 6)}.{rng.randint(0, 9)}.{rng.randint(2, 7)}"
    a, b, c = (int(x) for x in fixv.split("."))
    bugs = "support/bug_database.csv"
    rows = [["ticket", "component", "severity", "summary", "status", "fixed_in"],
            [tid, "power", "S2", "Die sensor over-reports under sustained small-batch load", "closed", fixv],
            [f"{tk}-{rng.randint(2000, 2999)}", "pcie", "S3", "Doorbell hang under sustained load", "closed", f"{a}.{b}.{c - 1}"],
            [f"{tk}-{rng.randint(3000, 3999)}", "scheduler", "S3", "Occupancy counter drifts after 18h uptime", "open", ""],
            [f"{tk}-{rng.randint(4000, 4999)}", "firmware", "S4", f"Release notes omit ticket list for {fixv}", "open", ""]]
    rng.shuffle(rows[1:])
    with open(corpus / bugs, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    row_text = f"ticket: {tid} | component: power | severity: S2 | summary: Die sensor over-reports under sustained small-batch load | status: closed | fixed_in: {fixv}"
    notes = f"engineering/{fw.lower()}_release_notes.txt"
    (corpus / notes).write_text(f"{fw} firmware - release notes\n\n{a}.{b}.0  Scheduler fairness rework.\n"
                                f"{a}.{b}.{c - 1}  Fixes a rare hang in the doorbell path.\n{a}.{b}.{c + 1}  Packaging only.\n\n"
                                "Fixes are tracked per ticket in the bug database, which is authoritative.\n", encoding="utf-8")
    add(f"Which firmware version fixed ticket {tid}?", fixv, [bugs], "csv, identifier given in the question",
        [{"file": bugs, "quote": row_text, "role": "value"}])

    # --- production log: error code, ticket, replaced part
    err = f"E{rng.randint(1000, 9999)}"
    day = f"2026-{rng.randint(7, 9):02d}-{rng.randint(10, 28):02d}"
    log = f"logs/prod_inference_{day}.log"
    node = f"inference-node-{rng.randint(1, 19):02d}"
    lines = [f"{day}T03:11:04Z {node} {fw.lower()}[2211]: INFO  scheduler: batch {rng.randint(10000, 99999)} accepted",
             f"{day}T03:14:22Z {node} {fw.lower()}[2211]: WARN  thermal: die 0 at {tj2 - 3}C, approaching limit",
             f"{day}T03:14:41Z {node} {fw.lower()}[2211]: ERROR {err}: thermal throttle engaged on die 0, clocks reduced to 60%",
             f"{day}T03:19:03Z {node} {fw.lower()}[2211]: INFO  thermal: throttle released",
             f"{day}T03:19:03Z {node} {fw.lower()}[2211]: INFO  incident logged against {tid}",
             f"{day}T05:40:12Z {node} fieldsvc: INFO  replaced part {hsk} during incident follow-up"]
    (corpus / log).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (corpus / "logs" / f"prod_inference_{day[:-2]}{int(day[-2:]) + 1:02d}.log").write_text(
        f"{day}T01:00:00Z {node} {fw.lower()}[2211]: INFO  scheduler: nominal\n", encoding="utf-8")
    add("What error code is logged when the thermal throttle engages?", err, [log], "log, single-hop",
        [{"file": log, "quote": f"ERROR {err}: thermal throttle engaged on die 0", "role": "value"}])
    add("The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?", fixv,
        [log, bugs], "multi-hop log -> csv",
        [{"file": bugs, "quote": row_text, "role": "value"}, {"file": log, "quote": f"incident logged against {tid}", "role": "link"}])
    add("A part was replaced during the logged incident follow-up. What is that part's supplier lead time in days?", str(lt_hsk),
        [log, parts], "multi-hop log -> xlsx second sheet",
        [{"file": parts, "quote": f"sheet lead_times | part_number: {hsk} | lead_time_days: {lt_hsk}", "role": "value"},
         {"file": log, "quote": f"replaced part {hsk}", "role": "link"}])

    # --- source code constant vs. old value in a comment
    svc, thing, const = rng.choice(SERVICES)
    newv, oldv = rng.choice([90, 120, 180, 240, 300]), rng.choice([30, 45, 60])
    py = f"engineering/{svc}_service.py"
    (corpus / py).write_text(f'"""{svc.title()} service for {company} telemetry."""\n\n'
                             f"# Seconds; raised from {oldv} after the backlog incident.\nDEFAULT_{const} = {newv}\n\n"
                             "MAX_INFLIGHT = 12\n\n\ndef run(job, timeout_s=DEFAULT_" + const + "):\n    return job.start(timeout_s)\n",
                             encoding="utf-8")
    add(f"What is the default {thing}, in seconds, in the {svc} service?", str(newv), [py], "python source, constant vs comment",
        [{"file": py, "quote": f"DEFAULT_{const} = {newv}", "role": "value"}])

    # --- images: pinout diagram and an asset label
    sigs = rng.sample(SIGNALS, 2)
    first = rng.randint(1, 3) * 10
    pins = [(f"B{first + i}", GENERIC[i]) for i in range(6)]
    hit = rng.randrange(6)
    pins[hit] = (pins[hit][0], sigs[0])
    png = f"specs/{slug}_backplane_pinout.png"
    draw_pinout(corpus / png, f"{cur} backplane connector - pin assignment", pins,
                f"{sigs[0]} is the only open-drain pin in the row.")
    transcripts[Path(png).name] = "\n".join(f"{p}: {s}" for p, s in pins) + f"\n{sigs[0]} is the only open-drain pin in the row."
    add(f"Which backplane pin carries {sigs[0]} on the {cur}?", pins[hit][0], [png], "png, image-only",
        [{"file": png, "quote": f"{pins[hit][0]}: {sigs[0]}", "role": "value"}])
    rev = f"REV-{rng.choice('BCDEF')}{rng.randint(1, 4)}"
    sn = f"{px}{rng.randint(1, 9)}-{rng.randint(100000, 999999)}-{rng.randint(10, 99)}"
    jpg = "support/asset_label.jpg"
    draw_label(corpus / jpg, [company, f"MODEL  {cur}", f"BOARD REVISION  {rev}", f"SN  {sn}"], rng)
    transcripts[Path(jpg).name] = f"{company}\nMODEL: {cur}\nBOARD REVISION: {rev}\nSN: {sn}"
    add("What board revision is printed on the asset label?", rev, [jpg], "jpg, image-only, qualifier kept",
        [{"file": jpg, "quote": f"BOARD REVISION: {rev}", "role": "value"}])

    # --- hazards and unanswerables
    vol = rng.choice(["5,000", "10,000", "25,000"])
    write_pdf(corpus / "vendor/supplier_agreement_ENCRYPTED.pdf",
              [[f"Supplier agreement - {cur}", f"Unit price at {vol} unit volume ........ {rng.randint(3000, 9000)} USD"]],
              user_pw=f"pw{seed}")
    add(f"What is the unit price of the {cur} at {vol} unit volume?", "", [], "refusal, answer only in an encrypted file", [])
    audit = f"AUD-{rng.randint(100, 999)}"
    (corpus / "vendor/internal_audit.txt").write_text(f"Internal audit findings. Finding reference {audit}.\n", encoding="utf-8")
    add("What is the reference number of the internal audit finding?", "", [], "refusal, answer only in an unreadable file", [])
    add(f"What is the board power of the {nxt}?", "", [], "refusal, attribute exists only for a sibling product", [])
    (corpus / "vendor/telemetry_capture.dat").write_bytes(bytes(range(256)) * 64)
    (corpus / "ops/oncall_runbook.md").write_text(f"# On-call runbook\n\nFor {err}, check fan curves before escalating.\n",
                                                   encoding="utf-8")

    (out / "questions.json").write_text(json.dumps({"challenge": "mc3-synthetic", "seed": seed, "corpus": "corpus",
                                                    "queries": q}, indent=1), encoding="utf-8")
    (out / "transcripts.json").write_text(json.dumps(transcripts, indent=1), encoding="utf-8")
    (out / "hazards.json").write_text(json.dumps({"chmod000": ["vendor/internal_audit.txt"], "empty_dirs": ["archive"]}),
                                      encoding="utf-8")
    return {"seed": seed, "questions": len(q)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    a = ap.parse_args()
    for s in a.seeds:
        print(json.dumps(generate(Path(a.out) / f"c{s}", s)))


if __name__ == "__main__":
    main()
````

- [ ] **Step 4: Run it and watch it pass**

Run: `python -m pytest -q tests/test_sb_generated.py`
Expected: `6 passed`. Two seeds × (self-consistency, suite runner end to end, oracle through the gates); every generated question scores strict

- [ ] **Step 5: Commit**

```bash
git add tests/test_sb_generated.py eval_mc3/vram.py eval_mc3/run_eval.py eval_mc3/run_suite.py eval_mc3/generate_corpus.py
git commit -m "feat(mc3): harness-faithful eval runner, suite runner and trap-complete corpus generator" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 11: GPU engine (compiled locally, exercised in Task 15)

Spec: §8, §1 (engine interface).

**Files:**
- Create: `sourcebound/engine_qwen.py`

- [ ] **Step 1: Write the engine**

````python
"""GPU engine: one Qwen VLM (reader + transcriber) and one Qwen3 embedding model, BF16, resident (spec §8).

GPU only. Loaded from local directories with local_files_only=True; never touches the network.
"""
from __future__ import annotations

import os
import tempfile
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

TRANSCRIBE = ("Transcribe all text printed in this image exactly as written, keeping every character, hyphen, "
              "underscore and # sign. Put each label on the same line as the thing it labels, for example "
              "'B14: THERM_ALERT#' for a pin and its signal, or 'BOARD REVISION: REV-C2' for a field and its value. "
              "Finish with one line that starts with 'IMAGE:' and says what the image shows. Output plain text only.")
QUERY_TASK = "Given a question about internal technical documents, retrieve the passage that answers it"
MAX_PIXELS = int(os.environ.get("SB_MAX_PIXELS", str(1600 * 1200)))


def load_image(path: str, max_pixels: int = MAX_PIXELS) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im.seek(0)
        im = im.convert("RGB")
    if im.width * im.height > max_pixels:
        s = (max_pixels / (im.width * im.height)) ** 0.5
        im = im.resize((max(28, int(im.width * s)), max(28, int(im.height * s))), Image.LANCZOS)
    return im


class QwenEngine:
    def __init__(self, reader: str, embedder: str, dtype=torch.bfloat16, attn: str = "sdpa"):
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm GPU not visible: refusing a silent CPU fallback")
        self.name = os.path.basename(os.path.realpath(reader.rstrip("/")))     # resolve the mc3-reader symlink
        self.proc = AutoProcessor.from_pretrained(reader, local_files_only=True)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            reader, dtype=dtype, attn_implementation=attn, local_files_only=True).to("cuda").eval()
        self.etok = AutoTokenizer.from_pretrained(embedder, padding_side="left", local_files_only=True)
        self.emb = AutoModel.from_pretrained(embedder, dtype=dtype, attn_implementation=attn,
                                             local_files_only=True).to("cuda").eval()
        self.lock = threading.Lock()

    def _inputs(self, prompts: list[str], images: list[list[str]]):
        texts, flat = [], []
        for prompt, imgs in zip(prompts, images):
            msgs = [{"role": "user", "content": [*({"type": "image"} for _ in imgs), {"type": "text", "text": prompt}]}]
            texts.append(self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                       enable_thinking=False))
            flat += [load_image(p) for p in imgs]
        return self.proc(text=texts, images=flat or None, padding=True, return_tensors="pt").to("cuda")

    @torch.inference_mode()
    def _generate(self, prompts, images, max_new_tokens) -> list[str]:
        with self.lock:
            inputs = self._inputs(prompts, images)
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)
            new = out[:, inputs["input_ids"].shape[1]:]
            return [self.proc.decode(row, skip_special_tokens=True).strip() for row in new]

    def generate(self, task, prompt, images=(), max_new_tokens=200, question=""):
        return self._generate([prompt], [list(images)], max_new_tokens)[0]

    def transcribe(self, paths, batch: int = int(os.environ.get("SB_OCR_BATCH", "4"))):
        out = []
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            out += self._generate([TRANSCRIBE] * len(chunk), [[p] for p in chunk], 384)
        return out

    @torch.inference_mode()
    def embed(self, texts, kind="document", batch: int = 32):
        if kind == "query":
            texts = [f"Instruct: {QUERY_TASK}\nQuery:{t}" for t in texts]
        rows = []
        with self.lock:
            for i in range(0, len(texts), batch):
                tok = self.etok(texts[i:i + batch], padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
                h = self.emb(**tok).last_hidden_state[:, -1]          # last-token pooling (left padding)
                rows.append(F.normalize(h.float(), dim=-1).cpu().numpy())
        return np.concatenate(rows) if rows else np.zeros((0, 1), dtype=np.float32)

    def warm_up(self):
        img = os.path.join(tempfile.gettempdir(), "sourcebound_warmup.png")
        Image.new("RGB", (256, 128), "white").save(img)
        self.transcribe([img])
        self.generate("read", "Reply with OK.", (), 4)
        self.embed(["warm up"], kind="query")

    def peak_gib(self) -> float:
        return round(torch.cuda.max_memory_reserved() / 2**30, 2)
````

- [ ] **Step 2: Check that it compiles, and that the rest of the package never imports it**

Run: `python -m py_compile sourcebound/engine_qwen.py && python -m pytest -q tests/test_sb_*.py`
Expected: no output from `py_compile`; the suite shows `72 passed, 1 skipped` on Windows (73 tests so far; the release tests arrive in Task 12). `worker.make_engine()` imports `engine_qwen` lazily, only when `SB_ENGINE` is not `fake`.

- [ ] **Step 3: Commit**

```bash
git add sourcebound/engine_qwen.py
git commit -m "feat(mc3): Qwen VLM + Qwen3-Embedding GPU engine" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 12: Release tooling: layers, publish, check, rehearsal

Spec: §10, Acceptance checks

**Files:**
- Create: `release/build_layers.py`
- Create: `release/check_image.py`
- Create: `release/publish.sh`
- Create: `release/rehearse.sh`
- Create: `docker/Dockerfile.mc3`
- Test: `tests/test_sb_release.py`

`check_image.py`, `publish.sh` and `rehearse.sh` run against a registry and the GPU pod (Task 21); only the layer builder is unit-tested here.

Four decisions are encoded in this code:
1. **The supervisor goes in `CMD`, never `ENTRYPOINT`** (spec §10). `docker run <image> python3 /app/app.py --index ...` must run that command, and `app.py` then starts the supervisor itself. `publish.sh` stops if the base ever gains an ENTRYPOINT.
2. **`push-weights` moves one group at a time** through the runner: download, tar, append, delete. Its disk peak is about one group plus its tar, not the whole model.
3. **A shard larger than 4 GiB becomes its own layer.** Qwen3-VL-8B shards are about 4.3–4.9 GB. `check_image.py` allows 5 GiB per layer, and Docker Hub has no 10 GB cap.
4. **`crane mutate --workdir` is optional.** It exists in current go-containerregistry releases; if `crane mutate --help` on the runner lacks it, delete that flag. Nothing depends on the working directory.

- [ ] **Step 1: Write the failing test** — `tests/test_sb_release.py`:

````python
import tarfile

from release.build_layers import LAYER_MAX, build_app, build_weights


def test_app_layer_has_contract_paths(tmp_path):
    names = tarfile.open(build_app(tmp_path)).getnames()
    for want in ("app/app.py", "app/requirements.txt", "app/sourcebound/worker.py", "app/corpus", "app/output", "app/index"):
        assert want in names
    assert not any("__pycache__" in n for n in names)
    assert all(m.uid == 0 and m.mtime == 1_790_000_000 for m in tarfile.open(build_app(tmp_path)).getmembers())


def test_weights_split_into_bounded_layers(tmp_path, monkeypatch):
    import release.build_layers as bl
    monkeypatch.setattr(bl, "LAYER_MAX", 10)
    src = tmp_path / "m"
    (src / "sub").mkdir(parents=True)
    for i in range(3):
        (src / f"model-0000{i}.safetensors").write_bytes(b"x" * 6)
    (src / "sub" / "config.json").write_text("{}")
    out = tmp_path / "out"
    out.mkdir()
    tars = bl.build_weights(out, src, "reader")
    assert len(tars) == 3
    names = [n for t in tars for n in tarfile.open(t).getnames()]
    assert "models/reader/sub/config.json" in names and "models/reader/model-00002.safetensors" in names
    assert LAYER_MAX == 4 * 2**30


def test_group_files_bounds_layers_and_isolates_big_shards():
    from release.build_layers import group_files
    items = [("a.safetensors", 3), ("b.safetensors", 3), ("c.safetensors", 12), ("config.json", 1)]
    assert group_files(items, limit=7) == [["a.safetensors", "b.safetensors"], ["c.safetensors"], ["config.json"]]
````

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest -q tests/test_sb_release.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'release.build_layers'`

- [ ] **Step 3: Implement**

`release/build_layers.py`:

````python
"""Build the layer tarballs that `crane append` puts on top of the mandated base (spec §10).

    python release/build_layers.py app  --out layers          # /app code, dirs, requirements.txt
    python release/build_layers.py deps --out layers          # /app/pylib from mc3/requirements.lock (cp314 wheels)
    python release/build_layers.py push-weights --out layers --repo Qwen/Qwen3-VL-8B-Instruct --rev <sha> \
        --slot reader --image <registry/repo:wip-tag>         # /models/reader, one <= 4 GiB group at a time

push-weights downloads ONE group of files, tars it, `crane append`s it and deletes it before the next group, so
the runner never holds more than one group plus its tar (spec Further Notes F). A single file larger than the
group limit (some safetensors shards are ~4.3-4.9 GB) becomes a layer of its own.
Tar entries are root-relative ("app/app.py"), owned by root, with fixed mtimes, so rebuilding gives identical layers.
"""
from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAYER_MAX = 4 * 2**30
MTIME = 1_790_000_000


def _info(name: str, size: int = 0, is_dir: bool = False) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = "root"
    ti.mtime = MTIME
    if is_dir:
        ti.type, ti.mode = tarfile.DIRTYPE, 0o755
    else:
        ti.size, ti.mode = size, 0o644
    return ti


def add_tree(tar: tarfile.TarFile, src: Path, dest: str) -> None:
    tar.addfile(_info(dest, is_dir=True))
    for p in sorted(src.rglob("*")):
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        name = f"{dest}/{p.relative_to(src).as_posix()}"
        if p.is_dir():
            tar.addfile(_info(name, is_dir=True))
        else:
            with open(p, "rb") as f:
                tar.addfile(_info(name, p.stat().st_size), f)


def build_app(out: Path) -> Path:
    path = out / "app.tar"
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
        tar.addfile(_info("app", is_dir=True))
        for d in ("app/corpus", "app/output", "app/index", "models"):
            tar.addfile(_info(d, is_dir=True))
        data = (ROOT / "mc3" / "app.py").read_bytes()
        tar.addfile(_info("app/app.py", len(data)), io.BytesIO(data))
        lock_path = ROOT / "mc3" / "requirements.lock"
        lock = (lock_path if lock_path.exists() else ROOT / "mc3" / "requirements.txt").read_bytes()
        tar.addfile(_info("app/requirements.txt", len(lock)), io.BytesIO(lock))
        add_tree(tar, ROOT / "sourcebound", "app/sourcebound")
    return path


def build_deps(out: Path) -> Path:
    stage = out / "stage-deps"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:", "--python-version", "3.14",
                    "--implementation", "cp", "--platform", "manylinux_2_28_x86_64", "--platform", "manylinux2014_x86_64",
                    "--target", str(stage), "-r", str(ROOT / "mc3" / "requirements.lock")], check=True)
    banned = [p.name for p in stage.iterdir() if p.name.split("-")[0].lower() in ("torch", "torchvision", "numpy", "pil", "pillow", "triton")]
    if banned:
        raise SystemExit(f"refusing to ship packages that shadow the base image: {banned}")
    path = out / "deps.tar"
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
        add_tree(tar, stage, "app/pylib")
    return path


def group_files(items: list[tuple[str, int]], limit: int = LAYER_MAX) -> list[list[str]]:
    """Pack (name, size) items, in name order, into groups whose total stays <= limit (a larger file goes alone)."""
    groups, cur, size = [], [], 0
    for name, s in sorted(items):
        if cur and size + s > limit:
            groups.append(cur)
            cur, size = [], 0
        cur.append(name)
        size += s
    if cur:
        groups.append(cur)
    return groups


def tar_group(out: Path, src: Path, names: list[str], slot: str, k: int) -> Path:
    """Tar the given files (relative to src) as models/<slot>/<name>."""
    path = out / f"weights-{slot}-{k:02d}.tar"
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
        tar.addfile(_info("models", is_dir=True))
        tar.addfile(_info(f"models/{slot}", is_dir=True))
        for rel in names:
            for parent in reversed(Path(rel).parents[:-1]):
                tar.addfile(_info(f"models/{slot}/{parent.as_posix()}", is_dir=True))
            p = src / rel
            with open(p, "rb") as f:
                tar.addfile(_info(f"models/{slot}/{rel}", p.stat().st_size), f)
    return path


def build_weights(out: Path, src: Path, slot: str) -> list[Path]:
    """Local variant (all files already on disk): one tar per group."""
    files = [(p.relative_to(src).as_posix(), p.stat().st_size) for p in src.rglob("*")
             if p.is_file() and ".cache" not in p.parts]
    return [tar_group(out, src, g, slot, k) for k, g in enumerate(group_files(files, LAYER_MAX))]


def push_weights(out: Path, repo: str, rev: str, slot: str, image: str) -> None:
    """Download, tar, crane-append and delete one group at a time (bounded runner disk)."""
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(repo, revision=rev, files_metadata=True)
    items = [(s.rfilename, s.size or 0) for s in info.siblings]
    stage = out / f"stage-{slot}"
    for k, group in enumerate(group_files(items, LAYER_MAX)):
        for name in group:
            hf_hub_download(repo, name, revision=rev, local_dir=stage)
        tar = tar_group(out, stage, group, slot, k)
        shutil.rmtree(stage)
        subprocess.run(["crane", "append", "-b", image, "-f", str(tar), "-t", image], check=True,
                       stdout=subprocess.DEVNULL)
        tar.unlink()
        print(f"{slot}: layer {k} appended ({len(group)} files)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["app", "deps", "push-weights"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo")
    ap.add_argument("--rev")
    ap.add_argument("--slot", choices=["reader", "embedder"])
    ap.add_argument("--image")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.what == "app":
        print(build_app(a.out))
    elif a.what == "deps":
        print(build_deps(a.out))
    else:
        push_weights(a.out, a.repo, a.rev, a.slot, a.image)


if __name__ == "__main__":
    main()
````

`release/check_image.py`:

````python
"""Release checks with crane, no Docker daemon (spec Acceptance checks): base-layer identity, uncompressed size,
config (no entrypoint; CMD = supervisor; env), and no obvious secrets. Prints one JSON line; exits 1 on any failure.

    MC3_IMAGE=docker.io/<user>/<repo>:<tag> python release/check_image.py [--base-bytes N]

--base-bytes skips re-streaming the 29 GiB base when its uncompressed size was already measured (gate G1).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import zlib

BASE = "docker.io/rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0"
LIMIT = 60 * 2**30
PLAT = ["--platform", "linux/amd64"]


def crane(*args: str) -> str:
    return subprocess.run(["crane", *args], check=True, capture_output=True, text=True).stdout


def layers(ref: str) -> list[dict]:
    return json.loads(crane("manifest", *PLAT, ref))["layers"]


def uncompressed_bytes(ref: str, digest: str) -> int:
    repo = ref.rsplit(":", 1)[0] if "@" not in ref else ref.split("@")[0]
    p = subprocess.Popen(["crane", "blob", f"{repo}@{digest}"], stdout=subprocess.PIPE)
    d, n = zlib.decompressobj(16 + zlib.MAX_WBITS), 0
    for chunk in iter(lambda: p.stdout.read(1 << 20), b""):
        n += len(d.decompress(chunk))
    n += len(d.flush())
    if p.wait() != 0:
        raise RuntimeError(f"crane blob failed for {digest}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-bytes", type=int)
    a = ap.parse_args()
    img = os.environ["MC3_IMAGE"]
    base_l, img_l = layers(BASE), layers(img)
    r: dict = {"base_layers": len(base_l), "image_layers": len(img_l)}
    r["base_prefix_ok"] = [x["digest"] for x in img_l[:len(base_l)]] == [x["digest"] for x in base_l]
    ours = img_l[len(base_l):]
    r["our_layer_gib"] = [round(x["size"] / 2**30, 2) for x in ours]
    r["largest_layer_ok"] = max((x["size"] for x in ours), default=0) <= 5 * 2**30
    base_bytes = a.base_bytes or sum(uncompressed_bytes(BASE, x["digest"]) for x in base_l)
    our_bytes = sum(uncompressed_bytes(img, x["digest"]) for x in ours)
    r["base_uncompressed_gib"] = round(base_bytes / 2**30, 3)
    r["total_uncompressed_gib"] = round((base_bytes + our_bytes) / 2**30, 3)
    r["size_ok"] = base_bytes + our_bytes <= LIMIT - 2 * 2**30
    cfg = json.loads(crane("config", *PLAT, img))["config"]
    r["entrypoint"], r["cmd"], r["workdir"] = cfg.get("Entrypoint"), cfg.get("Cmd"), cfg.get("WorkingDir")
    env = dict(e.split("=", 1) for e in cfg.get("Env", []))
    r["env_ok"] = env.get("HF_HUB_OFFLINE") == "1" and "/app/pylib" in env.get("PYTHONPATH", "")
    r["entrypoint_ok"] = not cfg.get("Entrypoint") and cfg.get("Cmd") == ["python3", "-m", "sourcebound.supervisor"]
    r["no_secrets"] = not re.search(r"(TOKEN|SECRET|PASSWORD|API_KEY)=", json.dumps(cfg.get("Env", [])), re.I)
    ok = all(r[k] for k in ("base_prefix_ok", "largest_layer_ok", "size_ok", "env_ok", "entrypoint_ok", "no_secrets"))
    r["ok"] = ok
    print(json.dumps(r))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
````

`release/publish.sh`:

````bash
#!/usr/bin/env bash
# Assemble and push the release image in the registry: base layers untouched, our layers appended (spec §10).
# Required env: MC3_IMAGE (registry/repo, from a CI secret; never echoed), TAG,
#               READER_REPO READER_REV EMBEDDER_REPO EMBEDDER_REV (pinned Hugging Face revisions).
set -euo pipefail
: "${MC3_IMAGE:?}" "${TAG:?}" "${READER_REPO:?}" "${READER_REV:?}" "${EMBEDDER_REPO:?}" "${EMBEDDER_REV:?}"
BASE=docker.io/rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
WIP="$MC3_IMAGE:wip-$TAG"
OUT=layers
mkdir -p "$OUT"

# The supervisor goes in CMD (spec §10), so `docker run <image> python3 /app/app.py ...` still runs that command.
# That only holds if the base has no ENTRYPOINT; stop rather than guess if it does.
BASE_CFG=$(crane config --platform linux/amd64 "$BASE")
BASE_EP=$(echo "$BASE_CFG" | python -c "import json,sys; print(json.load(sys.stdin)['config'].get('Entrypoint') or '')")
if [ -n "$BASE_EP" ]; then echo "base image has ENTRYPOINT $BASE_EP; decide how to clear it before publishing"; exit 1; fi

python release/build_layers.py deps --out "$OUT"
python release/build_layers.py app --out "$OUT"
crane append --platform linux/amd64 -b "$BASE" -f "$OUT/deps.tar" -f "$OUT/app.tar" -t "$WIP" > /dev/null
rm -rf "$OUT/deps.tar" "$OUT/app.tar" "$OUT/stage-deps"

# One <= 4 GiB group at a time: download, tar, append, delete.
python release/build_layers.py push-weights --out "$OUT" --repo "$READER_REPO" --rev "$READER_REV" --slot reader --image "$WIP"
python release/build_layers.py push-weights --out "$OUT" --repo "$EMBEDDER_REPO" --rev "$EMBEDDER_REV" --slot embedder --image "$WIP"

# Keep every base env var; prepend our PYTHONPATH to any the base defines.
BASE_PP=$(echo "$BASE_CFG" | python -c "import json,sys; e=dict(x.split('=',1) for x in json.load(sys.stdin)['config'].get('Env',[])); print(e.get('PYTHONPATH',''))")
PP="/app/pylib:/app${BASE_PP:+:$BASE_PP}"
crane mutate "$WIP" --cmd=python3,-m,sourcebound.supervisor --workdir /app \
  --env "PYTHONPATH=$PP" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env SB_READER=/models/reader --env SB_EMBEDDER=/models/embedder --env SB_INDEX_DIR=/app/index \
  -t "$MC3_IMAGE:$TAG" > /dev/null
echo "published tag $TAG"
````

`release/rehearse.sh`:

````bash
#!/usr/bin/env bash
# Release rehearsal on the GPU notebook (spec §10). The notebook pod runs the mandated base, so extracting ONLY
# our appended layers of the pushed image onto / reproduces the image filesystem. Then run the CMD and the harness.
# Usage (via remote-mc3.js rehearse): MC3_IMAGE=docker.io/<user>/<repo>:<tag> bash release/rehearse.sh
# Fallback when the pod cannot reach the registry (gate G5): LOCAL_LAYERS=1 bash release/rehearse.sh rebuilds the
# app and deps layers on the pod from the same commit and links /models to the notebook's model directories.
set -euo pipefail
LOCAL_LAYERS=${LOCAL_LAYERS:-0}
[ "$LOCAL_LAYERS" = 1 ] || : "${MC3_IMAGE:?}"
R=/workspace/mc3
BASE=docker.io/rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
CRANE=/workspace/bin/crane
if [ "$LOCAL_LAYERS" != 1 ] && [ ! -x "$CRANE" ]; then
  mkdir -p /workspace/bin
  curl -sL https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz \
    | tar -xz -C /workspace/bin crane
fi
pkill -f '[s]ourcebound' || true
rm -rf /app /models
if [ "$LOCAL_LAYERS" = 1 ]; then
  (cd $R && python release/build_layers.py deps --out /tmp/sb-layers && python release/build_layers.py app --out /tmp/sb-layers)
  tar -xf /tmp/sb-layers/deps.tar -C / && tar -xf /tmp/sb-layers/app.tar -C /
  mkdir -p /models && ln -sfn "$(readlink -f /workspace/models/mc3-reader)" /models/reader     && ln -sfn "$(readlink -f /workspace/models/mc3-embedder)" /models/embedder
else
  if ! $CRANE manifest --platform linux/amd64 "$MC3_IMAGE" > /dev/null; then
    echo "registry unreachable from the pod (gate G5): rerun with LOCAL_LAYERS=1"; exit 2
  fi
  nb=$($CRANE manifest --platform linux/amd64 "$BASE" | python -c "import json,sys; print(len(json.load(sys.stdin)['layers']))")
  repo=${MC3_IMAGE%:*}
  for d in $($CRANE manifest --platform linux/amd64 "$MC3_IMAGE" | python -c "import json,sys; print(' '.join(l['digest'] for l in json.load(sys.stdin)['layers'][$nb:]))"); do
    $CRANE blob "$repo@$d" | tar -xz -C /
  done
fi
du -shL /app /models/*

# The image's environment (check_image.py verifies the same values in the pushed config).
export PYTHONPATH=/app/pylib:/app HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SB_READER=/models/reader \
       SB_EMBEDDER=/models/embedder SB_INDEX_DIR=/app/index SB_OUTPUT_DIR=/app/output
cd /app
python3 -c "import torch; assert 'rocm' in torch.__version__, torch.__version__; print('torch', torch.__version__)"

T0=$(date +%s)
# Drop DAC_OVERRIDE like the grader when the pod allows it (gate G4), so the mode-000 file is really unreadable.
NODAC=""
if setpriv --bounding-set=-dac_override,-dac_read_search --inh-caps=-dac_override,-dac_read_search true 2>/dev/null; then
  NODAC="setpriv --bounding-set=-dac_override,-dac_read_search --inh-caps=-dac_override,-dac_read_search"
fi
nohup $NODAC python3 -m sourcebound.supervisor > /tmp/rehearse-cmd.log 2>&1 &
cp -r $R/eval_mc3/kit/mc3-corpus/. /app/corpus/ && mkdir -p /app/corpus/archive && chmod 000 /app/corpus/vendor/internal_audit.txt
python3 $R/eval_mc3/run_eval.py --questions $R/eval_mc3/kit/sample-questions.json --corpus /app/corpus \
  --out $R/results/rehearse-kit.jsonl --app /app/app.py | tail -1 | tee $R/results/rehearse-kit.summary
echo "container-start -> end of kit run: $(( $(date +%s) - T0 )) s"

# Liveness: kill the worker; the supervisor must bring it back and answer again.
pkill -9 -f '[s]ourcebound.worker'
sleep 5
for i in $(seq 1 120); do python3 -c "import sys; sys.path[:0]=['/app']; from sourcebound import protocol; protocol.request({'op':'status'},2)" 2>/dev/null && break; sleep 3; done
python3 /app/app.py --corpus /app/corpus --query-id liveness --query "What error code is logged when the thermal throttle engages?"
cat /app/output/liveness_output.json; echo
pgrep -f sourcebound.supervisor > /dev/null && echo "supervisor alive"
````

`docker/Dockerfile.mc3`:

````dockerfile
# REFERENCE ONLY (spec §10). The release is assembled in the registry by release/publish.sh with crane, which
# produces the same filesystem without a 50 GB local build. Use this file only on a Linux Docker host with
# >= 70 GB free, after copying the two model snapshots to models/reader and models/embedder.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
COPY mc3/requirements.lock /app/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: --target /app/pylib -r /app/requirements.txt \
 && PYTHONPATH=/app/pylib python3 -c "import torch; assert 'rocm' in torch.__version__, torch.__version__"
COPY models/reader /models/reader
COPY models/embedder /models/embedder
COPY sourcebound /app/sourcebound
COPY mc3/app.py /app/app.py
RUN mkdir -p /app/corpus /app/output /app/index
ENV PYTHONPATH=/app/pylib:/app HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    SB_READER=/models/reader SB_EMBEDDER=/models/embedder SB_INDEX_DIR=/app/index
WORKDIR /app
CMD ["python3", "-m", "sourcebound.supervisor"]
````

- [ ] **Step 4: Run it and watch it pass**

Run: `python -m pytest -q tests/test_sb_release.py`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_sb_release.py release/build_layers.py release/check_image.py release/publish.sh release/rehearse.sh docker/Dockerfile.mc3
git commit -m "feat(mc3): crane-based release tooling and notebook rehearsal" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 13: CI: unit tests and the grader-isolation contract run

Spec: Testing Decisions ("CI contract job"), user story 54.

**Files:**
- Create: `docker/mc3-contract.Dockerfile`, `release/contract_check.sh`, `.github/workflows/mc3-contract.yml`, `.github/workflows/mc3-release.yml`

- [ ] **Step 1: The CPU contract image and its check script**

`docker/mc3-contract.Dockerfile`:

````dockerfile
# CPU-only contract image for CI (NOT the submission). Same code paths as the release, FakeEngine instead of
# the GPU engine, so the grader's isolation flags (--network none, --cap-drop DAC_OVERRIDE, root) can be
# reproduced on a GitHub-hosted runner. See release/contract_check.sh.
FROM python:3.14-slim
RUN pip install --no-cache-dir pymupdf openpyxl numpy pillow
WORKDIR /app
COPY sourcebound /app/sourcebound
COPY mc3/app.py /app/app.py
COPY eval_mc3/__init__.py eval_mc3/score.py eval_mc3/run_eval.py eval_mc3/vram.py /app/eval_mc3/
COPY eval_mc3/kit /app/eval_mc3/kit
COPY release/contract_check.sh /app/release/contract_check.sh
ENV PYTHONPATH=/app SB_ENGINE=fake SB_INDEX_DIR=/tmp/sb-index SB_OUTPUT_DIR=/app/output \
    SB_FAKE_REPLIES=/fake/replies.json SB_FAKE_TRANSCRIPTS=/fake/transcripts.json
CMD ["sh", "/app/release/contract_check.sh"]
````

`release/contract_check.sh`:

````bash
#!/bin/sh
# Runs INSIDE the CPU contract container (docker/mc3-contract.Dockerfile) with --network none and
# --cap-drop DAC_OVERRIDE, as root, with SB_ENGINE=fake. Proves: the client starts the worker itself,
# indexing survives the hazards, a mode-000 file is unreadable to root, and the kit scores 10/10 strict.
set -eu
if cat /app/corpus/vendor/internal_audit.txt > /dev/null 2>&1; then
  echo "FAIL: root can read the mode-000 file; DAC_OVERRIDE was not dropped"; exit 1
fi
python3 /app/eval_mc3/run_eval.py --questions /app/eval_mc3/kit/sample-questions.json --corpus /app/corpus \
  --out /tmp/contract.jsonl --app /app/app.py | tail -1 > /tmp/summary.json
cat /tmp/summary.json
python3 - <<'EOF'
import json
s = json.load(open("/tmp/summary.json"))
assert s["strict"] == s["total"] == 10, s
assert s["violations"] == 0, s
print("contract OK")
EOF
grep -q "internal_audit.txt" /tmp/sourcebound-worker.log 2>/dev/null || true
````

- [ ] **Step 2: The workflows.** The release workflow reads the image reference **only** from `secrets.MC3_IMAGE`.

`.github/workflows/mc3-contract.yml`:

````yaml
name: MC3 contract (CPU)

on:
  push:
    paths: ['sourcebound/**', 'mc3/**', 'eval_mc3/**', 'tests/**', 'release/**', 'docker/mc3-contract.Dockerfile', '.github/workflows/mc3-contract.yml']
  workflow_dispatch:

jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.14' }
      - run: pip install pytest pymupdf openpyxl numpy pillow
      - name: Unit + contract tests as a NON-root user (real chmod 000 cases run here)
        run: python -m pytest -q tests/test_sb_*.py

  grader-isolation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Prepare the kit corpus with both zip-proof hazards
        run: |
          cp -r eval_mc3/kit/mc3-corpus /tmp/corpus
          mkdir -p /tmp/corpus/archive
          chmod 000 /tmp/corpus/vendor/internal_audit.txt
          python3 tests/fixtures/kit_fake.py /tmp/fake
      - run: docker build -f docker/mc3-contract.Dockerfile -t sb-contract .
      - name: Root, no network, DAC_OVERRIDE dropped (the grader's conditions)
        run: |
          docker run --rm --network none --cap-drop DAC_OVERRIDE \
            -v /tmp/corpus:/app/corpus -v /tmp/fake:/fake:ro sb-contract
````

`.github/workflows/mc3-release.yml`:

````yaml
name: MC3 release (crane append, no docker build)

on:
  workflow_dispatch:
    inputs:
      tag: { description: 'image tag', required: true }
      reader_repo: { description: 'reader VLM repo', default: 'Qwen/Qwen3-VL-8B-Instruct' }
      reader_rev: { description: 'reader revision (commit sha)', required: true }
      embedder_repo: { description: 'embedding repo', default: 'Qwen/Qwen3-Embedding-0.6B' }
      embedder_rev: { description: 'embedder revision (commit sha)', required: true }

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - name: Free runner disk (weights pass through it one <= 4 GiB layer at a time)
        run: |
          sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache/CodeQL
          df -h /
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.14' }
      - name: Install crane and huggingface_hub
        run: |
          curl -sL https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz | sudo tar -xz -C /usr/local/bin crane
          pip install huggingface_hub
      - name: Registry login (Docker Hub; the base lives there too, so base blobs are mounted, not copied)
        run: echo "${{ secrets.DOCKERHUB_TOKEN }}" | crane auth login index.docker.io -u "${{ secrets.DOCKERHUB_USERNAME }}" --password-stdin
      - name: Publish
        env:
          MC3_IMAGE: ${{ secrets.MC3_IMAGE }}
          TAG: ${{ inputs.tag }}
          READER_REPO: ${{ inputs.reader_repo }}
          READER_REV: ${{ inputs.reader_rev }}
          EMBEDDER_REPO: ${{ inputs.embedder_repo }}
          EMBEDDER_REV: ${{ inputs.embedder_rev }}
        run: bash release/publish.sh
      - name: Check the pushed image
        env:
          MC3_IMAGE: ${{ secrets.MC3_IMAGE }}:${{ inputs.tag }}
        run: python release/check_image.py
````

- [ ] **Step 3: Push the branch and check CI.** First ask the user whether this branch may be pushed; the command below pushes it.

Run: `git add docker/mc3-contract.Dockerfile release/contract_check.sh .github/workflows/mc3-contract.yml .github/workflows/mc3-release.yml && git commit -m "ci(mc3): contract job under --network none and --cap-drop DAC_OVERRIDE; crane release workflow" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -u origin feat/sourcebound-mc3`
Then: `gh run watch $(gh run list --workflow mc3-contract.yml --limit 1 --json databaseId -q '.[0].databaseId')`
Expected: both jobs are green.
- `unit` shows `76 passed` (the chmod test runs here as a non-root user).
- `grader-isolation` prints a summary with `"strict": 10, "total": 10, "violations": 0` and then `contract OK`.

If `grader-isolation` fails with `FAIL: root can read the mode-000 file`, the runner did not apply `--cap-drop`. Fix the workflow; do not weaken the check.

### Task 14: Remote GPU workflow and session gates

Spec: Further Notes G (G2–G4), H.

**Files:**
- Create: `tools/amd-gpu/remote-mc3.js`, `eval_mc3/gates.py`, `eval_mc3/worker_ctl.sh`

- [ ] **Step 1: The session gates** — `eval_mc3/gates.py`:

````python
"""MC3 first-session gates (spec Further Notes G2-G4). Prints one JSON line. Run on the GPU notebook."""
import importlib
import json
import os
import shutil
import subprocess
import time


def sh(c: str) -> tuple[int, str]:
    r = subprocess.run(["bash", "-lc", c], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


g = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
import torch  # noqa: E402

g["torch"], g["torch_file"], g["hip"] = torch.__version__, torch.__file__, torch.version.hip
g["torch_is_rocm"] = "rocm" in torch.__version__
g["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
g["gfx"] = sh("rocminfo | grep -m1 -o 'gfx[0-9a-f]*'")[1]
for mod in ("transformers", "tokenizers", "safetensors", "fitz", "openpyxl", "numpy", "PIL"):
    try:
        m = importlib.import_module(mod)
        g[f"{mod}_version"] = getattr(m, "__version__", getattr(m, "VersionBind", "?"))
    except Exception as e:  # noqa: BLE001
        g[f"{mod}_version"] = f"IMPORT FAILED: {e}"
g["unshare_n"] = sh("unshare -n true")[0] == 0
g["unshare_rn"] = sh("unshare -rn true")[0] == 0
NODAC = "setpriv --bounding-set=-dac_override,-dac_read_search --inh-caps=-dac_override,-dac_read_search"
g["setpriv_drop_dac"] = sh(f"{NODAC} true")[0] == 0
g["capsh"] = shutil.which("capsh") is not None
probe = "/tmp/sb_dac_probe.txt"
open(probe, "w").write("x")
os.chmod(probe, 0)
g["root_reads_mode_000"] = sh(f"cat {probe}")[0] == 0
g["dac_drop_blocks_read"] = sh(f"{NODAC} cat {probe}")[0] != 0
g["workspace_free_gb"] = round(shutil.disk_usage("/workspace").free / 1e9, 1)
g["root_free_gb"] = round(shutil.disk_usage("/").free / 1e9, 1)
for slot in ("reader", "embedder"):
    p = f"/workspace/models/mc3-{slot}"
    g[f"{slot}_path"] = os.path.realpath(p) if os.path.exists(p) else None
    g[f"{slot}_bytes"] = int(sh(f"du -sbL {p} | cut -f1")[1] or 0) if os.path.exists(p) else 0
print(json.dumps(g))
````

- [ ] **Step 2: The remote driver** — `tools/amd-gpu/remote-mc3.js`. It reuses `gpu.js` and `lib.js` exactly as ROADREAD's `remote.js` does. Read-only commands never launch a pod.

**Never put the worker's process name in a `gpu.js sh` command.** `gpu.js sh` runs `bash -lc "<cmd>"`, so a command line that contains `sourcebound.worker` anywhere, even in a `nohup python -m sourcebound.worker` later in the same line, is matched by `pkill -f` and killed (`[exit -15]`, which the exit check now reports as a failure). Start and stop therefore live in `eval_mc3/worker_ctl.sh`; the remaining inline patterns (`end`) are bracketed (`'[s]ourcebound'`) and appear nowhere else in their command line.

`eval_mc3/worker_ctl.sh`:

````bash
#!/usr/bin/env bash
# Start or stop the resident worker on the GPU notebook:
#   bash eval_mc3/worker_ctl.sh start [nodac] | stop
# This lives in a file on purpose. gpu.js runs commands as `bash -lc "<command>"`; if that command line itself
# contained "sourcebound.worker", `pkill -f` would match and kill the shell running it. Here the only processes
# whose command lines contain the name are the worker and the supervisor.
set -u
NODAC="setpriv --bounding-set=-dac_override,-dac_read_search --inh-caps=-dac_override,-dac_read_search"
LOG=/workspace/mc3-worker.log

stop() {
  pkill -f '[s]ourcebound.supervisor'   # a supervisor started by app.py would otherwise respawn the worker
  pkill -f '[s]ourcebound.worker'
  sleep 1
  rm -f /tmp/sourcebound.ready /tmp/sourcebound.sock
}

case "${1:-}" in
  stop)
    stop
    echo stopped
    ;;
  start)
    stop
    rm -rf "${SB_INDEX_DIR:-/tmp/sb-index}"        # fresh index and transcript cache for every worker
    PREFIX=""
    [ "${2:-}" = nodac ] && PREFIX="$NODAC"
    nohup $PREFIX python -m sourcebound.worker > "$LOG" 2>&1 &
    WPID=$!
    echo "spawned worker pid $WPID"
    for _ in $(seq 1 300); do
      if [ -f /tmp/sourcebound.ready ]; then cat /tmp/sourcebound.ready; echo; exit 0; fi
      if ! kill -0 "$WPID" 2>/dev/null; then echo "worker exited early:"; tail -40 "$LOG"; exit 1; fi
      sleep 2
    done
    echo "timeout waiting for ready file"; tail -40 "$LOG"; exit 1
    ;;
  *)
    echo "usage: worker_ctl.sh start [nodac] | stop"; exit 2
    ;;
esac
````

````javascript
#!/usr/bin/env node
// Remote SOURCEBOUND (Mini-Challenge 3) workflow on the AMD notebook GPU (built on gpu.js; see GPUWEBSKILL.md).
//   node remote-mc3.js sync                         tar sourcebound/ mc3/ eval_mc3/ (not eval_mc3/data) release/ -> /workspace/mc3
//   node remote-mc3.js data <dir>                   tar a local dir (e.g. eval_mc3/data/dev) -> /workspace/mc3/<dir> (<= ~40 MB)
//   node remote-mc3.js setup                        pip --target /workspace/pylib-mc3, purge torch shadows, freeze -> mc3/requirements.lock
//   node remote-mc3.js model <hf_repo> <slot> [tmp] snapshot_download -> /workspace/models/<name> (or /root/models with tmp); link mc3-<slot>
//   node remote-mc3.js gates                        eval_mc3/gates.py -> results/mc3-gates-<ts>.json
//   node remote-mc3.js worker-start [nodac]         (re)start the resident worker; nodac drops DAC_OVERRIDE like the grader
//   node remote-mc3.js worker-stop | worker-log
//   node remote-mc3.js suite <split_dir> <tag>     every corpus in the split (or the kit): index + one process per question, detached
//   node remote-mc3.js tail <tag> | pull <tag>      progress / fetch results (never launch a pod)
//   node remote-mc3.js rehearse [local]             release/rehearse.sh against $MC3_IMAGE (env var; never committed)
//   node remote-mc3.js end                          stop worker/eval + kernels, then Turn-off Session (never launches)
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { withLab, sh, put, get, killKernels, stop } = require('./gpu');
const { openBrowser, OUT_DIR } = require('./lib');

const REPO = path.resolve(__dirname, '..', '..');
const R = '/workspace/mc3';
// pkill patterns are written '[s]ourcebound...' so they never match the bash -lc that runs them.
const ENV = `export PYTHONPATH=/workspace/pylib-mc3:${R} HF_HOME=/workspace/.cache/huggingface ` +
  `SB_READER=/workspace/models/mc3-reader SB_EMBEDDER=/workspace/models/mc3-embedder SB_INDEX_DIR=/tmp/sb-index ` +
  `SB_OUTPUT_DIR=${R}/results/app_output HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1;`;
const TAR = process.platform === 'win32' ? 'C:\\Windows\\System32\\tar.exe' : 'tar'; // bsdtar; Git's GNU tar breaks on C:
const [cmd, a, b, c] = process.argv.slice(2);
const NO_LAUNCH = new Set(['tail', 'pull', 'worker-stop', 'worker-log', 'end']);
if (NO_LAUNCH.has(cmd)) process.env.AUTO_LAUNCH = '0';

async function run(shell, timeoutMs) {
  return withLab(async (page, base) => {
    const r = await sh(page, base, shell, timeoutMs);
    process.stdout.write(r.output);
    if (!r.ok || /\[exit -?[1-9]/.test(r.output)) process.exitCode = 1;   // -15: killed by a signal
    return r.output;
  });
}

async function upload(relPaths, remoteDir, { excludes = ['__pycache__', 'eval_mc3/data', '*.pyc'] } = {}) {
  const rel = path.relative(REPO, path.join(OUT_DIR, 'upload-mc3.tgz'));
  execFileSync(TAR, ['-czf', rel, ...excludes.map((e) => '--exclude=' + e), ...relPaths], { cwd: REPO });
  const size = fs.statSync(path.join(REPO, rel)).size;
  if (size > 40e6) throw new Error(`archive is ${(size / 1e6).toFixed(1)} MB; split it (put goes through page.evaluate)`);
  return withLab(async (page, base) => {
    await put(page, base, path.join(REPO, rel), 'upload-mc3.tgz');
    const r = await sh(page, base, `mkdir -p ${remoteDir} && tar -xzf /workspace/upload-mc3.tgz --no-same-owner -C ${remoteDir} && rm /workspace/upload-mc3.tgz && ls ${remoteDir}`);
    process.stdout.write(r.output);
  });
}

const cmds = {
  sync: () => upload(['sourcebound', 'mc3', 'eval_mc3', 'release', 'tests/fixtures'], R),
  data: () => upload([a], R, { excludes: ['__pycache__'] }),
  async setup() {
    await run(`${ENV} cd ${R} && H=$(sha1sum mc3/requirements.txt | cut -c1-12); \
if [ "$(cat /workspace/pylib-mc3/.req 2>/dev/null)" != "$H" ]; then rm -rf /workspace/pylib-mc3 && \
pip install -q --target /workspace/pylib-mc3 -r mc3/requirements.txt && \
(cd /workspace/pylib-mc3 && rm -rf torch torch-* torchgen functorch torchvision* torchaudio* triton* numpy numpy-* numpy.libs nvidia* PIL pillow* bin/torch*) && \
echo $H > /workspace/pylib-mc3/.req; fi; \
python -c "import torch,transformers,fitz,openpyxl,PIL,numpy; assert 'rocm' in torch.__version__, torch.__file__; assert '/opt/venv' in torch.__file__; print('torch', torch.__version__, torch.__file__); print('transformers', transformers.__version__, 'fitz', fitz.VersionBind, 'openpyxl', openpyxl.__version__)" && \
pip freeze --path /workspace/pylib-mc3 > mc3/requirements.lock && cat mc3/requirements.lock`, 30 * 60_000);
    if (process.exitCode) return;
    await withLab((page, base) => get(page, base, 'mc3/mc3/requirements.lock', path.join(REPO, 'mc3', 'requirements.lock')));
    console.log('pulled mc3/requirements.lock');
  },
  model: () => {
    if (!['reader', 'embedder'].includes(b)) throw new Error('model <hf_repo> <reader|embedder> [tmp]');
    const dir = c === 'tmp' ? '/root/models' : '/workspace/models';
    return run(`${ENV} N=$(basename ${a}); mkdir -p ${dir} && HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download as d; print(d('${a}', local_dir='${dir}/$N'))" && \
ln -sfn ${dir}/$N /workspace/models/mc3-${b} && du -sh ${dir}/$N && df -h /workspace / | tail -2`, 40 * 60_000);
  },
  async gates() {
    const out = await run(`${ENV} cd ${R} && python eval_mc3/gates.py`, 10 * 60_000);
    const line = out.split('\n').find((l) => l.startsWith('{')) || out;
    fs.mkdirSync(path.join(REPO, 'results'), { recursive: true });
    const file = path.join(REPO, 'results', `mc3-gates-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
    fs.writeFileSync(file, line);
    console.log('saved', file);
  },
  // Start/stop live in eval_mc3/worker_ctl.sh so that no `bash -lc` command line here contains the process name.
  'worker-start': () => run(`${ENV} cd ${R} && bash eval_mc3/worker_ctl.sh start ${a === 'nodac' ? 'nodac' : ''}`, 15 * 60_000),
  'worker-stop': () => run(`cd ${R} && bash eval_mc3/worker_ctl.sh stop`),
  'worker-log': () => run(`tail -60 /workspace/mc3-worker.log`),
  suite: () => {
    if (!a || !b) throw new Error('suite <split_dir relative to /workspace/mc3, e.g. eval_mc3/kit> <tag>');
    return run(`${ENV} cd ${R} && mkdir -p results && (nohup python eval_mc3/run_suite.py --split ${a} --tag ${b} > results/${b}.log 2>&1 &) && echo started ${b}`);
  },
  tail: () => run(`cd ${R}/results && echo "answered: $(wc -l < ${a}.jsonl 2>/dev/null)"; tail -2 ${a}.jsonl 2>/dev/null | cut -c1-400; \
tail -3 ${a}.log 2>/dev/null | cut -c1-600; echo "summary: $(tail -1 ${a}.summary 2>/dev/null)"; rocm-smi --showmeminfo vram | grep -E 'Used|Total'`),
  pull: () => withLab(async (page, base) => {
    fs.mkdirSync(path.join(REPO, 'results'), { recursive: true });
    for (const ext of ['jsonl', 'diag.jsonl', 'summary', 'log']) {
      const local = path.join(REPO, 'results', `${a}.${ext}`);
      await get(page, base, `mc3/results/${a}.${ext}`, local).then(() => console.log('pulled', local)).catch((e) => console.error(ext, e.message));
    }
  }),
  rehearse: () => {
    if (a === 'local') return run(`cd ${R} && LOCAL_LAYERS=1 bash release/rehearse.sh`, 55 * 60_000);
    if (!process.env.MC3_IMAGE) throw new Error('set MC3_IMAGE in your shell (never commit it), or use: rehearse local');
    return run(`cd ${R} && MC3_IMAGE='${process.env.MC3_IMAGE}' bash release/rehearse.sh`, 55 * 60_000);
  },
  async end() {
    await withLab(async (page, base) => {
      await sh(page, base, "pkill -f '[s]ourcebound'; pkill -f '[r]un_eval.py'; pkill -f '[r]un_suite.py'; true");
      await killKernels(page, base);
    }).catch((e) => console.log('no running pod or cleanup skipped:', e.message));
    const { context, page } = await openBrowser();
    try { await stop(page); } catch (e) { console.log('stop:', e.message); } finally { await context.close(); }
  },
};

if (!cmds[cmd]) { console.log(fs.readFileSync(__filename, 'utf8').split('\n').slice(1, 15).join('\n')); process.exit(2); }
if (['tail', 'pull'].includes(cmd) && !a) { console.error(`${cmd}: missing tag`); process.exit(2); }
Promise.resolve().then(() => cmds[cmd]()).catch((e) => { console.error('ERROR:', e.message); process.exit(1); });
````

- [ ] **Step 3: Syntax check and commit**

Run: `node --check tools/amd-gpu/remote-mc3.js && bash -n eval_mc3/worker_ctl.sh && python -m py_compile eval_mc3/gates.py && node tools/amd-gpu/remote-mc3.js`
Expected: the last command prints the usage header (lines 2–14) and exits 2, without opening a browser.

```bash
git add tools/amd-gpu/remote-mc3.js eval_mc3/gates.py eval_mc3/worker_ctl.sh
git commit -m "feat(mc3): remote GPU workflow and session gates" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 15: GPU session 1: gates, dependency lock, models, kit smoke (E0 + E1)

Spec: Testing Decisions E0–E1; Further Notes G2–G4.

- [ ] **Step 1: Quota and sync**

Run: `node tools/amd-gpu/gpu.js status` (record `quota_remaining_seconds`), then `node tools/amd-gpu/remote-mc3.js sync`
Expected: the remote `ls` lists `eval_mc3  mc3  release  sourcebound  tests`.

- [ ] **Step 2: Resolve and freeze the dependencies on the base image**

Run: `node tools/amd-gpu/remote-mc3.js setup`
Expected: a line `torch 2.13.0+rocm10.0.0 /opt/venv/...` (**must** contain `rocm` and `/opt/venv`), then the lock printed and `pulled mc3/requirements.lock`. If the assert fails, a dependency pulled a torch build. Delete it from `/workspace/pylib-mc3` and add it to the purge list in `setup`. Never install torch.

- [ ] **Step 3: Models.** The development reader is the MC2 checkpoint, already on `/workspace`. The embedder is downloaded.

Run: `node tools/amd-gpu/gpu.js sh "ls /workspace/models; ln -sfn /workspace/models/Qwen3-VL-4B-Instruct /workspace/models/mc3-reader && ls -l /workspace/models"`
If `Qwen3-VL-4B-Instruct` is missing: `node tools/amd-gpu/remote-mc3.js model Qwen/Qwen3-VL-4B-Instruct reader`
Then: `node tools/amd-gpu/remote-mc3.js model Qwen/Qwen3-Embedding-0.6B embedder`
Expected: `du` reports about 1.2G for the embedder, and `df` shows ≥ 10 GB free on `/workspace`.

- [ ] **Step 4: Gates**

Run: `node tools/amd-gpu/remote-mc3.js gates`
Expected: `results/mc3-gates-<ts>.json` with:
- `torch_is_rocm: true`;
- `gfx: "gfx1100"`;
- `fitz_version` and `openpyxl_version` not `IMPORT FAILED`;
- `root_reads_mode_000: true`, which is expected for root.

Record two values, which decide later steps:
- `dac_drop_blocks_read`: if true, use `worker-start nodac` everywhere; if false, the unreadable-file questions are only proven by CI.
- `unshare_n`.

- [ ] **Step 5: Start the worker and run the kit (E1 smoke)**

Run: `node tools/amd-gpu/remote-mc3.js worker-start nodac` (or without `nodac` if the gate said false)
Expected: a ready JSON such as `{"pid": ..., "startup_s": ...}`. Record `startup_s`: model load plus warm-up, which belongs to the 600 s budget.

Run: `node tools/amd-gpu/remote-mc3.js suite eval_mc3/kit kit-q3vl4b-e1`, then repeat `node tools/amd-gpu/remote-mc3.js tail kit-q3vl4b-e1` every 1–2 min until `summary:` shows the aggregate.
Expected: `strict` is ideally 10/10, with `violations: 0`, `max_s` < 25 and `max_index_s` < 300.

- [ ] **Step 6: Pull, record, end the session**

Run: `node tools/amd-gpu/remote-mc3.js pull kit-q3vl4b-e1 && node tools/amd-gpu/remote-mc3.js end`

Write `results/mc3-session1.md`:
- the gates JSON fields above;
- the lock versions;
- `startup_s`, the summary, and per-question `kind`;
- for every non-strict question, the stage at fault from `results/kit-q3vl4b-e1.diag.jsonl`: `ranked`, `bridges`, `attempts[].raw`, `attempts[].verdict`, `warrant`;
- quota at start and end.

**Exit criterion (spec E1):** 10/10, **or** a written failure list with the stage at fault.

- [ ] **Step 7: Commit.** `results/*` is gitignored except `*.md`, so the gates JSON needs `-f`:

```bash
git add mc3/requirements.lock results/mc3-session1.md && git add -f results/mc3-gates-*.json
git commit -m "docs(mc3): GPU session 1 gates, dependency lock and kit smoke" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 16: Development and holdout corpora; E1 baseline on development

Spec: Testing Decisions (evaluation corpora; adoption rule).

- [ ] **Step 1: Generate locally** (seeds are fixed so results are comparable; the holdout stays locked)

```bash
python eval_mc3/generate_corpus.py --out eval_mc3/data/dev --seeds 101 102 103 104
python eval_mc3/generate_corpus.py --out eval_mc3/data/holdout --seeds 901 902
du -sh eval_mc3/data/dev eval_mc3/data/holdout
```

Expected: four and two lines of `{"seed": ..., "questions": 16}`, and each split well under 40 MB. Open two generated images (`specs/*_pinout.png`, `support/asset_label.jpg`) and confirm the text is legible.

- [ ] **Step 2: Upload development data only** (the holdout is uploaded in Task 20)

Run: `node tools/amd-gpu/remote-mc3.js data eval_mc3/data/dev`

- [ ] **Step 3: Baseline (dev reader, all defaults)**

Run: `node tools/amd-gpu/remote-mc3.js worker-start nodac && node tools/amd-gpu/remote-mc3.js suite eval_mc3/data/dev dev-q3vl4b-e1`. Tail until the aggregate appears, then `pull dev-q3vl4b-e1`.

Expected: a summary with about 64 questions, `violations: 0`, and `max_s` < 25. Record the strict score, `kinds` (wrong_answer / over_citation / under_citation / false_refusal / false_answer) and per-corpus index times in `results/mc3-dev.md`.

- [ ] **Step 4: Failure analysis.** For each non-strict question, classify the stage from the diag log:
- `retrieval`: the expected file is not in `ranked` or `bridges`;
- `reader`: the expected file was in the pack, but `raw` has the wrong value or quotes;
- `gate`: a correct `raw` was rejected; see `verdict`;
- `warrant`: a correct answer got NO, or a decoy got YES;
- `transcription`: the image transcript in the index is wrong.

Put the table in `results/mc3-dev.md`.

### Task 17: E2 (dense retrieval) and E3 (bridge hop) ablations

Spec: Testing Decisions E2–E3. The adoption rule: keep a change only if it fixes more strict questions than it breaks, and costs no category more than one question.

- [ ] **Step 1: BM25 only.** Start the worker with dense retrieval disabled. `ENV` is shared, so prefix the variable:

`node tools/amd-gpu/gpu.js sh "cd /workspace/mc3 && export SB_DENSE=0 PYTHONPATH=/workspace/pylib-mc3:/workspace/mc3 SB_READER=/workspace/models/mc3-reader SB_EMBEDDER=/workspace/models/mc3-embedder SB_INDEX_DIR=/tmp/sb-index HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && bash eval_mc3/worker_ctl.sh start nodac"`

Expected: the ready JSON. Drop `nodac` if Task 15's gate reported `dac_drop_blocks_read: false`.

Then `suite eval_mc3/data/dev dev-q3vl4b-e2-bm25only`, tail, pull.

- [ ] **Step 2: No bridge hop.** Same as Step 1 with `SB_BRIDGE=0` (and `SB_DENSE=1`), tag `dev-q3vl4b-e3-nobridge`.

- [ ] **Step 3: Decide and record.** In `results/mc3-dev.md`, compare the three runs per category:
- E2: keep dense if it wins;
- E3: the bridge must win on both chain categories and must not raise `over_citation`.

If a component loses, change its default in code (the `SB_DENSE` / `SB_BRIDGE` default), with a test showing the new default, and commit.

- [ ] **Step 4: End the session.** Run `node tools/amd-gpu/remote-mc3.js end`, then commit the notes.

```bash
git add results/mc3-dev.md
git commit -m "docs(mc3): E1-E3 development results" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 18: E4: release reader sweep (Qwen3-VL-8B vs Qwen3.5-9B vs 4B)

Spec: §8, Testing Decisions E4, Further Notes D. Keep one large checkpoint at a time. Large checkpoints go to the ephemeral root (`tmp`), about 4 min each at about 90 MB/s.

- [ ] **Step 1: Qwen3-VL-8B-Instruct**

Run: `node tools/amd-gpu/remote-mc3.js model Qwen/Qwen3-VL-8B-Instruct reader tmp`, then `worker-start nodac` (record `startup_s`), then `suite eval_mc3/data/dev dev-q3vl8b-e4` and `suite eval_mc3/kit kit-q3vl8b-e4`.
Expected: `startup_s` < 180, `max_s` < 25, and peak VRAM < 40 GiB.

Each candidate does its own OCR, so OCR quality is part of the comparison. Two mechanisms make sure of that:
- `worker_ctl.sh start` deletes `/tmp/sb-index`, including the transcript cache;
- the cache key is `<engine name>:<sha1>`, where the engine name is the **resolved** model directory (for example `Qwen3-VL-8B-Instruct`), not the `mc3-reader` symlink.

- [ ] **Step 2: Qwen3.5-9B** (thinking is disabled by `enable_thinking=False` in `engine_qwen.py`)

Run: `node tools/amd-gpu/remote-mc3.js model Qwen/Qwen3.5-9B reader tmp`, then **`node tools/amd-gpu/remote-mc3.js worker-start nodac`**, then the same two suites with tags `dev-q35-9b-e4` and `kit-q35-9b-e4`. The restart is required: re-pointing the `mc3-reader` symlink does not reload a running worker, so without it the 8B would be evaluated again under the 9B tags.
If loading fails because `AutoModelForImageTextToText` does not map the architecture in transformers 5.17, record the error and skip this candidate. Do not upgrade transformers mid-experiment.

- [ ] **Step 3: Optional `derived` answers.** Rerun the winner with `SB_ALLOW_DERIVED=1` (prefixed as in Task 17). Adopt only by the adoption rule.

- [ ] **Step 4: Choose the release reader.** Choose by strict score, then by worst-case latency. Record the exact Hugging Face commit hash, from `huggingface_hub.model_info(repo).sha`, for both the reader and the embedder. Task 21 needs them. Write `results/mc3-e4.md`, end the session and commit.

### Task 19: Fix what the failure analysis found (TDD, no GPU)

Spec: Implementation Decisions §5–§7; adoption rule.

For each recurring failure class in `results/mc3-dev.md`:

- [ ] **Step 1: Turn it into a failing local test first.** Copy the reader's actual `raw` reply from the diag log into a FakeEngine script against a generated corpus. Use `tests/test_sb_gates_pipeline.py` and `tests/test_sb_generated.py` as templates.
- [ ] **Step 2: Fix it in the narrowest module.**
  - A prompt wording change goes in `reader.INSTRUCTIONS`.
  - A gate rule goes in `gates.judge`.
  - Tokenization goes in `terms.py`.

  Never add a rule keyed to a sample or generated value, file name or question wording.
- [ ] **Step 3: Run `python -m pytest -q tests/test_sb_*.py`.** Everything must pass. Commit with a message naming the failure class.
- [ ] **Step 4: Re-verify on the GPU** in the next session. Rerun `suite eval_mc3/data/dev` with a new tag and apply the adoption rule. If the fix does not help on the GPU, revert it.

Experiments marked E5 in the spec (reranker, ablation check, whole-corpus mode) are **not** in this plan. Start a follow-up plan for one only if this analysis shows a failure class that it addresses and Tasks 17–19 could not fix.

### Task 20: Freeze, holdout once, scale timing

Spec: Testing Decisions E6; Acceptance checks (startup, per question, whole run, VRAM).

- [ ] **Step 1: Freeze.** Tag the commit (`git tag mc3-freeze-1`). From here, configuration changes require a new tag and a new holdout.

- [ ] **Step 2: Holdout, once.** This is a new session, so the ephemeral `/root/models` copy is gone. Run `node tools/amd-gpu/remote-mc3.js model <winner repo> reader tmp` again, then `node tools/amd-gpu/remote-mc3.js data eval_mc3/data/holdout`, then `worker-start nodac`, then `suite eval_mc3/data/holdout holdout-final`, tail and pull. Record the strict score and the failure analysis in `results/mc3-holdout.md`. **Do not tune on it.**

- [ ] **Step 3: Scale timing corpus** (about 400 files, built from 25 generated corpora under one root):

```bash
python eval_mc3/generate_corpus.py --out eval_mc3/data/scale_src --seeds $(seq 2001 2025)
python - <<'EOF'
import shutil, pathlib, json
src, dst = pathlib.Path("eval_mc3/data/scale_src"), pathlib.Path("eval_mc3/data/scale/c1")
(dst / "corpus").mkdir(parents=True, exist_ok=True)
for d in sorted(src.iterdir()):
    shutil.copytree(d / "corpus", dst / "corpus" / d.name)
q = json.loads((src / "c2001" / "questions.json").read_text())
for x in q["queries"]:
    x["expected_citations"] = [f"c2001/{c}" for c in x["expected_citations"]]
(dst / "questions.json").write_text(json.dumps(q))
(dst / "hazards.json").write_text(json.dumps({"chmod000": [f"{d.name}/vendor/internal_audit.txt" for d in src.iterdir()], "empty_dirs": []}))
EOF
du -sh eval_mc3/data/scale
```

Upload with `data eval_mc3/data/scale` (split into several `data` calls if over 40 MB). Then run `worker-start nodac` **immediately** before `suite eval_mc3/data/scale scale-final`. The first index after a worker start gets only what is left of the 540 s startup budget, exactly as in grading, so a worker left over from Step 2 would time the wrong thing.

Expected:
- `max_index_s` + `startup_s` < 480 s;
- `max_s` < 25 s;
- peak VRAM < 44 GiB;
- in the summary's `corpora[0].index_stats`, `transcribed` = 50 (2 images × 25 corpora). Fewer means transcription ran out of budget, and the timing is not valid. **Read only the timing fields.** Accuracy on this corpus is not meaningful, because 25 companies share the same question shapes ("the production log" is ambiguous here).

If indexing is too slow, raise `SB_OCR_BATCH` (measured in Task 15) or `SB_PARSE_WORKERS`, then rerun.

- [ ] **Step 4: Record, end the session, commit**

```bash
git add results/mc3-holdout.md results/mc3-scale.md
git commit -m "docs(mc3): frozen configuration, holdout and scale timing" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 21: Publish, check and rehearse the release image

Spec: §10, Acceptance checks, user stories 56–60.

- [ ] **Step 1: Secrets** (the user does this once, in the GitHub repository settings):
- `MC3_IMAGE` = `docker.io/<user>/<private-sounding-name>`, with no tag;
- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`, which already exist for ROADREAD.

Confirm with the user. Never write the value anywhere in the repository.

- [ ] **Step 2: Measure the base once (gate G1).** Run on any machine with crane:

```bash
python - <<'EOF'
import json, subprocess, zlib
B="docker.io/rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0"
L=json.loads(subprocess.run(["crane","manifest","--platform","linux/amd64",B],capture_output=True,text=True,check=True).stdout)["layers"]
n=0
for l in L:
    p=subprocess.Popen(["crane","blob",f"docker.io/rocm/pytorch@{l['digest']}"],stdout=subprocess.PIPE); d=zlib.decompressobj(16+zlib.MAX_WBITS)
    for c in iter(lambda:p.stdout.read(1<<20),b""): n+=len(d.decompress(c))
print(n, round(n/2**30,3), "GiB")
EOF
```

Record the byte count in `results/mc3-release.md`. The release budget is weights ≤ 60 GiB − base − 2 GiB margin − deps and app (about 1.5 GiB).

- [ ] **Step 3: Run the release workflow** with the pinned revisions from Task 18:

`gh workflow run mc3-release.yml -f tag=rc1 -f reader_repo=<winner> -f reader_rev=<sha> -f embedder_repo=Qwen/Qwen3-Embedding-0.6B -f embedder_rev=<sha>`, then `gh run watch`.

Expected: `published tag rc1`, then `check_image.py` prints `"base_prefix_ok": true`, `"size_ok": true`, `"entrypoint_ok": true` (no ENTRYPOINT; `Cmd` is the supervisor), `"env_ok": true`, `"no_secrets": true` and `"ok": true`.

If an upload times out, lower `LAYER_MAX` to 2 GiB. If the size fails, the reader is too large for the measured base, so fall back to the next E4 candidate.

- [ ] **Step 4: Rehearse the exact pushed image on the notebook**

```bash
export MC3_IMAGE='docker.io/<user>/<name>:rc1'   # your shell only
node tools/amd-gpu/remote-mc3.js sync && node tools/amd-gpu/remote-mc3.js rehearse
```

Expected:
- `torch 2.13.0+rocm...`;
- a kit summary with `"strict": 10` (or the session-1 score if it was not 10), `"violations": 0`, and `index_s` plus the reported container-start time under 480 s;
- the liveness query prints `{"answer": "E7731", ...}`, then `supervisor alive`.

If `rehearse` exits 2 with `registry unreachable from the pod`, the pod cannot pull from Docker Hub (gate G5). In a new session, first run `node tools/amd-gpu/remote-mc3.js model <winner repo> reader tmp` again (the ephemeral copy is gone and `/models/reader` would point nowhere). Then run `node tools/amd-gpu/remote-mc3.js rehearse local`. It rebuilds the app and deps layers on the pod from the same commit and links `/models` to the notebook's models. Record that the rehearsal used local layers; the pushed image's integrity then rests on `check_image.py`.

Run `node tools/amd-gpu/remote-mc3.js end`. Write `results/mc3-release.md` with the check JSON, the rehearsal output (keep the image reference out), and quota.

- [ ] **Step 5: Optional, strongly recommended.** If any Linux host with Docker and about 70 GB free is available, run the official self-check there:

`python3 selfcheck.py "$MC3_IMAGE" eval_mc3/kit/mc3-corpus`, plus a manual `docker run --cap-drop DAC_OVERRIDE --network none ...`.

- [ ] **Step 6: Commit the release notes.** Submitting the image reference on the lablab.ai MC3 form is done **by the user**.

```bash
git add results/mc3-release.md
git commit -m "docs(mc3): release rc1 checks and notebook rehearsal" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 22: Records

- [ ] **Step 1:** Add a "Mini-Challenge 3 (SOURCEBOUND)" section to `docs/STATUS.md` covering: the spec and plan links; kit, development and holdout strict scores; the release checks; and what remains (submission and deadline gate G6).
- [ ] **Step 2:** In `README.md`, add one line that links the spec. **No image reference.**
- [ ] **Step 3:** Commit.

```bash
git add docs/STATUS.md README.md
git commit -m "docs(mc3): record SOURCEBOUND status" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Self-review against the spec

| Spec item | Where |
|---|---|
| §2 process model, two invocations, self-start, refusal on any error, atomic write, stale removal | Task 9 (`mc3/app.py`, `worker.py`, `supervisor.py`), tests `test_worker_absent_writes_valid_refusal_fast`, `test_query_id_is_used_verbatim`, `test_bad_request_does_not_kill_worker` |
| §3 defensive walk, magic bytes, encryption (incl. owner-password PDFs, OLE, zip flags), per-file timeout | Tasks 4–6 |
| §4 parsing per type, headers on every row, PDF page OCR, DOCX media, status and families | Tasks 3, 5, 7 (`build_index` transcription phase) |
| §5 tokenizer, BM25 + dense RRF, demotion, bridge hop, reader-requested hop | Tasks 3, 7, 8 (`need_lookup` test) |
| §6 evidence pack, image attachment, prompt rules, reply schema | Task 8 |
| §7 grounding, answer-in-evidence (segment, not file), supersession, warrant, necessity, implied link | Task 8; contract kit 10/10 in Task 9; oracle on generated corpora in Task 10 |
| §8 models, BF16, offline, dependency rules, VRAM | Tasks 11, 14, 15, 18 |
| §9 time budget | Tasks 15, 18, 20 (startup_s, max_index_s, max_s) |
| §10 crane release, ≤ 4 GiB layers, secrets, rehearsal, reference Dockerfile | Tasks 12, 13, 21 |
| Testing: highest seam, FakeEngine, CI under the grader's flags, generator, metrics, E0–E4, E6 | Tasks 9, 10, 13, 15–20 |
| E5 (reranker, ablation, whole-corpus) | Deliberately deferred to a follow-up plan if Task 19's analysis calls for it |
| Further Notes G1–G7 | G1 Task 21 Step 2; G2–G4 Task 15; G5 Task 21 Step 3; G6 Task 22; G7 `score.py` ignores aliases |
