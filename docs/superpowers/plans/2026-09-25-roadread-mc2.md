# ROADREAD (Mini-Challenge 2) Implementation Plan — revision 2

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ROADREAD plate and sign OCR submission specified in `docs/MINI_CHALLENGE_2_SPEC.md` (the "spec"). It has four parts:
- a thin `/app/app.py` client that never imports torch
- a resident GPU worker running Qwen3-VL or Qwen3.5
- narrow domain rules
- a harness-faithful exact-match evaluation

All GPU work runs on the AMD notebook and is driven from this Windows machine with the Playwright tooling in `tools/amd-gpu/`, which `GPUWEBSKILL.md` documents.

**Architecture:** Everything that can be tested without a GPU is written test-first and tested locally with pytest and a fake engine:
- normalization
- rules
- decoding and views
- the IPC protocol
- the thin client
- selection and escalation
- the evaluation runner

The GPU-dependent code is one engine module plus the worker. `tools/amd-gpu/remote.js` wraps `gpu.js` and does the remote work:
- syncs the code to `/workspace/roadread`
- installs pinned dependencies into `/workspace/pylib` without shadowing the ROCm torch
- downloads weights into `/workspace/models`
- runs the gates, the worker and the evaluations
- pulls results back to `results/`
- turns the pod off

Docker packaging comes last and runs on a Linux builder, because the notebook has no Docker.

**Tech Stack:**
- GPU side: Python 3.14, torch 2.13.0+rocm10.0.0 (preinstalled, never reinstalled), transformers 5.x (pinned in Task 11)
- Local: Python 3.10+ (tested with 3.12), Pillow, pytest, PyMuPDF (sample extraction only)
- Remote driver: Node 22 + Playwright 1.63 (`tools/amd-gpu`)

---

## Revision 2: review findings

Revision 1's code was extracted and run: all 24 of its tests passed. The review then probed edge cases, and each bug below was confirmed by running it, not assumed. Every item is fixed in this revision and pinned by a new test.

| # | Severity | Finding (rev 1) | Fix (rev 2) |
|---|---|---|---|
| 1 | **Critical (quota)** | `remote.js end/tail/pull` call `withLab`, which **launches a pod** if none is running, so ending the day could start a new 3 h clock | read-only commands set `AUTO_LAUNCH=0` (Task 10) |
| 2 | **Critical (correctness)** | `pip install --target /workspace/pylib` with `accelerate` in requirements would pull a PyPI/CUDA **torch into pylib and shadow the ROCm torch** (pylib comes first on PYTHONPATH) | requirements list only `transformers`; after install, shadowing packages are purged and `'rocm' in torch.__version__` is asserted (Task 10 `setup`) |
| 3 | **Critical (Windows)** | `remote.js sync` ran `tar -czf C:/…`; the `tar` on PATH is Git's GNU tar, which parses `C:` as a remote host (**reproduced**: "Cannot connect to C: resolve failed") | tar with a path relative to the repo; `C:\Windows\System32\tar.exe` (bsdtar) on win32 |
| 4 | High | A Chinese read that the model labelled `KIND: sign` skipped I/O mapping and the format check (**reproduced**) | rules route by content: a Chinese-registration shape is always treated as a plate |
| 5 | High | Separator look-alikes `•` `・` `‧` and full-width ASCII `ＳＴＯＰ` survive, but evaluator normalization only removes `·` (**reproduced**: `京A•12345` ≠ `京A12345`) | canonicalize look-alikes to `·` and full-width ASCII to ASCII before the rules |
| 6 | High | The reread used `bounded(image, 2.5 MP)`, which is **identical** to the first view for any image ≤ 1 MP (all ten samples are ≤ 0.47 MP), so a greedy reread repeats the same answer | `reread_view()`: more original pixels if the first view was downscaled, otherwise a 2× upscale |
| 7 | High | Special tokens (`<|im_end|>`) leaked into the answer (**reproduced**) | `parse_reply` strips `<|…|>` and code fences |
| 8 | Medium | NY slogan **"EXCELSIOR"** (printed on sample 3) missing from the banner list | added, with a test built from sample 3's text |
| 9 | Medium | `run_eval.py` crashed on a timeout or a missing output file instead of recording a failure | both recorded as failures; `violations` count in the summary |
| 10 | Medium | No measurement of client overhead (spec acceptance: < 1 s) | `ROADREAD_DIAG_LOG`; `run_eval` reports `max_overhead_s` |
| 11 | Medium | `worker-start` did not stop an existing worker (port in use → the new worker dies, ready file never appears) | `pkill` first, and the worker log is tailed on failure |
| 12 | Medium | Engine used one-step `apply_chat_template(tokenize=True)` with a PIL image, which varies across transformers versions | two-step: template text → `processor(text, images)` |
| 13 | Low | Worker `except` referenced `req` before assignment | initialized to `None` |
| 14 | Low | The gates HF test file name was guessed | uses `Qwen/Qwen2.5-0.5B/model.safetensors`, measured at ~100 MB/s on 2026-09-24 |
| 15 | Low | Spec's cross-view disagreement trigger (§6) had no implementation path | optional `ROADREAD_XVIEW=1`, evaluated in Task 13 |
| 16 | Low | Sample extraction was vague | exact PyMuPDF script; verified that the PDF holds exactly 10 images in sample order |

---

## Before you start (read once)

1. Read `GPUWEBSKILL.md` §0, §3, §5.3 and §8. You need `node tools/amd-gpu/gpu.js status` to work; if it fails with `NOT_SIGNED_IN`, ask the user to run `! cd tools/amd-gpu && node login.js`. **Never type the password yourself.**
2. Facts measured on 2026-09-24 that this plan relies on:
   - GPU: Radeon Pro W7900D, `gfx1100` (RDNA3), 48 GiB. So **BF16/FP16 only, no FP8.**
   - Storage: `/workspace` is persistent, 25 GB, and is the Jupyter root; paths in `gpu.js put/get` are relative to it. `/opt/venv` is reset every session.
   - Environment: `HF_HOME=/workspace/.cache/huggingface` and `HF_ENDPOINT=https://hf-mirror.com` are preset.
   - Launch takes ~80 s, and every `gpu.js` or `remote.js` call has ~10 s of browser overhead.
3. **Quota rule:** 3 h of pod time per day, counted while the pod exists, including idle time. **Every GPU task ends with `node tools/amd-gpu/remote.js end`.** Plan at most 150 minutes per day. Record the `quota_remaining_seconds` from `gpu.js status` at the start and end of each GPU task in its results note.
4. **Hard contract (spec §A):**
   - Invocation: `python3 /app/app.py --input-image /app/input/image_01.png`
   - Output: `/app/output/image_01_output.json` = `{"text": str, "confidence"?: float∈[0,1]}`
   - Time limits: 30 s per image, 600 s startup, 600 s total for ten images
   - GPU memory: 1–48 GiB, sampled every 3 s
   - Base image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`, unpacked ≤ 64,424,509,440 bytes
   - Scoring: normalization = uppercase, remove whitespace and `- . · _`
5. Run commands from the repo root `H:\augsepthacks\amd`. The Bash tool is Git Bash; PowerShell also works for `python` and `node`.

## Time and quota budget

| Task | Where | Pod minutes (estimate) |
|---|---|---|
| 1–10 | local | 0 |
| 11 gates + setup + 4B download | GPU | 35–50 |
| 12 smoke (10 samples) | GPU | 15–20 |
| 13 dev baseline (120 images at ≤ 25 s) | GPU | 45–60 per configuration |
| 13 challenger / 9B / x-view | GPU | 45–60 each, on separate days |
| 13 holdout (once) | GPU | 45–60 |
| 14 container | Linux builder | 0 notebook |

---

## File structure

```
roadread/                     # importable package, copied to /app/roadread in the image
  __init__.py
  normalize.py                # evaluator normalization; used for scoring/comparison, never to rewrite output
  rules.py                    # canonicalization, Chinese format check, serial I/O mapping, US banner filter
  decode.py                   # PNG/JPEG/TIFF open, EXIF transpose, first frame, RGB, bounded/reread/crop views
  prompts.py                  # the frozen instruction + PROMPT_VERSION
  protocol.py                 # newline-JSON over 127.0.0.1 TCP; stdlib only (the client imports it)
  pipeline.py                 # parse reply, rules, optional cross-view check, ≤1 escalation, selection
  engine_fake.py              # Read dataclass + deterministic FakeEngine (tests, worker smoke)
  engine_qwen.py              # transformers Qwen3-VL/Qwen3.5 engine (GPU only)
  worker.py                   # resident worker: load, warm up, ready file, serve sequentially
app/
  app.py                      # evaluator entry: thin client; stdlib + roadread.protocol only
  requirements.txt            # candidate deps (transformers only)
  requirements.lock           # frozen by Task 11 from the GPU box (pip freeze --path /workspace/pylib)
eval/
  extract_samples.py          # PDF → eval/samples/*.{png,jpg,tiff} + samples.jsonl
  samples/                    # 10 brief images in their stated formats + samples.jsonl (smoke only; committed)
  run_eval.py                 # fresh app.py process per image, exact-match by slice, latency, overhead
  gates.py                    # session-1 gates → one JSON line
  data/                       # dev/holdout sets (gitignored)
tests/                        # pytest, CPU only
tools/amd-gpu/remote.js       # Playwright-driven remote workflow
docker/Dockerfile  docker/entrypoint.sh
results/                      # pulled GPU results (gitignored except *.md notes)
pyproject.toml
```

---

### Task 1: Scaffold the Python project

**Files:** Create `pyproject.toml`, `roadread/__init__.py`, `tests/__init__.py`, and `app/requirements.txt`, and modify `.gitignore`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "roadread"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["pillow>=10"]

[project.optional-dependencies]
dev = ["pytest>=8", "pymupdf>=1.24"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Write `app/requirements.txt`**

```
# Candidate only. Task 11 installs this on the GPU box and freezes exact versions into requirements.lock.
# NEVER add torch, torchvision, triton, numpy or accelerate: `pip install --target` would pull a
# non-ROCm build into /workspace/pylib and shadow the preinstalled ROCm torch.
transformers>=5.17,<6
```

- [ ] **Step 3:** Create empty `roadread/__init__.py` and `tests/__init__.py`, and append to `.gitignore`:

```
results/*
!results/*.md
eval/data/
*.egg-info/
```

- [ ] **Step 4:** `python -m pip install -e ".[dev]"` then `python -m pytest -q`. Expect `no tests ran` with exit code 5, which is fine here.
- [ ] **Step 5:** Commit with `git add pyproject.toml roadread tests app/requirements.txt .gitignore && git commit -m "chore: scaffold roadread package"`

---

### Task 2: Evaluator normalization

**Files:** Create `roadread/normalize.py` and `tests/test_normalize.py`.

- [ ] **Step 1: Write the failing test**

```python
from roadread.normalize import normalize, matches

def test_normalize_rules_from_brief():
    assert normalize("京A·12345") == "京A12345"
    assert normalize("speed limit 65") == "SPEEDLIMIT65"
    assert normalize("7-AB.C_1 23") == "7ABC123"

def test_normalize_keeps_other_punctuation_and_cjk():
    assert normalize("A/B!") == "A/B!"          # only - . · _ and whitespace are removed
    assert normalize("沪b·88888") == "沪B88888"
    assert normalize("京A•12345") == "京A•12345"  # look-alikes are NOT removed by the evaluator (see Task 3)

def test_matches():
    assert matches("ROAD WORK AHEAD", "road work ahead")
    assert matches("JHT2951", "JHT 2951")
    assert not matches("7ABC123", "7ABC128")
```

- [ ] **Step 2:** `python -m pytest tests/test_normalize.py -q` should fail with `ModuleNotFoundError: roadread.normalize`.
- [ ] **Step 3: Implement**

```python
"""Evaluator normalization (brief p.7). Used only for scoring/comparison, never to rewrite predictions."""
import re

_DROP = re.compile(r"[\s\-\.·_]")

def normalize(text: str) -> str:
    return _DROP.sub("", text.upper())

def matches(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)
```

- [ ] **Step 4:** `python -m pytest tests/test_normalize.py -q` should show `3 passed`.
- [ ] **Step 5:** Commit with `git add roadread/normalize.py tests/test_normalize.py && git commit -m "feat: evaluator normalization"`

---

### Task 3: Domain rules (spec §5)

**Files:** Create `roadread/rules.py` and `tests/test_rules.py`.

What the rules do, and must not do:
- **Canonicalize first:** full-width ASCII becomes ASCII, dot look-alikes become `·`, and whitespace collapses.
- **Chinese path:** triggered by content, not by KIND. It maps I→1 and O→0 in the serial only, and flags a malformed read as suspect without editing it.
- **US banner filter:** plate path only. It removes whole phrases and keeps the text when nothing plausible would remain.
- **Never** collapse repeated characters, pad, truncate, or map O→0 or B→8 globally.

- [ ] **Step 1: Write the failing tests**

```python
import random
from roadread.rules import postprocess, is_chinese_plate, chinese_format_ok

def test_repeated_digits_never_collapsed():
    for s in ["沪B·88888", "京A·11111", "粤B·D88888"]:
        assert postprocess(s, kind="plate").text == s

def test_chinese_length_preserved_property():
    rnd = random.Random(0)
    for _ in range(500):
        s = rnd.choice("京沪粤川") + rnd.choice("ABCDE") + "·" + "".join(rnd.choice("0123456789ABDF8") for _ in range(rnd.choice([4, 5, 6, 7])))
        assert len(postprocess(s, kind="plate").text) == len(s)

def test_chinese_format_check_flags_not_edits():
    r = postprocess("沪B·888888", kind="plate")          # 6-char serial without D/F -> suspect, unchanged
    assert r.text == "沪B·888888" and r.suspect
    assert not postprocess("京A·12345", kind="plate").suspect
    assert not postprocess("粤B·12345F", kind="plate").suspect   # large new-energy: D/F at the end

def test_chinese_routed_by_content_even_if_kind_sign():
    assert postprocess("京A·1O2I5", kind="sign").text == "京A·10215"

def test_io_mapping_only_inside_chinese_serial():
    assert postprocess("京A·1O2I5", kind="plate").text == "京A·10215"
    assert postprocess("京I·12345", kind="plate").text == "京I·12345"   # province letter untouched
    assert postprocess("OIL AHEAD", kind="sign").text == "OIL AHEAD"
    assert postprocess("7OIL123", kind="plate").text == "7OIL123"       # US plate untouched

def test_canonicalizes_separators_case_and_fullwidth():
    assert postprocess("京A•12345", kind="plate").text == "京A·12345"
    assert postprocess("京a・12345", kind="plate").text == "京A·12345"
    assert postprocess("ＳＴＯＰ", kind="sign").text == "STOP"

def test_banner_filter_plate_only_and_keeps_vanity():
    assert postprocess("CALIFORNIA 7ABC123", kind="plate").text == "7ABC123"
    assert postprocess("NEW YORK JHT 2951 EXCELSIOR", kind="plate").text == "JHT 2951"   # sample 3
    assert postprocess("7ABC123 THE LONE STAR STATE", kind="plate").text == "7ABC123"
    assert postprocess("TEXAS", kind="plate").text == "TEXAS"            # nothing plausible left -> keep
    assert postprocess("STOP", kind="sign").text == "STOP"
    assert postprocess("WASHINGTON ST", kind="sign").text == "WASHINGTON ST"

def test_helpers():
    assert is_chinese_plate("京A·12345") and not is_chinese_plate("7ABC123")
    assert chinese_format_ok("京A·12345") and chinese_format_ok("粤B·D12345")
    assert not chinese_format_ok("京A·1234")
```

- [ ] **Step 2:** `python -m pytest tests/test_rules.py -q` should fail with `ModuleNotFoundError`.
- [ ] **Step 3: Implement**

```python
"""Narrow, test-pinned transcription rules (spec §5). No step may shorten a run of identical characters."""
from dataclasses import dataclass
import re

PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领"
_CN = re.compile(rf"^([{PROVINCES}])([A-Z])([·.\- ]?)([A-Z0-9]+)(.*)$")
_DOTS = str.maketrans({"•": "·", "・": "·", "‧": "·", "∙": "·", "⋅": "·", "●": "·", "\u3000": " "})
_US_BANNERS = sorted({
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT", "DELAWARE", "FLORIDA",
    "GEORGIA", "HAWAII", "IDAHO", "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE",
    "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA",
    "NEVADA", "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA", "OHIO",
    "OKLAHOMA", "OREGON", "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA", "TENNESSEE", "TEXAS",
    "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WEST VIRGINIA", "WISCONSIN", "WYOMING", "DISTRICT OF COLUMBIA",
    "THE LONE STAR STATE", "EMPIRE STATE", "EXCELSIOR", "SUNSHINE STATE", "THE GOLDEN STATE", "LAND OF LINCOLN",
    "GARDEN STATE", "KEYSTONE STATE", "GREAT LAKES STATE", "PURE MICHIGAN", "GRAND CANYON STATE",
    "FAMOUS POTATOES", "LIVE FREE OR DIE", "FIRST IN FLIGHT", "THE OCEAN STATE", "DMV.CA.GOV",
}, key=len, reverse=True)

@dataclass
class Rule:
    text: str
    suspect: bool = False

def canonicalize(s: str) -> str:
    s = "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in s)   # full-width ASCII
    return " ".join(s.translate(_DOTS).split())

def is_chinese_plate(s: str) -> bool:
    return bool(_CN.match(s.strip()))

def chinese_format_ok(s: str) -> bool:
    m = _CN.match(s.strip())
    if not m or m.group(5):
        return False
    serial = m.group(4)
    return len(serial) == 5 or (len(serial) == 6 and (serial[0] in "DF" or serial[-1] in "DF"))

def _map_serial_io(s: str) -> str:
    m = _CN.match(s)
    serial = m.group(4).replace("I", "1").replace("O", "0")
    return f"{m.group(1)}{m.group(2)}{m.group(3)}{serial}{m.group(5)}"

def _strip_banners(s: str) -> str:
    out = f" {s.upper()} "
    for b in _US_BANNERS:
        out = re.sub(rf"(?<=\s){re.escape(b)}(?=\s)", " ", out)
    out = " ".join(out.split())
    plausible = bool(out) and (any(ch.isdigit() for ch in out) or re.fullmatch(r"[A-Z0-9]{2,8}", out))
    return out if plausible else s

def postprocess(raw: str, kind: str) -> Rule:
    text = canonicalize(raw)
    upper = text.upper()
    if is_chinese_plate(upper):                     # content wins over the model's KIND label
        text = _map_serial_io(upper)
        return Rule(text, suspect=not chinese_format_ok(text))
    if kind != "plate":
        return Rule(text)
    return Rule(_strip_banners(text))
```

- [ ] **Step 4:** `python -m pytest tests/test_rules.py -q` should show `8 passed`. If a test fails, fix the implementation, never the test. The tests encode spec §5.
- [ ] **Step 5:** Commit with `git add roadread/rules.py tests/test_rules.py && git commit -m "feat: plate/sign domain rules with repeat-safe tests"`

---

### Task 4: Image decoding and views (spec §4)

**Files:** Create `roadread/decode.py` and `tests/test_decode.py`.

- [ ] **Step 1: Write the failing tests**

```python
from PIL import Image
from roadread.decode import load_rgb, bounded, center_crop, reread_view

def _save(tmp_path, name, img, **kw):
    p = tmp_path / name; img.save(p, **kw); return p

def test_formats_and_modes(tmp_path):
    for name, img in [("a.png", Image.new("P", (40, 20))), ("b.jpg", Image.new("L", (40, 20))),
                      ("c.tiff", Image.new("RGBA", (40, 20))), ("d.tif", Image.new("I;16", (40, 20)))]:
        im = load_rgb(_save(tmp_path, name, img))
        assert im.mode == "RGB" and im.size == (40, 20)

def test_transparent_becomes_white(tmp_path):
    im = load_rgb(_save(tmp_path, "t.png", Image.new("RGBA", (4, 4), (0, 0, 0, 0))))
    assert im.getpixel((0, 0)) == (255, 255, 255)

def test_exif_orientation(tmp_path):
    img = Image.new("RGB", (40, 20)); ex = img.getexif(); ex[0x0112] = 6
    assert load_rgb(_save(tmp_path, "r.jpg", img, exif=ex)).size == (20, 40)

def test_multipage_tiff_first_frame(tmp_path):
    a, b = Image.new("RGB", (10, 10), "red"), Image.new("RGB", (30, 30), "blue")
    p = tmp_path / "m.tiff"; a.save(p, save_all=True, append_images=[b])
    assert load_rgb(p).size == (10, 10)

def test_bounded_preserves_aspect_and_never_upscales():
    out = bounded(Image.new("RGB", (4000, 1000)), max_pixels=1_000_000)
    assert out.width * out.height <= 1_000_000 and abs(out.width / out.height - 4.0) < 0.02
    assert bounded(Image.new("RGB", (100, 50)), 1_000_000).size == (100, 50)

def test_reread_view_differs_from_first_view():
    assert reread_view(Image.new("RGB", (300, 100)), 1_000_000, 2_500_000).size == (600, 200)   # small: 2x up
    v = reread_view(Image.new("RGB", (4000, 1000)), 1_000_000, 2_500_000)                       # big: more pixels
    assert 1_000_000 < v.width * v.height <= 2_500_000

def test_center_crop_fraction():
    assert center_crop(Image.new("RGB", (100, 100)), 0.8).size == (80, 80)
```

- [ ] **Step 2:** `python -m pytest tests/test_decode.py -q` should fail with `ModuleNotFoundError`.
- [ ] **Step 3: Implement**

```python
"""Decode without throwing away evidence: EXIF transpose, first TIFF frame, RGB, aspect-safe views."""
from pathlib import Path
from PIL import Image, ImageOps

def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        im.seek(0)                                   # multipage TIFF: first frame (spec §G provisional policy)
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            im = im.point(lambda v: v / 256).convert("L")
        if im.mode in ("RGBA", "LA", "P", "PA"):
            im = im.convert("RGBA")
            im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
        return im.convert("RGB")

def bounded(im: Image.Image, max_pixels: int) -> Image.Image:
    w, h = im.size
    if w * h <= max_pixels:
        return im
    s = (max_pixels / (w * h)) ** 0.5
    return im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)

def upscale(im: Image.Image, factor: float, max_pixels: int) -> Image.Image:
    w, h = im.size
    f = min(factor, (max_pixels / (w * h)) ** 0.5)
    return im if f <= 1 else im.resize((int(w * f), int(h * f)), Image.Resampling.LANCZOS)

def reread_view(im: Image.Image, first_pixels: int, reread_pixels: int) -> Image.Image:
    """A genuinely different, higher-detail view for the single escalation."""
    if im.width * im.height > first_pixels:
        return bounded(im, reread_pixels)            # we downscaled the first view: give back original pixels
    return upscale(im, 2.0, reread_pixels)           # already full-res: enlarge small characters

def center_crop(im: Image.Image, frac: float) -> Image.Image:
    w, h = im.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return im.crop((left, top, left + cw, top + ch))
```

- [ ] **Step 4:** `python -m pytest tests/test_decode.py -q` should show `7 passed`.
- [ ] **Step 5:** Commit with `git add roadread/decode.py tests/test_decode.py && git commit -m "feat: image decoding and reread views"`

---

### Task 5: IPC protocol and the thin evaluator client (spec §7)

**Files:** Create `roadread/protocol.py`, `app/app.py`, and `tests/test_protocol_app.py`.

The protocol is one TCP connection to `127.0.0.1:$ROADREAD_PORT` (default 47811) per request, carrying one UTF-8 JSON line each way:
- request: `{"id","image","deadline_s"}`
- response: `{"id","text","confidence","diag"}`

A response whose id doesn't match is treated as stale and rejected. `app.py` deletes any old output before it asks, writes atomically (tmp then `os.replace`), and always writes a valid file, with empty text if it has to fail.

- [ ] **Step 1: Write the failing tests**

```python
import json, os, socket, subprocess, sys, threading, time
from pathlib import Path
from roadread import protocol

def _stub_server(port, reply):
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port)); srv.listen()
    def loop():
        while True:
            c, _ = srv.accept()
            req = protocol.recv_line(c)
            protocol.send_line(c, {"id": req["id"], **reply}); c.close()
    threading.Thread(target=loop, daemon=True).start()
    return srv

def _run_app(tmp_path, port, name="image_01.png", extra_env=None):
    img = tmp_path / name; img.write_bytes(b"x")
    env = {**os.environ, "ROADREAD_PORT": str(port), "ROADREAD_OUTPUT_DIR": str(tmp_path / "out"),
           "PYTHONPATH": str(Path.cwd()), **(extra_env or {})}
    t = time.time()
    r = subprocess.run([sys.executable, "app/app.py", "--input-image", str(img)], env=env, capture_output=True, text=True, timeout=60)
    return r, time.time() - t, tmp_path / "out" / (Path(name).stem + "_output.json")

def test_app_writes_named_json_utf8(tmp_path):
    _stub_server(47901, {"text": "沪B·88888", "confidence": 0.9, "diag": {"elapsed_s": 0.1}})
    r, _, out = _run_app(tmp_path, 47901, "image_07.tiff")
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text(encoding="utf-8")) == {"text": "沪B·88888", "confidence": 0.9}

def test_app_clamps_confidence_and_logs_diag(tmp_path):
    _stub_server(47904, {"text": "STOP", "confidence": 7, "diag": {"elapsed_s": 0.2}})
    log = tmp_path / "diag.jsonl"
    r, _, out = _run_app(tmp_path, 47904, extra_env={"ROADREAD_DIAG_LOG": str(log)})
    assert json.loads(out.read_text(encoding="utf-8"))["confidence"] == 1.0
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert row["image"] == "image_01.png" and row["diag"]["elapsed_s"] == 0.2 and row["client_wall_s"] > 0

def test_app_removes_stale_output_and_writes_empty_when_worker_absent(tmp_path):
    stale = tmp_path / "out" / "image_01_output.json"; stale.parent.mkdir(parents=True); stale.write_text('{"text":"OLD"}')
    r, dt, out = _run_app(tmp_path, 47999)
    assert r.returncode == 0 and json.loads(out.read_text(encoding="utf-8"))["text"] == "" and dt < 10

def test_app_does_not_import_torch():
    code = "import sys; import app.app; print('torch' in sys.modules, 'transformers' in sys.modules, 'PIL' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path.cwd()).stdout
    assert out.strip() == "False False False"
```

- [ ] **Step 2:** `python -m pytest tests/test_protocol_app.py -q` should fail with `ModuleNotFoundError`.
- [ ] **Step 3: Implement `roadread/protocol.py`**

```python
"""Newline-delimited JSON over localhost TCP. Standard library only (imported by the thin client)."""
import json, os, socket, uuid

PORT = int(os.environ.get("ROADREAD_PORT", "47811"))

def send_line(sock: socket.socket, obj: dict) -> None:
    sock.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))

def recv_line(sock: socket.socket) -> dict:
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("closed before newline")
        buf += chunk
    return json.loads(buf.decode("utf-8"))

def request(image: str, deadline_s: float, port: int = PORT) -> dict:
    rid = uuid.uuid4().hex
    with socket.create_connection(("127.0.0.1", port), timeout=2) as s:
        s.settimeout(deadline_s)
        send_line(s, {"id": rid, "image": image, "deadline_s": deadline_s})
        resp = recv_line(s)
    if resp.get("id") != rid:
        raise ValueError("stale response id")
    return resp
```

- [ ] **Step 4: Implement `app/app.py`**

```python
"""Evaluator entry point. Thin client: stdlib + roadread.protocol only. Never imports torch/transformers/PIL."""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # /app (roadread/ is copied beside app.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root in development
from roadread import protocol  # noqa: E402

BUDGET_S = float(os.environ.get("ROADREAD_BUDGET_S", "27"))       # internal deadline under the 30 s hard limit

def write_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)

def main() -> int:
    t0 = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", required=True)
    a = ap.parse_args()
    img = Path(a.input_image)
    out = Path(os.environ.get("ROADREAD_OUTPUT_DIR", "/app/output")) / f"{img.stem}_output.json"
    if out.exists():
        out.unlink()                                               # never leave a stale answer
    result, diag = {"text": ""}, {}
    try:
        resp = protocol.request(str(img.resolve()), deadline_s=max(1.0, BUDGET_S - (time.monotonic() - t0)))
        result["text"] = str(resp.get("text", ""))
        if isinstance(resp.get("confidence"), (int, float)):
            result["confidence"] = max(0.0, min(1.0, float(resp["confidence"])))
        diag = resp.get("diag") or {}
    except Exception as e:                                         # worker absent/slow: fast valid failure
        diag = {"error": f"{type(e).__name__}: {e}"}
        print(f"roadread: worker error: {e}", file=sys.stderr)
    write_atomic(out, result)
    log = os.environ.get("ROADREAD_DIAG_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"image": img.name, "client_wall_s": round(time.monotonic() - t0, 3), "diag": diag},
                               ensure_ascii=False) + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5:** `python -m pytest tests/test_protocol_app.py -q` should show `4 passed`.
- [ ] **Step 6:** Commit with `git add roadread/protocol.py app/app.py tests/test_protocol_app.py && git commit -m "feat: thin evaluator client and IPC protocol"`

---

### Task 6: Pipeline with a fake engine (spec §3, §6)

**Files:** Create `roadread/engine_fake.py`, `roadread/prompts.py`, `roadread/pipeline.py`, and `tests/test_pipeline.py`.

The engine interface is `engine.read(image, instruction, max_new_tokens) -> Read(text, truncated, seconds)`, and engines may add `peak_gib()`. The reply format is two lines, `KIND: plate|sign` and `TEXT: …`.

The selection policy follows spec §6:
1. Read once at ≤ 1 MP.
2. With `cross_view` on, read one cheap second view (a 0.9 centre crop), and mark the image uncertain if the two normalized texts disagree.
3. If the first read is suspect or uncertain, and the measured budget allows it, run **one** escalation read on `reread_view()`.
4. Choose a complete candidate that isn't suspect: the escalation first, then earlier reads. Never merge characters across candidates.

- [ ] **Step 1: Write `roadread/prompts.py`**

```python
PROMPT_VERSION = "p1"
INSTRUCTION = (
    "You read one road image. Decide whether the main target is a vehicle license plate or a road sign.\n"
    "Plate: output only the registration characters exactly as printed. Omit state or province names, slogans, "
    "web addresses and dealer frames. For Chinese plates keep the leading province character and letter, e.g. 京A·12345.\n"
    "Sign: output all printed words and numbers top to bottom, joined by single spaces. Do not add units or words "
    "that are not printed.\n"
    "Copy characters literally; do not correct spelling. Reply in exactly two lines:\nKIND: plate or sign\nTEXT: <text>"
)
```

- [ ] **Step 2: Write the failing tests**

```python
from PIL import Image
from roadread.engine_fake import FakeEngine
from roadread.pipeline import Pipeline, parse_reply

IMG = Image.new("RGB", (64, 32))

def test_parse_reply():
    assert parse_reply("KIND: plate\nTEXT: 7ABC123") == ("plate", "7ABC123", True)
    assert parse_reply("KIND: Plate\nTEXT: 京A·12345\n") == ("plate", "京A·12345", True)
    assert parse_reply("KIND: sign\nTEXT: ROAD\nWORK\nAHEAD") == ("sign", "ROAD WORK AHEAD", True)
    assert parse_reply("KIND: sign\nTEXT: STOP<|im_end|>") == ("sign", "STOP", True)
    assert parse_reply("```\nKIND: sign\nTEXT: STOP\n```") == ("sign", "STOP", True)
    assert parse_reply("7ABC123") == ("sign", "7ABC123", False)            # malformed -> flagged

def test_single_read_when_confident():
    eng = FakeEngine(["KIND: sign\nTEXT: SPEED LIMIT 65"])
    r = Pipeline(eng).run(IMG, deadline_s=27)
    assert r.text == "SPEED LIMIT 65" and eng.calls == 1 and r.confidence == 0.9

def test_suspect_triggers_one_reread_on_a_different_view():
    eng = FakeEngine(["KIND: plate\nTEXT: 沪B·888888", "KIND: plate\nTEXT: 沪B·88888"])
    r = Pipeline(eng).run(IMG, deadline_s=27)
    assert r.text == "沪B·88888" and eng.calls == 2
    assert eng.sizes == [(64, 32), (128, 64)]

def test_never_more_than_one_escalation_and_keeps_first_when_both_suspect():
    eng = FakeEngine(["KIND: plate\nTEXT: 沪B·888888", "KIND: plate\nTEXT: 沪B·8888888"])
    r = Pipeline(eng).run(IMG, deadline_s=27)
    assert eng.calls == 2 and r.text == "沪B·888888" and r.confidence == 0.3

def test_no_escalation_when_budget_too_small():
    eng = FakeEngine(["KIND: plate\nTEXT: 沪B·888888"], seconds=5)
    r = Pipeline(eng, reread_upper_s=10).run(IMG, deadline_s=12)
    assert eng.calls == 1 and r.text == "沪B·888888"

def test_truncated_output_is_suspect():
    eng = FakeEngine(["KIND: sign\nTEXT: ROAD WORK", "KIND: sign\nTEXT: ROAD WORK AHEAD"], truncated=[True, False])
    assert Pipeline(eng).run(IMG, deadline_s=27).text == "ROAD WORK AHEAD"

def test_cross_view_disagreement_triggers_escalation():
    eng = FakeEngine(["KIND: plate\nTEXT: 7ABC123", "KIND: plate\nTEXT: 7ABC128", "KIND: plate\nTEXT: 7ABC123"])
    r = Pipeline(eng, cross_view=True).run(IMG, deadline_s=27)
    assert eng.calls == 3 and r.text == "7ABC123" and r.diag["uncertain"]

def test_cross_view_agreement_stops_early():
    eng = FakeEngine(["KIND: plate\nTEXT: 7ABC123", "KIND: plate\nTEXT: 7ABC 123"])
    r = Pipeline(eng, cross_view=True).run(IMG, deadline_s=27)
    assert eng.calls == 2 and r.text == "7ABC123" and not r.diag["uncertain"]
```

- [ ] **Step 3:** `python -m pytest tests/test_pipeline.py -q` should fail with `ModuleNotFoundError`.
- [ ] **Step 4: Implement `roadread/engine_fake.py`**

```python
from dataclasses import dataclass

@dataclass
class Read:
    text: str
    truncated: bool
    seconds: float

class FakeEngine:
    name = "fake"

    def __init__(self, replies, seconds=0.1, truncated=None):
        self.replies, self.seconds, self.truncated = list(replies), seconds, truncated or []
        self.calls, self.sizes = 0, []

    def read(self, image, instruction, max_new_tokens):
        i = min(self.calls, len(self.replies) - 1)
        self.calls += 1
        self.sizes.append(image.size)
        return Read(self.replies[i], self.truncated[i] if i < len(self.truncated) else False, self.seconds)
```

- [ ] **Step 5: Implement `roadread/pipeline.py`**

```python
"""One read; optional cheap cross-view check; at most one escalation; choose a complete candidate (spec §6)."""
import os, re, time
from dataclasses import dataclass, field
from .decode import bounded, center_crop, reread_view
from .normalize import normalize
from .prompts import INSTRUCTION, PROMPT_VERSION
from .rules import postprocess

_SPECIAL = re.compile(r"<\|[^|>]*\|>")
_REPLY = re.compile(r"KIND:\s*(plate|sign)\s*\n\s*TEXT:\s*(.*)", re.I | re.S)

def parse_reply(raw: str):
    raw = _SPECIAL.sub("", raw).replace("```", "").strip()
    m = _REPLY.search(raw)
    if not m:
        return "sign", " ".join(raw.split()), False
    return m.group(1).lower(), " ".join(m.group(2).split()), True

@dataclass
class Result:
    text: str
    confidence: float
    diag: dict = field(default_factory=dict)

class Pipeline:
    def __init__(self, engine, max_pixels=1_000_000, reread_pixels=2_500_000, max_new_tokens=128,
                 reread_upper_s=12.0, cross_view=None):
        self.e, self.mp, self.rp, self.mnt, self.upper = engine, max_pixels, reread_pixels, max_new_tokens, reread_upper_s
        self.xv = cross_view if cross_view is not None else os.environ.get("ROADREAD_XVIEW") == "1"

    def _read(self, view, tag):
        r = self.e.read(view, INSTRUCTION, self.mnt)
        kind, text, well_formed = parse_reply(r.text)
        rule = postprocess(text, kind)
        return {"tag": tag, "kind": kind, "text": rule.text, "raw": r.text[:200], "view": list(view.size),
                "suspect": rule.suspect or r.truncated or not well_formed or not rule.text, "seconds": round(r.seconds, 3)}

    def run(self, image, deadline_s: float) -> Result:
        t0, spent = time.monotonic(), 0.0

        def left():                      # fake engines report seconds without sleeping; real ones elapse
            return deadline_s - max(time.monotonic() - t0, spent)

        first = self._read(bounded(image, self.mp), "first"); spent += first["seconds"]
        cands, uncertain = [first], first["suspect"]
        if self.xv and not uncertain and left() > 2 * self.upper:
            x = self._read(bounded(center_crop(image, 0.9), self.mp), "xview"); spent += x["seconds"]
            cands.append(x)
            uncertain = normalize(x["text"]) != normalize(first["text"])
        escalated = None
        if uncertain and left() > self.upper:
            escalated = self._read(reread_view(image, self.mp, self.rp), "reread"); spent += escalated["seconds"]
            cands.append(escalated)
        order = ([escalated] if escalated else []) + [c for c in cands if c is not escalated]
        best = next((c for c in order if not c["suspect"]), first)
        diag = {"prompt": PROMPT_VERSION, "engine": getattr(self.e, "name", "?"), "uncertain": uncertain,
                "reads": cands, "chosen": best["tag"], "elapsed_s": round(time.monotonic() - t0, 3)}
        if hasattr(self.e, "peak_gib"):
            diag["vram_peak_gib"] = self.e.peak_gib()
        return Result(best["text"], 0.9 if not best["suspect"] else 0.3, diag)
```

- [ ] **Step 6:** `python -m pytest tests/test_pipeline.py -q` should show `8 passed`.
- [ ] **Step 7:** Commit with `git add roadread tests/test_pipeline.py && git commit -m "feat: bounded pipeline with single conditional escalation"`

---

### Task 7: Resident worker (spec §7)

**Files:** Create `roadread/worker.py`, and add a test to `tests/test_protocol_app.py`.

- [ ] **Step 1: Write the failing test** (append it to `tests/test_protocol_app.py`)

```python
def test_worker_roundtrip_and_survives_bad_request(tmp_path):
    from PIL import Image
    img = tmp_path / "s.png"; Image.new("RGB", (32, 16)).save(img)
    ready = tmp_path / "ready"
    env = {**os.environ, "ROADREAD_ENGINE": "fake", "ROADREAD_FAKE_REPLY": "KIND: sign\nTEXT: STOP",
           "ROADREAD_PORT": "47902", "ROADREAD_READY_FILE": str(ready), "PYTHONPATH": str(Path.cwd())}
    p = subprocess.Popen([sys.executable, "-m", "roadread.worker"], env=env)
    try:
        for _ in range(100):
            if ready.exists(): break
            time.sleep(0.1)
        assert json.loads(ready.read_text())["engine"] == "fake"
        assert protocol.request(str(tmp_path / "missing.png"), 10, port=47902)["text"] == ""   # error -> empty
        assert protocol.request(str(img), 10, port=47902)["text"] == "STOP"                     # still serving
    finally:
        p.terminate()
```

- [ ] **Step 2:** `python -m pytest tests/test_protocol_app.py -q -k worker` should fail with `No module named roadread.worker`.
- [ ] **Step 3: Implement**

```python
"""Resident worker: load engine once, warm up, write ready file, serve requests sequentially."""
import json, os, socket, time, traceback
from PIL import Image
from . import protocol
from .decode import load_rgb
from .pipeline import Pipeline

def make_engine():
    kind = os.environ.get("ROADREAD_ENGINE", "qwen")
    if kind == "fake":
        from .engine_fake import FakeEngine
        return FakeEngine([os.environ.get("ROADREAD_FAKE_REPLY", "KIND: sign\nTEXT: ")])
    from .engine_qwen import QwenEngine
    return QwenEngine(os.environ.get("ROADREAD_MODEL", "/models/current"), attn=os.environ.get("ROADREAD_ATTN", "sdpa"))

def main() -> None:
    t0 = time.monotonic()
    engine = make_engine()
    pipe = Pipeline(engine)
    pipe.run(Image.new("RGB", (256, 128), "white"), deadline_s=60)            # warm-up: kernels, allocator, caches
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", protocol.PORT)); srv.listen(8)
    ready = os.environ.get("ROADREAD_READY_FILE", "/tmp/roadread.ready")
    with open(ready, "w") as f:
        json.dump({"pid": os.getpid(), "startup_s": round(time.monotonic() - t0, 2), "engine": engine.name}, f)
    print(f"roadread worker ready in {time.monotonic() - t0:.1f}s engine={engine.name}", flush=True)
    while True:
        conn, _ = srv.accept()
        req = None
        try:
            req = protocol.recv_line(conn)
            r = pipe.run(load_rgb(req["image"]), deadline_s=float(req.get("deadline_s", 25)))
            protocol.send_line(conn, {"id": req["id"], "text": r.text, "confidence": r.confidence, "diag": r.diag})
            print(json.dumps({"image": req["image"], "text": r.text, "elapsed_s": r.diag["elapsed_s"]}, ensure_ascii=False), flush=True)
        except Exception:
            traceback.print_exc()
            try:
                protocol.send_line(conn, {"id": (req or {}).get("id"), "text": "", "diag": {"error": traceback.format_exc(limit=2)}})
            except Exception:
                pass
        finally:
            conn.close()

if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** `python -m pytest -q` should pass everything.
- [ ] **Step 5:** Commit with `git add roadread/worker.py tests/test_protocol_app.py && git commit -m "feat: resident worker with ready file"`

---

### Task 8: Qwen engine (GPU only, spec §1–2)

**Files:** Create `roadread/engine_qwen.py`. There is no local test; Task 12 smoke-tests it on the GPU.

Rules for the engine:
- BF16, greedy decoding, no repetition penalty (it would delete repeated digits), and at most 128 new tokens.
- Thinking is off (`enable_thinking=False`, which Qwen3.5 needs and Qwen3-VL ignores).
- Refuse to run on CPU.

- [ ] **Step 1: Implement**

```python
"""Transformers engine for Qwen3-VL / Qwen3.5 (BF16, greedy, no repetition penalty, thinking off)."""
import time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from .engine_fake import Read

class QwenEngine:
    def __init__(self, path: str, dtype=torch.bfloat16, attn: str = "sdpa"):
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm GPU not visible: refusing silent CPU fallback")
        self.name = path.rstrip("/").split("/")[-1]
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, attn_implementation=attn)
        self.model.to("cuda").eval()

    @torch.inference_mode()
    def read(self, image, instruction, max_new_tokens):
        t = time.monotonic()
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        inputs = self.proc(text=[prompt], images=[image], return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)
        new = out[0, inputs["input_ids"].shape[1]:]
        text = self.proc.decode(new, skip_special_tokens=True)
        return Read(text, truncated=len(new) >= max_new_tokens, seconds=time.monotonic() - t)

    def peak_gib(self) -> float:
        return round(torch.cuda.max_memory_reserved() / 2**30, 2)
```

- [ ] **Step 2:** `python -m py_compile roadread/engine_qwen.py` should exit 0. This only checks syntax, so it doesn't need torch.
- [ ] **Step 3:** Commit with `git add roadread/engine_qwen.py && git commit -m "feat: Qwen transformers engine"`

If the fallbacks are needed on the GPU (Task 12): when `sdpa` errors or produces garbage on gfx1100, set `ROADREAD_ATTN=eager`. When BF16 generation is wrong, change `dtype=torch.float16` and record it in `results/session1.md`.

---

### Task 9: Sample extraction and the evaluation runner

**Files:** Create `eval/extract_samples.py`, `eval/run_eval.py`, `eval/samples/`, and `tests/test_eval.py`.

The brief PDF holds exactly 10 embedded images, in sample order (verified 2026-09-25: pages 9, 10×3, 11×3, 12×2, 13). All are JPEGs of 480–750 px. The script saves each one in the **format the brief states for that sample**, so the smoke set also exercises PNG, JPEG and TIFF decoding.

- [ ] **Step 1: Write `eval/extract_samples.py`**

```python
"""Extract the 10 brief sample images (in order) into eval/samples/ with gold answers. Smoke/conventions only."""
import io, json, sys
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image

PDF = Path(__file__).resolve().parent.parent / "LabLab_AMD AI Challenge - Mini Challenge.pdf"
OUT = Path(__file__).resolve().parent / "samples"
SAMPLES = [  # (file name, gold, slice, degradation) — brief pp. 9-13
    ("image_01.png", "7ABC123", "us_plate", "clean"),
    ("image_02.png", "京A·12345", "cn_plate", "clean"),
    ("image_03.jpg", "JHT 2951", "us_plate", "angled"),
    ("image_04.png", "5XYZ891", "us_plate", "motion_blur"),
    ("image_05.jpg", "沪B·88888", "cn_plate", "low_light_glare"),
    ("image_06.png", "STOP", "word_sign", "clean"),
    ("image_07.tiff", "STOP", "word_sign", "noise"),
    ("image_08.jpg", "SPEED LIMIT 65", "speed_sign", "clean"),
    ("image_09.png", "ROAD WORK AHEAD", "warning_sign", "clean"),
    ("image_10.tiff", "35", "advisory_plaque", "clean"),
]

def main() -> int:
    doc = fitz.open(PDF)
    xrefs = [x[0] for page in doc for x in page.get_images(full=True)]
    if len(xrefs) != len(SAMPLES):
        print(f"expected {len(SAMPLES)} images, found {len(xrefs)}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for xref, (name, gold, sl, deg) in zip(xrefs, SAMPLES):
        im = Image.open(io.BytesIO(doc.extract_image(xref)["image"])).convert("RGB")
        im.save(OUT / name)
        rows.append({"image": name, "gold": gold, "slice": sl, "degradation": deg, "source": "brief pp.9-13"})
    (OUT / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} samples to {OUT}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it.** `python eval/extract_samples.py` should print `wrote 10 samples to …\eval\samples`. Open `eval/samples/image_03.jpg` and `image_05.jpg` to check by eye that they show the NY plate and the dark 沪B plate.
- [ ] **Step 3: Write the failing test** (it runs `app.py` against a fake worker)

```python
import json, os, subprocess, sys, time
from pathlib import Path
from PIL import Image

def _worker(tmp_path, port, reply):
    env = {**os.environ, "ROADREAD_ENGINE": "fake", "ROADREAD_FAKE_REPLY": reply, "ROADREAD_PORT": str(port),
           "ROADREAD_READY_FILE": str(tmp_path / f"ready{port}"), "PYTHONPATH": str(Path.cwd())}
    w = subprocess.Popen([sys.executable, "-m", "roadread.worker"], env=env)
    for _ in range(200):
        if (tmp_path / f"ready{port}").exists(): break
        time.sleep(0.1)
    return w, env

def test_run_eval_scores_times_and_overhead(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png"); Image.new("RGB", (8, 8)).save(tmp_path / "b.png")
    rows = [{"image": "a.png", "gold": "stop", "slice": "word_sign"}, {"image": "b.png", "gold": "35", "slice": "advisory_plaque"}]
    (tmp_path / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    w, env = _worker(tmp_path, 47903, "KIND: sign\nTEXT: STOP")
    try:
        r = subprocess.run([sys.executable, "eval/run_eval.py", "--manifest", str(tmp_path / "m.jsonl"),
                            "--out", str(tmp_path / "res.jsonl")], env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        s = json.loads(r.stdout.strip().splitlines()[-1])
        assert s["correct"] == 1 and s["total"] == 2 and s["by_slice"]["word_sign"] == [1, 1]
        assert s["max_s"] < 30 and s["violations"] == 0 and s["max_overhead_s"] < 5
        res = [json.loads(l) for l in (tmp_path / "res.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [x["ok"] for x in res] == [True, False] and res[1]["pred"] == "STOP"
    finally:
        w.terminate()

def test_run_eval_records_missing_worker_as_failure(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    (tmp_path / "m.jsonl").write_text(json.dumps({"image": "a.png", "gold": "STOP"}) + "\n", encoding="utf-8")
    env = {**os.environ, "ROADREAD_PORT": "47998", "PYTHONPATH": str(Path.cwd())}
    r = subprocess.run([sys.executable, "eval/run_eval.py", "--manifest", str(tmp_path / "m.jsonl"),
                        "--out", str(tmp_path / "res.jsonl")], env=env, capture_output=True, text=True, timeout=120)
    s = json.loads(r.stdout.strip().splitlines()[-1])
    assert r.returncode == 0 and s["correct"] == 0 and s["total"] == 1
```

- [ ] **Step 4:** `python -m pytest tests/test_eval.py -q` should fail because `eval/run_eval.py` doesn't exist.
- [ ] **Step 5: Implement `eval/run_eval.py`**

```python
"""Invoke app.py exactly like the harness (fresh process per image); score by normalized exact match.
Appends one JSON line per image (survives quota cut-off) and prints a summary JSON as the last stdout line."""
import argparse, json, os, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roadread.normalize import matches  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--app", default=str(Path(__file__).resolve().parent.parent / "app" / "app.py"))
a = ap.parse_args()
root = Path(a.manifest).parent
out_path = Path(a.out)
outdir = Path(os.environ.get("ROADREAD_OUTPUT_DIR", str(out_path.parent / "app_output")))
diag_path = out_path.with_suffix(".diag.jsonl")
env = {**os.environ, "ROADREAD_OUTPUT_DIR": str(outdir), "ROADREAD_DIAG_LOG": str(diag_path)}
rows = [json.loads(l) for l in Path(a.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]

def last_diag() -> dict:
    try:
        return json.loads(diag_path.read_text(encoding="utf-8").splitlines()[-1])
    except Exception:
        return {}

by = defaultdict(lambda: [0, 0]); lat, over, violations = [], [], 0
with open(out_path, "a", encoding="utf-8") as f:
    for row in rows:
        img = root / row["image"]
        t = time.monotonic()
        try:
            subprocess.run([sys.executable, a.app, "--input-image", str(img)], env=env, timeout=35)
        except subprocess.TimeoutExpired:
            pass
        dt = time.monotonic() - t
        try:
            pred = json.loads((outdir / f"{img.stem}_output.json").read_text(encoding="utf-8"))["text"]
            if not isinstance(pred, str):
                raise TypeError("text is not a string")
        except Exception:
            pred = None                                   # missing/malformed output is a failure, not a crash
        d = last_diag()
        inference = (d.get("diag") or {}).get("elapsed_s")
        overhead = round(dt - inference, 3) if isinstance(inference, (int, float)) else None
        ok = pred is not None and matches(pred, row["gold"])
        violations += int(dt >= 30 or pred is None)
        sl = row.get("slice", "all"); by[sl][0] += int(ok); by[sl][1] += 1
        lat.append(dt)
        if overhead is not None:
            over.append(overhead)
        f.write(json.dumps({**row, "pred": pred, "ok": ok, "seconds": round(dt, 3), "overhead_s": overhead,
                            "diag": d.get("diag")}, ensure_ascii=False) + "\n")
        f.flush()
lat.sort()
print(json.dumps({"correct": sum(v[0] for v in by.values()), "total": len(rows), "by_slice": dict(by),
                  "p50_s": round(lat[len(lat) // 2], 3) if lat else 0, "max_s": round(lat[-1], 3) if lat else 0,
                  "max_overhead_s": max(over) if over else None, "violations": violations}, ensure_ascii=False))
```

- [ ] **Step 6:** `python -m pytest -q` should pass all tests (35 at this point).
- [ ] **Step 7: Local end-to-end check with the fake engine**, in two shells:

```bash
ROADREAD_ENGINE=fake ROADREAD_FAKE_REPLY=$'KIND: sign\nTEXT: STOP' ROADREAD_READY_FILE=/tmp/rr.ready python -m roadread.worker &
python eval/run_eval.py --manifest eval/samples/samples.jsonl --out results/local-fake.jsonl ; kill %1
```

Expect `"correct": 2, "total": 10` (the two STOP samples) and `"violations": 0`. This shows all ten files decode, including both TIFFs.
- [ ] **Step 8:** Commit with `git add eval tests/test_eval.py && git commit -m "feat: sample extraction and harness-faithful evaluation runner"`

---

### Task 10: Playwright remote workflow (`tools/amd-gpu/remote.js`) and gates

**Files:** Create `tools/amd-gpu/remote.js` and `eval/gates.py`.

`remote.js` is the single entry point for all GPU work. It reuses `withLab`, `sh`, `put`, `get`, `killKernels` and `stop` from `gpu.js` (GPUWEBSKILL.md §7), with one browser session per command, about 10 s each.

Only `sync`, `setup`, `model`, `gates`, `worker-start`, `eval` and `data` may launch a pod. All other commands set `AUTO_LAUNCH=0` so they can never start the quota clock.

- [ ] **Step 1: Write `eval/gates.py`** (spec §8 and gate 0)

```python
"""Session-1 gates: unpacked-base proxy, GPU identity, BF16 correctness, HF throughput. Prints one JSON line."""
import json, shutil, subprocess, time

def sh(c):
    return subprocess.run(["bash", "-lc", c], capture_output=True, text=True).stdout.strip()

g = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
# Proxy for the mandated base's unpacked size: used bytes on the root overlay, excluding the persistent mount
# and our own additions. Authoritative number comes later from `docker image inspect` (Task 14).
g["root_used_bytes_excl_workspace"] = int(sh("timeout 600 du -sxb / --exclude=/workspace --exclude=/proc "
                                             "--exclude=/sys --exclude=/tmp --exclude=/root/.cache 2>/dev/null | cut -f1") or 0)
g["gfx"] = sh("rocminfo | grep -m1 -o 'gfx[0-9a-f]*'")
g["gpu"] = sh("amd-smi static --asic | grep -m1 MARKET_NAME | cut -d: -f2").strip()
g["vram"] = sh("amd-smi static --vram | grep -m1 -i size | cut -d: -f2").strip()
import torch
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
ref = a.float() @ b.float()
g["bf16_matmul_rel_err"] = float(((a @ b).float() - ref).abs().max() / ref.abs().max())
g["bf16_ok"] = g["bf16_matmul_rel_err"] < 2e-2
g["torch"], g["hip"] = torch.__version__, torch.version.hip
t = time.time()
n = sh("timeout 60 curl -sL -o /dev/null -w '%{size_download}' $HF_ENDPOINT/Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors")
g["hf_bytes_per_s"] = int(float(n or 0) / max(1e-6, time.time() - t))
g["workspace_free_bytes"] = shutil.disk_usage("/workspace").free
GiB = 2**30
g["headroom_bytes"] = 60 * GiB - g["root_used_bytes_excl_workspace"] - 3 * GiB
g["tier"] = "bake_9b" if g["headroom_bytes"] >= 20e9 else "bake_4b_or_download_9b" if g["headroom_bytes"] >= 9e9 else "download_at_startup"
print(json.dumps(g))
```

- [ ] **Step 2: Write `tools/amd-gpu/remote.js`**

```js
#!/usr/bin/env node
// Remote ROADREAD workflow on the AMD notebook GPU (built on gpu.js; see GPUWEBSKILL.md).
//   node remote.js sync                  tar roadread/ app/ eval/ (not eval/data) -> /workspace/roadread
//   node remote.js data <dir>            tar a local data dir (e.g. eval/data/dev) -> /workspace/roadread/<dir> (<= ~40 MB per call)
//   node remote.js setup                 pip --target /workspace/pylib, purge torch shadows, freeze -> app/requirements.lock
//   node remote.js model <hf_repo>       snapshot_download -> /workspace/models/<name>; symlink /workspace/models/current
//   node remote.js gates                 eval/gates.py -> results/gates-<ts>.json
//   node remote.js worker-start          (re)start the resident worker, wait for ready
//   node remote.js worker-stop | worker-log
//   node remote.js eval <manifest> <tag> run eval detached; then `tail <tag>` until the summary appears
//   node remote.js tail <tag>            progress + GPU memory (never launches a pod)
//   node remote.js pull <tag>            results/<tag>.{jsonl,summary,diag.jsonl} -> ./results (never launches)
//   node remote.js end                   stop worker/eval + kernels, then Turn-off Session (never launches)
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { withLab, sh, put, get, killKernels, stop } = require('./gpu');
const { openBrowser, OUT_DIR } = require('./lib');

const REPO = path.resolve(__dirname, '..', '..');
const R = '/workspace/roadread';
const ENV = `export PYTHONPATH=/workspace/pylib:${R} HF_HOME=/workspace/.cache/huggingface ROADREAD_MODEL=/workspace/models/current ROADREAD_OUTPUT_DIR=${R}/results/app_output;`;
const TAR = process.platform === 'win32' ? 'C:\\Windows\\System32\\tar.exe' : 'tar'; // bsdtar; Git's GNU tar breaks on C:
const [cmd, a, b] = process.argv.slice(2);
const NO_LAUNCH = new Set(['tail', 'pull', 'worker-stop', 'worker-log', 'end']);
if (NO_LAUNCH.has(cmd)) process.env.AUTO_LAUNCH = '0';

async function run(shell, timeoutMs) {
  return withLab(async (page, base) => {
    const r = await sh(page, base, shell, timeoutMs);
    process.stdout.write(r.output);
    if (!r.ok || /\[exit [1-9]/.test(r.output)) process.exitCode = 1;
    return r.output;
  });
}

async function upload(relPaths, remoteDir) {
  const rel = path.relative(REPO, path.join(OUT_DIR, 'upload.tgz'));
  execFileSync(TAR, ['-czf', rel, '--exclude=__pycache__', '--exclude=eval/data', ...relPaths], { cwd: REPO });
  const size = fs.statSync(path.join(REPO, rel)).size;
  if (size > 40e6) throw new Error(`archive is ${(size / 1e6).toFixed(1)} MB; split it (put goes through page.evaluate)`);
  return withLab(async (page, base) => {
    await put(page, base, path.join(REPO, rel), 'upload.tgz');
    const r = await sh(page, base, `mkdir -p ${remoteDir} && tar -xzf /workspace/upload.tgz -C ${remoteDir} && rm /workspace/upload.tgz && ls ${remoteDir}`);
    process.stdout.write(r.output);
  });
}

const cmds = {
  sync: () => upload(['roadread', 'app', 'eval', 'pyproject.toml'], R),
  data: () => upload([a], R),
  async setup() {
    await run(`${ENV} cd ${R} && H=$(sha1sum app/requirements.txt | cut -c1-12); \
if [ "$(cat /workspace/pylib/.req 2>/dev/null)" != "$H" ]; then rm -rf /workspace/pylib && \
pip install -q --target /workspace/pylib -r app/requirements.txt && \
(cd /workspace/pylib && rm -rf torch torch-* torchgen functorch torchvision* torchaudio* triton* numpy numpy-* numpy.libs nvidia* PIL pillow* bin/torch*) && \
echo $H > /workspace/pylib/.req; fi; \
python -c "import torch,transformers,PIL; assert 'rocm' in torch.__version__, torch.__file__; assert '/opt/venv' in torch.__file__; print('torch', torch.__version__, torch.__file__); print('transformers', transformers.__version__)" && \
pip freeze --path /workspace/pylib > app/requirements.lock && cat app/requirements.lock`, 30 * 60_000);
    if (process.exitCode) return;
    await withLab((page, base) => get(page, base, 'roadread/app/requirements.lock', path.join(REPO, 'app', 'requirements.lock')));
    console.log('pulled app/requirements.lock');
  },
  model: () => run(`${ENV} N=$(basename ${a}); python -c "from huggingface_hub import snapshot_download as d; print(d('${a}', local_dir='/workspace/models/$N'))" && \
ln -sfn /workspace/models/$N /workspace/models/current && du -sh /workspace/models/$N && df -h /workspace | tail -1`, 30 * 60_000),
  async gates() {
    const out = await run(`${ENV} cd ${R} && python eval/gates.py`, 15 * 60_000);
    const line = out.split('\n').find((l) => l.startsWith('{')) || out;
    fs.mkdirSync(path.join(REPO, 'results'), { recursive: true });
    const file = path.join(REPO, 'results', `gates-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
    fs.writeFileSync(file, line);
    console.log('saved', file);
  },
  'worker-start': () => run(`${ENV} cd ${R} && pkill -f roadread.worker; sleep 1; rm -f /tmp/roadread.ready; \
(nohup python -m roadread.worker > /workspace/worker.log 2>&1 &) ; \
for i in $(seq 1 300); do [ -f /tmp/roadread.ready ] && cat /tmp/roadread.ready && echo && exit 0; \
pgrep -f roadread.worker >/dev/null || break; sleep 2; done; tail -40 /workspace/worker.log; exit 1`, 12 * 60_000),
  'worker-stop': () => run(`pkill -f roadread.worker; echo stopped`),
  'worker-log': () => run(`tail -60 /workspace/worker.log`),
  eval: () => run(`${ENV} cd ${R} && mkdir -p results && rm -f results/${b}.jsonl results/${b}.diag.jsonl && \
(nohup python eval/run_eval.py --manifest ${a} --out results/${b}.jsonl > results/${b}.summary 2> results/${b}.err &) && echo started ${b}`),
  tail: () => run(`cd ${R}/results && echo "done: $(wc -l < ${a}.jsonl 2>/dev/null)"; tail -2 ${a}.jsonl 2>/dev/null | cut -c1-300; \
echo "summary: $(cat ${a}.summary 2>/dev/null)"; tail -3 ${a}.err 2>/dev/null; rocm-smi --showmeminfo vram | grep -E 'Used|Total'`),
  pull: () => withLab(async (page, base) => {
    fs.mkdirSync(path.join(REPO, 'results'), { recursive: true });
    for (const ext of ['jsonl', 'summary', 'diag.jsonl']) {
      const local = path.join(REPO, 'results', `${a}.${ext}`);
      await get(page, base, `roadread/results/${a}.${ext}`, local).then(() => console.log('pulled', local)).catch((e) => console.error(ext, e.message));
    }
  }),
  async end() {
    await withLab(async (page, base) => {
      await sh(page, base, 'pkill -f roadread.worker; pkill -f run_eval.py; true');
      await killKernels(page, base);
    }).catch((e) => console.log('no running pod or cleanup skipped:', e.message));
    const { context, page } = await openBrowser();
    try { await stop(page); } catch (e) { console.log('stop:', e.message); } finally { await context.close(); }
  },
};

if (!cmds[cmd]) { console.log(fs.readFileSync(__filename, 'utf8').split('\n').slice(1, 15).join('\n')); process.exit(2); }
if (['eval', 'tail', 'pull'].includes(cmd) && !(cmd === 'eval' ? b : a)) { console.error(`${cmd}: missing tag`); process.exit(2); }
cmds[cmd]().catch((e) => { console.error('ERROR:', e.message); process.exit(1); });
```

- [ ] **Step 3: Check it locally, without a pod.**
  - `node --check tools/amd-gpu/remote.js` should give no output.
  - `node tools/amd-gpu/remote.js` should print the usage text and exit with code 2.
  - `node -e "require('child_process').execFileSync(process.platform==='win32'?'C:\\\\Windows\\\\System32\\\\tar.exe':'tar',['-czf','tools/amd-gpu/out/t.tgz','roadread'],{stdio:'inherit'})"` should exit 0, which confirms the tar fix.
- [ ] **Step 4: Check that read-only commands never launch** (with the pod off). `node tools/amd-gpu/gpu.js status` should show `"status": "not_found"`. Then `node tools/amd-gpu/remote.js tail smoke` should end with `ERROR: NO_RUNNING_SESSION`, and `gpu.js status` should still show `not_found`.
- [ ] **Step 5:** Commit with `git add tools/amd-gpu/remote.js eval/gates.py && git commit -m "feat: Playwright remote workflow and session-1 gates"`

---

### Task 11: GPU session 1, gates and setup (spec §F, session 1)

Budget: 35–50 pod minutes. Run each command, read its output, then start the next. Record everything in `results/session1.md`.

- [ ] **Step 1:** `node tools/amd-gpu/gpu.js status`. Note `quota_remaining_seconds`. If it's `NOT_SIGNED_IN`, stop and ask the user to run `! cd tools/amd-gpu && node login.js`.
- [ ] **Step 2:** `node tools/amd-gpu/remote.js sync`. This launches the pod (about 80 s), and the listing should show `app eval pyproject.toml roadread`.
- [ ] **Step 3:** `node tools/amd-gpu/remote.js gates` takes 1–10 minutes because of `du`. Expect `"gfx": "gfx1100"`, `"bf16_ok": true`, and `hf_bytes_per_s` around 1e8.
  - Read the `tier` field. `bake_9b` means ship the 9B model inside the image. `bake_4b_or_download_9b` means either bake the 4B or prove the startup download (Task 14). `download_at_startup` means the image carries code only.
  - **If `bf16_ok` is false:** use FP16 everywhere, via `dtype=torch.float16` in `engine_qwen.py`, and note it.
- [ ] **Step 4:** `node tools/amd-gpu/remote.js setup` takes 2–6 minutes. It must print `torch 2.13.0+rocm10.0.0 /opt/venv/...` and `transformers 5.x`, and it pulls `app/requirements.lock`.
  - **If the pip resolve fails on Python 3.14:** pin `transformers==<latest 5.x that installs>` in `app/requirements.txt` and rerun.
  - **If the torch assert fails:** a dependency pulled torch into pylib. Add that package to the purge list and rerun. Never "fix" it by reinstalling torch.
- [ ] **Step 5:** `node tools/amd-gpu/remote.js model Qwen/Qwen3-VL-4B-Instruct` takes about 2–4 minutes at ~100 MB/s. `du` should show about 8.9G. Record the revision with `gpu.js sh "cat /workspace/models/Qwen3-VL-4B-Instruct/.cache/huggingface/download/*.metadata 2>/dev/null | head -3"`, or from the `snapshot_download` output.
- [ ] **Step 6:** `node tools/amd-gpu/remote.js end` should end with `SESSION_STOPPED`. Then `gpu.js status` should show `not_found`, and you should note the quota.
- [ ] **Step 7:** Write `results/session1.md` with the gates JSON as a table, the tier decision, the pinned versions from `requirements.lock`, the model revision, and quota used. Commit with `git add results/session1.md app/requirements.txt app/requirements.lock && git commit -m "docs: session-1 gate measurements and dependency lock"`

---

### Task 12: GPU smoke test (10 brief samples)

Budget: 15–20 pod minutes.

- [ ] **Step 1:** `node tools/amd-gpu/remote.js sync`. This picks up `eval/samples/`, and the pod launches if needed.
- [ ] **Step 2:** `node tools/amd-gpu/remote.js worker-start` should print JSON like `{"pid":…,"startup_s":…,"engine":"current"}`. Record `startup_s`; it must be well under 420.
  - **On failure** the command prints the worker log tail. Common fixes:
    - `sdpa` error: set `ROADREAD_ATTN=eager` by adding it to `ENV` in `remote.js` for this run.
    - Processor or template error: check the transformers version, and pin it in `app/requirements.txt`.
  - After a fix, run `sync` and then `worker-start` again.
- [ ] **Step 3:** `node tools/amd-gpu/remote.js eval eval/samples/samples.jsonl smoke`, then repeat `node tools/amd-gpu/remote.js tail smoke` (about 1 minute apart) until `summary:` shows JSON. Note `rocm-smi` VRAM used: it should be between 1 and 44 GiB.
- [ ] **Step 4:** `node tools/amd-gpu/remote.js pull smoke`. Check the pulled files:
  - `results/smoke.summary`: target `correct` 10/10, `max_s` < 25, `violations` 0, and `max_overhead_s` < 1. Overhead is client wall time minus worker elapsed time, which is spec acceptance "client overhead".
  - For each miss in `results/smoke.jsonl`, read `diag.reads[].raw`. Put it in one of three categories: prompt (wording or format), rule (post-processing), or resolution (the reread needed).
- [ ] **Step 5: Fix misses test-first.**
  - Rule misses: add a failing case to `tests/test_rules.py` from the raw text, then fix `rules.py`.
  - Prompt misses: edit `INSTRUCTION` **and** bump `PROMPT_VERSION` to `p2`.
  - Never add an image-to-answer lookup (spec: the samples establish conventions only).
  - Then run `python -m pytest -q`, `sync`, `worker-start`, and `eval … smoke2`.
- [ ] **Step 6:** `node tools/amd-gpu/remote.js end`. Write `results/smoke-notes.md` covering every run: correct/total, max_s, VRAM, the misses and their categories, what changed, and quota used. Commit it along with the code fixes.

---

### Task 13: Development set, challengers and release-tier sweep (spec Testing Decisions 1–5)

- [ ] **Step 1: Build the dev and holdout sets locally** under the gitignored `eval/data/`. Each gets 120 images, 20 per slice across `us_plate`, `cn_plate`, `word_sign`, `speed_sign`, `advisory_plaque` and `warning_sign`, and uses the same jsonl row format as `samples.jsonl`, plus `family` (the original image or template id) and `sha256`.
  - Sources, from spec §E: a CCPD subset for Chinese plates (decode the labels from the file names; keep the labels out of runtime file names), OpenALPR benchmark US plates (check the terms), MUTCD-rendered signs, and permitted real photos.
  - Add synthetic degradations: blur, glare, low light, noise, skew and small text.
  - **Split by `family`**, so no family appears in both sets. Don't open the holdout results until Step 6.
  - Keep each archive under 40 MB by downscaling the originals to ≤ 1600 px on the long side, or split by slice.
- [ ] **Step 2:** `node tools/amd-gpu/remote.js sync`, then `node tools/amd-gpu/remote.js data eval/data/dev`.
- [ ] **Step 3: Baseline.** Run `worker-start`, then `eval eval/data/dev/dev.jsonl dev-q3vl4b-p1`, then repeat `tail dev-q3vl4b-p1` about every 10 minutes (120 × ≤ 25 s is about 50 minutes at most), then `pull`, then `end`. In `results/dev.md`, record exact match overall and per slice, p50/max latency, max overhead, peak VRAM (`diag.vram_peak_gib`), and a per-error category table.
- [ ] **Step 4: Cross-view experiment** (spec §6), on the same model and the same day if quota allows. Add `ROADREAD_XVIEW=1` to `ENV` in `remote.js`, run `worker-start`, then `eval … dev-q3vl4b-p1-xv`. Adopt it only if it **fixes more complete answers than it breaks**, has no slice regression, and keeps max_s ≤ 25.
- [ ] **Step 5: Challengers, one at a time and on separate days.**
  - (a) `Qwen/Qwen3.5-9B` (19.3 GB). First free the space with `gpu.js sh "rm -rf /workspace/models/Qwen3-VL-4B-Instruct"`, because 8.9 + 19.3 GB plus caches is too tight for 25 GB. Then run `model Qwen/Qwen3.5-9B`, `worker-start`, and `eval … dev-q35-9b-p1`.
  - (b) `Qwen/Qwen3-VL-8B-Instruct`, the same way.
  - (c) Optional: PaddleOCR-VL-1.6. This needs a new `roadread/engine_paddle.py` behind `ROADREAD_ENGINE=paddle` with the same interface, `read()` → `Read`. Write it only if (a) and (b) leave consequential errors.
  - The adoption rule for every challenger: more fixes than breaks, no slice regression, max_s ≤ 25, peak VRAM ≤ 44 GiB, and it passes the Task 11 tier (size) gate.
- [ ] **Step 6: Freeze and run the holdout once.** Write `results/release.md` with the model repo and revision, prompt version, `max_pixels`/`reread_pixels`, XVIEW on or off, `requirements.lock`, attention and dtype. Upload the holdout with `data eval/data/holdout`, then run `eval eval/data/holdout/holdout.jsonl holdout` once and record the result. If it leads to a change, the holdout becomes dev data and you need a new holdout (spec step 5).
- [ ] **Step 7:** Commit `results/*.md` and any code changes, each with its tests.

---

### Task 14: Container packaging (spec §8), on a Linux Docker builder, not the notebook

**Files:** Create `docker/Dockerfile`, `docker/entrypoint.sh`, and `docker/check_container.sh`.

- [ ] **Step 1: Dockerfile.** Weights are baked in only if the Task 11 tier allows it. For `download_at_startup`, delete the `COPY models` line and set `ROADREAD_MODEL_REPO` and `ROADREAD_MODEL_REV` instead.

```dockerfile
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
COPY app/requirements.lock /app/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /app/requirements.txt && python -c "import torch; assert 'rocm' in torch.__version__"
COPY roadread /app/roadread
COPY app/app.py /app/app.py
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY models/current /models/current
ENV ROADREAD_MODEL=/models/current HF_HUB_OFFLINE=1 PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1
RUN mkdir -p /app/input /app/output && chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]
```

`models/current` must be a **real directory** in the build context. Docker won't follow a symlink out of the context, so copy the frozen snapshot there. Use `--no-deps` because the lock already lists every package that isn't torch.

- [ ] **Step 2: `docker/entrypoint.sh`**

```sh
#!/bin/sh
set -e
if [ -n "$ROADREAD_MODEL_REPO" ] && [ ! -f /models/current/config.json ]; then   # download fallback (spec §8)
  HF_HUB_OFFLINE=0 python3 -c "import os,time;from huggingface_hub import snapshot_download as d;t=time.time();d(os.environ['ROADREAD_MODEL_REPO'],revision=os.environ['ROADREAD_MODEL_REV'],local_dir='/models/current');print('download_s',round(time.time()-t,1))"
fi
python3 -m roadread.worker > /tmp/worker.log 2>&1 &
while [ ! -f /tmp/roadread.ready ]; do
  kill -0 $! 2>/dev/null || { cat /tmp/worker.log; exit 1; }
  sleep 1
done
echo "roadread ready: $(cat /tmp/roadread.ready)"
exec tail -f /tmp/worker.log
```

- [ ] **Step 3: `docker/check_container.sh`**, which runs the release gates against a built image:

```sh
#!/bin/sh
# usage: docker/check_container.sh <image> <dir-with-image_01..10>
set -e
IMG=$1; IN=$2
SIZE=$(docker image inspect --format '{{.Size}}' "$IMG"); echo "size_bytes=$SIZE"; [ "$SIZE" -le 64424509440 ]
BASE=rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
docker image inspect --format '{{json .RootFS.Layers}}' "$BASE" > /tmp/base.json
docker image inspect --format '{{json .RootFS.Layers}}' "$IMG" | python3 -c "import json,sys;b=json.load(open('/tmp/base.json'));i=json.load(sys.stdin);assert i[:len(b)]==b,'base layers not an ordered prefix';print('layers ok',len(b))"
T0=$(date +%s); C=$(docker run -d --device=/dev/kfd --device=/dev/dri --group-add video -v "$IN":/app/input "$IMG")
until docker logs "$C" 2>&1 | grep -q "roadread ready"; do sleep 2; [ $(( $(date +%s) - T0 )) -lt 600 ]; done
echo "startup_s=$(( $(date +%s) - T0 ))"
for f in $(ls "$IN"); do s=$(date +%s%N); docker exec "$C" python3 /app/app.py --input-image /app/input/$f
  echo "$f $(( ($(date +%s%N) - s) / 1000000 ))ms $(docker exec "$C" cat /app/output/${f%.*}_output.json)"; done
docker rm -f "$C" >/dev/null
```

- [ ] **Step 4: Build and check.** `docker build -f docker/Dockerfile -t roadread:rc1 .` then `sh docker/check_container.sh roadread:rc1 eval/samples`.
  - Required: `size_bytes` ≤ 64424509440, `layers ok 11`, `startup_s` < 420 (measure it twice, each from a clean `docker run`), every image under 25000 ms, and the outputs match `samples.jsonl`.
  - Also sample GPU memory externally every 3 s with `rocm-smi --showmeminfo vram`; the peak must be between 1 and 44 GiB.
- [ ] **Step 5: Publish.** Push to a public registry and test `docker logout && docker pull <ref>` anonymously. Submit the reference **only** through the lablab "Mini Challenge 2 Image" form. Never commit the reference.
- [ ] **Step 6:** Commit with `git add docker && git commit -m "feat: release container and checks"`

---

## Self-review against the spec

| Spec requirement | Task | Pinned by |
|---|---|---|
| Stories 1–3: CLI, file name, string `text` | 5 | `test_app_writes_named_json_utf8` |
| Story 4: PNG, JPEG and TIFF | 4, 9 | `test_formats_and_modes`; the smoke set is saved in its stated formats |
| Stories 5, 15: deadlines | 5, 6, 9 | `BUDGET_S`, `test_no_escalation_when_budget_too_small`, `violations` |
| Story 6: GPU use and memory | 8, 12, 14 | CPU refusal, `vram_peak_gib`, rocm-smi sampling |
| Stories 7–12, 31–32: rules | 3 | 8 rule tests incl. the property test and sample 3 EXCELSIOR |
| Stories 13–14: orientation and resolution | 4, 6 | EXIF test, `reread_view` test |
| Stories 16–18: persistence | 10, 11 | `/workspace/pylib`, `/workspace/models`, append-only results |
| Stories 19–22: evaluation and records | 9, 13 | by-slice summary, `results/*.md` |
| Story 23: malformed output | 6, 9 | `parse_reply` tests, `pred=None` failure |
| Stories 24–26: container | 14 | `check_container.sh` |
| Stories 27–28: gates | 10, 11 | `gates.py` tier and `bf16_ok` |
| Story 29: thin client | 5, 12 | `test_app_does_not_import_torch`, `max_overhead_s` < 1 |
| Story 30: worker | 7, 14 | ready file, entrypoint |
| Story 33: disagreement trigger | 6, 13 | cross-view tests, Task 13 Step 4 |
| Stories 34–35: model choice and download fallback | 13, 14 | adoption rule, entrypoint download branch |

**Still open (spec §G):**
- Harness startup semantics: whether it honours `CMD`. Ask in Discord.
- The closing date.
- GA 36 verification of the I/O rule.
- The grading GPU family.

None of these block Tasks 1–12.
