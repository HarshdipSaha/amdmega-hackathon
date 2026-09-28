# Effort 012: SILENTPATH CLI, MCP, and D1 Verification

## Metadata

- **Effort ID:** `012`
- **Reference:** `silentpath-cli-mcp-verification`
- **Tasks:** SILENTPATH Tasks 13-16
- **State:** `in-progress`
- **Updated:** 2026-09-28
- **Authorization:** The user instructed execution of the W7900D plan and subsequently asked to finish its remaining work.

## Outcome

SILENTPATH now has an installable Python package, pinned core/GPU dependency files, dual-track root README, MIT license, fake and ROCm SDPA CLI paths, store-backed MCP query functions, W7900 matrices, and evidence capture. The SDPA producer passes a configured model revision to both Transformers loads; the W7900 configs pin Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

Hardware validation ran on AMD Radeon Pro W7900D (`gfx1100`), PyTorch `2.13.0+rocm10.0.0`, ROCm/HIP 10.0.0/7.15.26333, and Transformers 5.17.0. Profiler operators distinguished forced Flash from Math. No positive silent fallback was observed. Five-repeat decode runs found Math about 1.30-1.36x slower than Flash in the first session and 1.292x/1.299x slower in a fresh reversed-order session. The short sequence first differed at generated token 65 in both sessions; the long sequence matched. No chosen-token log-probabilities were collected, so no log-probability delta is available. These observations apply only to the measured prompts and setup. Default dispatch was observed as Flash, while the comparator leaves Default-versus-Flash unresolved.

See the [W7900D evidence bundle](../../../docs/evidence/silentpath-w7900/README.md), [submission draft](../../../docs/submission-d1-draft.md), [execution plan](../../../docs/superpowers/plans/2026-09-28-silentpath-w7900-winning-path.md), and [root project README](../../../README.md).

## Delivered

- Installable `silentpath` project metadata and console command.
- `requirements-silentpath.lock` for the GPU-free tool and `requirements-silentpath-gpu.lock` for Transformers additions inside the pinned AMD ROCm image. PyTorch remains supplied by that image.
- Root README distinguishes SILENTPATH main-track D1 from ROADREAD Mini-Challenge 2 and documents quickstart, evidence, reproduction, and limitations.
- Local [submission copy](../../../docs/submission-d1-draft.md) and timed [video script](../../../docs/demo-script.md) are prepared; neither is a submitted project or recorded video.
- `LICENSE`, tracked sample run records, reports, environment metadata, selected traces, and hashes for full trace archives.
- Revision-pinned SDPA model and tokenizer loading, with W7900 configs pinned to the evidence revision.
- `run`/`report` CLI; `which_path`/`summarise_store` MCP functions over the same store.
- Runnable PowerShell demo walkthrough for the fake CLI and both MCP query functions.
- GPU Sessions A, B, B2, and C, with artifacts retrieved before notebook shutdown.

Implementation commits: `08b530a`, `800ab18`, `a2895a9` on `feat/silentpath-w7900`.

## Remaining before completion

- Record a polished 2-3 minute demo video. `tools/demo-silentpath.ps1` is the runnable walkthrough; the video itself has not been captured.
- Confirm the signed-in LabLab project form and D1 category/eligibility. The public live page says submissions are open; no LabLab platform draft or submission has been created.
- Publish the reviewed branch and submit the project once the demo and signed-in requirements are complete.
- Participant-only Legend and final-challenge mechanics remain unverified.

## Deferred and limitations

- vLLM is deferred; real D1 producer support is PyTorch SDPA.
- MCP functions were called against six GPU records; a networked MCP transport process was not launched.
- OCR/VLM was not tested as part of this D1 GPU work.
- No general answer-quality, invoice-correctness, or universal speed claim follows from these workloads.
- The notebook's wall-clock quota is not enforced by the per-cell run budget.

## Verification evidence

- Fresh full local suite: `python -m pytest -q` — **162 passed in 20.19s**.
- Isolated Python 3.12 virtual environment: installed the pinned dependencies, built and installed the package, invoked `silentpath --help` outside the source directory, ran four fake matrix cells, and rendered two comparisons.
- `tools/demo-silentpath.ps1` succeeded: four fake records, two comparisons, and both MCP query functions reading that store.
- Revision propagation test first failed because the helper was absent; after implementation, the focused SDPA test file passed.
- Notebook was stopped after artifact retrieval; GPU status returned `not_found`.
