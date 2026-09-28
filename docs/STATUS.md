# SILENTPATH project status

Last updated: 2026-09-28. See the [specification](SPEC.md), [public rules check](hackathon-rules-2026-09-28.md), [Effort 012](../aidlc-docs/efforts/012-silentpath-cli-mcp-verification/effort-state.md), and [AI-DLC registry](../aidlc-docs/registry.md).

## Current state

The W7900D implementation and measurements are complete. The release is published on [`feat/silentpath-w7900`](https://github.com/HarshdipSaha/amdmega-hackathon/tree/feat/silentpath-w7900). The root README presents SILENTPATH and ROADREAD as separate projects. The package, pinned user-space dependencies, MIT license, and GPU-free reproduction path were verified in an isolated Python 3.12 environment. Fresh local verification passed **162 tests**. Effort 012 remains in progress for the recorded demo, signed-in submission review, and submission.

## W7900D result

On AMD Radeon Pro W7900D (`gfx1100`, PyTorch `2.13.0+rocm10.0.0`), profiler operators distinguished forced Flash Attention from Math. Default SDPA dispatch used Flash in the profiler, but the comparator leaves Default-to-Flash unresolved. No positive silent fallback was observed.

For two prompts with `max_new_tokens=128` and five repetitions per cell, Math was 1.30-1.36x slower than Flash in the first run. A fresh reversed-order session measured ratios of 1.292x and 1.299x. The short prompt first diverged at generated token 65 in both sessions; the long prompt matched. The model was pinned to Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. These observations are workload-specific. No chosen-token log-probabilities were collected, so no log-probability delta is available.

See the [W7900D evidence bundle](evidence/silentpath-w7900/README.md) and [execution plan](superpowers/plans/2026-09-28-silentpath-w7900-winning-path.md).

The fact-checked [D1 submission draft](submission-d1-draft.md) and [timed demo script](demo-script.md) are ready; category, recorded video, and submission URL are still pending.

## Remaining

- Capture and add a 2-3 minute demo video; the runnable walkthrough is `tools/demo-silentpath.ps1`.
- Confirm the signed-in LabLab submission form and whether D1 qualifies for a main-track project entry. The public live page says submissions are open; no draft or submission has been made.
- Record the video, inspect the signed-in form, and submit the project. The release branch is already public.
- Verify private Legend/final-challenge rules. Do not treat them as established from the public page.
