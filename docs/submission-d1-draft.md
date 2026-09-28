# SILENTPATH D1 submission draft

Use these verified details when the signed-in LabLab form confirms the project category and required fields. Do not publish until the demo video link is available.

## Project details

- **Title:** SILENTPATH: See Which Attention Path Ran on AMD ROCm
- **Tagline:** Measure the attention operator your AMD inference actually used, compare its output and latency, and keep the profiler evidence with every run.
- **Source:** https://github.com/HarshdipSaha/amdmega-hackathon (release branch: `feat/silentpath-w7900`)
- **Demo video:** Pending a 2-3 minute recording.
- **Category / challenge theme:** Confirm in the signed-in participant form before selecting.
- **AMD technology:** AMD ROCm, PyTorch ROCm, AMD Radeon Pro W7900D (`gfx1100`).
- **Additional technology:** PyTorch SDPA, Transformers, Python, profiler traces, CLI, MCP query functions.

## Short description

PyTorch can accept an attention-backend request without that request alone proving which kernel ran. SILENTPATH runs controlled PyTorch SDPA comparisons on an AMD GPU, identifies the selected operator from profiler evidence, measures repeated generation outside the profiler pass, and stores outputs, timings, metadata, and raw artifacts together. Its CLI and MCP query functions read the same record store.

## Results and limits

On a Radeon Pro W7900D with Qwen2.5-1.5B-Instruct at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, the profiler distinguished forced Flash Attention and Math. Across two prompts configured with `max_new_tokens=128` and five repetitions per cell, Math was 1.30-1.36x slower in the first run. A fresh reversed-order session measured 1.292x and 1.299x. The short generated sequence first differed at token 65 in both sessions; the long sequence matched. No positive silent fallback was observed. Default dispatch used Flash according to the profiler, while the comparator currently reports the Default-versus-Flash pairing as unresolved. These are workload-specific measurements, not general speed or answer-quality claims.

In a separate one-run G1 arithmetic test, Flash and Math produced identical `1064.875` text and token IDs with a maximum chosen-token log-probability delta of `0.0086715`. The G1 path labels were derived from requests rather than independent profiler evidence; those measurements do not support a speed claim.

The tracked evidence includes configs, records, reports, environment metadata, representative profiler traces, and checksums for the full raw archives: [`docs/evidence/silentpath-w7900/README.md`](evidence/silentpath-w7900/README.md).

## Reproduction

The CPU-only fake path and isolated installation are documented in the [repository README](../README.md). Real W7900D reproduction requires the AMD ROCm/PyTorch image and the pinned model snapshot; the exact commands and session configs are in the evidence README.

## Submission checklist

- [ ] Confirm project category and form requirements in the signed-in dashboard.
- [ ] Record and upload the demo video.
- [ ] Publish the reviewed source branch.
- [ ] Submit the project and record its URL and submission timestamp.
