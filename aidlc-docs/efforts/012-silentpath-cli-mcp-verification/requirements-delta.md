# Effort 012 requirements delta

Date: 2026-09-28
Baseline: SILENTPATH inception documents and existing Tasks 13-16.

## Scope decision

Continue the existing main-track SILENTPATH effort. D1 is narrowed to a reproducible PyTorch ROCm SDPA path-inspection and measurement tool on the available W7900D. vLLM, OCR/VLM, and broader theme expansion are deferred. This does not change the project baseline; it records an implementation choice consistent with the existing plan’s stop/pivot rules.

## Requirements made concrete

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| D1-R1 | Run a GPU-free matrix and generate a report from persisted fake records. | `run --fake`, then `report`; recorded smoke result. | Met |
| D1-R2 | Run real PyTorch SDPA matrix cells for default, forced Flash, and forced Math, including unsupported/failure records. | W7900 records plus profiler operator evidence. | Met for tested shapes |
| D1-R3 | Separate synchronized generation timing from profiler overhead and preserve raw path evidence. | Producer implementation, records, profiler logs/traces. | Met |
| D1-R4 | Record reproducibility metadata, prompt fingerprints, repetitions, results, and artifact references; pass the pinned revision to model and tokenizer loads. | W7900 configs, records, environment manifest, archive hashes, revision propagation test. | Met |
| D1-R5 | Provide canonical `run`/`report` CLI and store-backed `which_path`/`summarise_store` MCP functions. | CLI smoke, MCP function calls against six GPU records, tests. | Met; transport process deferred |
| D1-R6 | Compare repeated path performance and generated outputs without overstating conclusions. | Five-repeat B and reversed-order B2 results; evidence README. | Met for these workloads |
| D1-R7 | Package a clean, reviewable release that a reviewer can reproduce and understand. | Root README, dependency/license/source details, isolated wheel install and fake run/report; 2-3 minute demo recording. | In progress: video remains |
| D1-R8 | Publish a competition entry and track its credit only after verifying the actual dashboard route/rules. | Submission record and confirmed dashboard requirements. | Open; no submission made |

## Constraints and non-goals

- Target environment: W7900D (`gfx1100`), pinned Qwen2.5-1.5B-Instruct snapshot, PyTorch ROCm SDPA.
- Do not claim positive silent fallback from these results. Default-versus-Flash remains unresolved by the comparator.
- Do not generalize the observed decode speed ratios or token divergence beyond the measured prompts/configuration.
- vLLM producer/runtime, OCR/VLM demo, networked MCP transport, and multi-theme expansion are deferred and are not acceptance criteria for this bounded D1 implementation.
- Competition dashboard mechanics remain unverified and are not treated as established requirements.
