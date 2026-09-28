# SILENTPATH W7900D Winning Path — Execution Plan

> **For agentic workers:** Execute this plan in order, checking the gates before using another AMD notebook session. The implementation details for the existing core are in `docs/superpowers/plans/2026-09-10-silentpath-core-and-d1.md`; do not rebuild completed tasks.

**Goal:** Ship one credible, independently usable AMD inference-path and cost probe for the main AI Academy Challenge, then expand only where measured results and confirmed program rules justify it.

**Architecture:** Keep `RunRecord` as the single interface between GPU producers and the existing store, comparator, budget runner, reporter, CLI, and MCP tools. D1 uses the implemented PyTorch SDPA producer for the W7900D; vLLM is deferred. Path claims rely on runtime profiler evidence, and measurements preserve raw logs and traces.

**Tech stack:** Python, PyTorch ROCm/HIP events, `amd-smi`/`rocprofv3` where available, Pydantic, PyYAML, FastMCP, and the Playwright notebook driver in `tools/amd-gpu/`.

---

## Decision snapshot — 28 September 2026

**Execution status:** GPU Sessions A, B, B2, and C are complete. The SDPA producer, `run`/`report` CLI, store-backed query functions, synchronized timing, profiler artifacts, and repeated W7900D evidence are in place. Packaging is installable and was tested in an isolated Python 3.12 environment; the fake run and report worked from outside the source tree. The model revision is now passed explicitly to both Transformers loaders. Math was 1.292x/1.299x slower than Flash on the two reversed-order decode prompts; the short output diverged at generated token 65, while the long output matched. No positive silent fallback was observed, and Default-versus-Flash remains unresolved in the comparator. The pod is stopped. Effort 012 remains open for a recorded demo, signed-in submission review, and competition submission. Details: `aidlc-docs/efforts/012-silentpath-cli-mcp-verification/effort-state.md` and `docs/evidence/silentpath-w7900/README.md`.

- This is the **main SILENTPATH project**, not the Mini-Challenge 2 ROADREAD Docker entry. The mini challenge has a separate grading path.
- The live AMD notebook provides a Radeon Pro W7900D (`gfx1100`, 48 GiB). It has a three-hour daily session quota and `/workspace` is persistent on this `rgapi-*` pod. Check live quota each time; stop the pod after every GPU session. See `GPUWEBSKILL.md` and `COMPUTE.md`.
- The September 28 G1 rerun produced the same output (`1064.875`) for forced PyTorch `FLASH_ATTENTION` and `MATH`, with maximum chosen-token log-probability difference `0.0086715`. Forced `EFFICIENT_ATTENTION` failed with “No available kernel” for Qwen's GQA shape. Those files are under `results/g1/`. **There is no demonstrated answer change or silent fallback.** One timing per path is not a speed result.
- The SDPA producer, CLI, MCP query functions, and GPU evidence are implemented. Effort 012 remains open for clean-checkout packaging, README reconciliation, a short demo, and submission. `GpuBudget` meters cell duration but does not enforce notebook pod-lifetime quota. See the effort state and evidence bundle for measured results and limitations.
- The [event page](https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge) confirms September 1–December 1 and individual participation. The [live dashboard](https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge/live) explicitly separates Quest XP from Mini-Challenge grading and says profile points do not determine submission evaluation. The public page does **not** currently expose the numeric Legend threshold, final gate, or whether one codebase may count as multiple projects. Treat the September 10 `docs/hackathon-brief.md` interpretation as provisional until the signed-in dashboard or organizers confirm it.

## Win condition and scope

The first deliverable is **D1: “Which path did my AMD inference run take, how much GPU time did it consume, and how certain are we?”** A judge should be able to run a fake matrix without a GPU, see a real W7900D result with reproducible commands, and use the CLI or MCP tools to query the same records. A result may say “path unresolved.” It must not equate a requested setting with an observed kernel or call numeric drift an answer change.

**Canonical interfaces for D1:** `python -m silentpath.cli run --matrix ... --out ... --cap-hours ... [--fake]` and `python -m silentpath.cli report --out ...`; real W7900D runs select `--producer sdpa`. The MCP functions are `which_path` and `summarise_store`, backed by the same records. vLLM remains deferred; `--fake` is the GPU-free route.

Optimize two independent outcomes: (1) a strong, finished technical submission with a real demo and evidence; (2) the account's XP/Legend progress if the private dashboard confirms those prize mechanics. Shipping D1 takes priority over building six partly finished theme artifacts. ROADREAD is useful as an existing AMD vision workload after D1 works, not as proof of SILENTPATH's correctness claim.

## Task 0 — Before spending more GPU time (local/browser, no pod)

- [ ] Read the signed-in program dashboard and record: current XP, Legend threshold, final challenge eligibility/unlock, deadline and grading dates, milestone requirements, and whether D1–D6 can be submitted as distinct projects. Save dated screenshots or exact page text in `docs/hackathon-rules-2026-09-28.md`. Ask organizers only for rules the dashboard does not answer; do not assume approval from silence.
- [x] Reconcile `docs/STATUS.md`, `aidlc-docs/registry.md`, `aidlc-docs/efforts/012-silentpath-cli-mcp-verification/effort-state.md`, and `results/g1/README.md` with the actual code and G1 artifacts. Remove the unsupported “8.8/10 contender” as a factual claim; it is an opinion, not a measured outcome. The G1 README now identifies its requested backend labels as lacking independent runtime evidence. Record that G1 covers one model/prompt and two successful SDPA requests, not the vLLM CDNA backend matrix in the original spec. Revise the changing-invoice headline in `docs/SPEC.md` to a testable hypothesis until an answer change is actually measured.
- [x] Complete the GPU-free CLI `run --fake` and `report` parts of Task 14, plus the MCP functions of Task 15 using fixture records. The runnable fake walkthrough and fixture tests pass. vLLM is deferred under the stop/pivot rule.
- [x] Implement `silentpath/producers/sdpa.py` and focused tests for observed-path evidence, failures, timing, and artifacts. Add the W7900 prefill/decode configs with different output lengths and the `--producer sdpa` selector. The model revision is passed to both Transformers loaders and covered by a test. vLLM remains an optional follow-on.
- [x] Capture the exact notebook image, software versions, model revision, and workload manifests. Full raw traces are in retrieved archives with hashes; selected traces and reviewable records are tracked in `docs/evidence/silentpath-w7900/`. Workspace limits required scratch storage during GPU runs; all evidence was retrieved before shutdown.

**Gate 0:** `python -m silentpath.cli run --matrix configs/smoke.yaml --out runs/dryrun --cap-hours 1 --fake` and `python -m silentpath.cli report --out runs/dryrun` produce a readable report from stored fake records. The dashboard/rules notes distinguish confirmed facts from open questions. No GPU needed.

## GPU session A — Prove what the W7900D actually runs (target: 30–45 minutes)

1. Run `node gpu.js status` from `tools/amd-gpu/`; note quota and whether a pod is already running. Set an absolute stop deadline at least 15 minutes before quota exhaustion. Stage code and caches under `/workspace/silentpath` and `/workspace/.cache`, then launch with `node gpu.js launch`. Recheck the hub quota during the session: `GpuBudget` does not account for pod startup, idle time, downloads, or browser overhead.
2. Inventory `torch`, HIP/ROCm, `rocprofv3`, available memory, and exact GPU target. Save the output to `results/silentpath-w7900/environment.json` in the repo and a matching copy on the pod.
3. Run **one** fixed Qwen prompt with the default SDPA dispatcher, then with forced `FLASH_ATTENTION` and `MATH`. Capture the runtime's actual attention implementation and a profiler/kernel trace or equivalent independent runtime evidence. `gate/g1_transformers.py` currently writes `backend_observed` from the backend it was *asked* to use; that is not observation. Fix this before claiming path detection or fallback.
4. **Deferred optional check:** vLLM was not initialized in this run. The working PyTorch SDPA path met D1's scope; revisit vLLM only after release or on suitable hardware.
5. Pull the raw stdout, stderr, trace, and environment manifest locally. Run `node gpu.js stop` even after an error.

**Gate A:** At least two working paths are distinguishable from independent runtime evidence. If not, ship an honest single-path probe and spend the next session testing fallback/unsupported requests; do not label requested settings as observed paths. A forced `EFFICIENT_ATTENTION` exception is an incompatibility result, not proof of a silent fallback. The SDPA producer must write a valid `RunRecord` for successes and failures before moving to Session B.

**Gate A status:** Passed for forced Flash and Math on the measured W7900D shapes. Profiler operator evidence was saved. No positive silent fallback was observed.

## GPU session B — Produce defensible cost evidence (target: 45–75 minutes)

1. Use the existing `configs/smoke.yaml` as a template, but replace unsupported backend names after Gate A. Run `configs/w7900-prefill.yaml` and `configs/w7900-decode.yaml` as separate matrices so each can set its own global output length; the current matrix expander cannot vary `max_tokens` by workload. Use at least two input lengths. Keep model, revision, dtype, input, output length, seed, and software fixed within each paired comparison. Put a prompt/config revision fingerprint in each cell's config and use a fresh output directory for changed prompts; `RunRecord.compute_id` does not hash prompt text and would otherwise treat an edited prompt with the same workload ID as a cached cell.
2. Before trusting metrics, fix `silentpath/telemetry.py`: synchronize before/after the timed region, name HIP-event elapsed time as an elapsed region rather than active GPU-seconds unless profiler counters support the latter, and use a real peak-memory API or time-series sample if the field is called `peak_memory_bytes`. Keep a one-off AMD-SMI reading under a separately named sample field. Warm the model before timing.
3. Randomize or alternate backend order across repeated runs to reduce cache/thermal ordering bias. Record at least **five** repetitions per comparable cell, separate model initialization from generation, and store synchronized wall time, HIP-event elapsed time, variation, measured memory, token count, full logs, and actual selected path. If `rocprofv3` evidence is available, retain the raw trace and the command that generated it.
4. Run through the existing matrix/budget/store interface; confirm that an interrupted sweep resumes without repeating completed cells. Check the **hub's live quota** and absolute stop deadline separately from the cell budget. Pull records into a dated `results/silentpath-w7900/<run-id>/` bundle. Copy a minimal manifest, representative records, and generated report to `docs/evidence/silentpath-w7900/<run-id>/` for the public submission.
5. Calculate per-shape paired ratios and uncertainty. Publish a speed claim only where path identity is independently confirmed, repetitions are sufficient, and the observed difference exceeds run-to-run variation. Report a null result if it does not.
6. Stop the pod with `node gpu.js stop`.

**Session B2 / Gate B:** Repeat the best comparison in a fresh notebook session with the same pinned environment, identical prompts/config, a new output directory, and new raw artifacts. The W7900 pod was turned off and relaunched (the hub retained the same pod identifier); the pinned model was downloaded again after scratch reset. The decode ratios reproduced. D1 remains a path-visibility and measurement tool; the observed sequence difference is workload-specific. The pod was stopped after retrieval.

**Gate B status:** Passed for these two decode prompts: reversed-order Math/Flash ratios reproduced at 1.292x and 1.299x. This does not establish a general workload performance claim.

## GPU session C — End-to-end user demo (target: 30–60 minutes)

1. Run the completed `python -m silentpath.cli run --matrix configs/w7900-prefill.yaml --out runs/w7900-demo --cap-hours <cell-cap> --producer sdpa` on the W7900D, store the records, and generate `python -m silentpath.cli report --out runs/w7900-demo` without hand-editing the output. Verify the `which_path` and `summarise_store` MCP tools answer from the same records. The `<cell-cap>` limits matrix cell time; the hub quota and stop deadline are enforced separately.
2. Demonstrate one operational decision: a requested path matched, differed, failed, or remained unresolved, and show the measured cost with its confidence/variance. Include raw evidence links in the report. A clean “unsupported on this model” result is useful if that is what the GPU shows.
3. If D1 is stable, add a small Qwen3-VL/ROADREAD document workload to test the original OCR angle. Compare answer-level output only with a fixed image/prompt and verified path. If answers agree, say so. Do not block D1 release on finding a divergent invoice total.
4. Capture a short terminal/demo video, exact setup and run commands, and a result table. Stop the pod.

**Gate C:** A new reviewer can reproduce the fake demo locally and the real W7900D example from the documented command and evidence bundle. The current `README.md` must identify ROADREAD and SILENTPATH separately so the main entry is not mistaken for the OCR submission.

**Gate C status:** Fake run/report/MCP walkthrough passed from an isolated install, and the W7900D CLI/MCP example and evidence are recorded. A clean-checkout rerun on the GPU and a recorded video remain, so Gate C is not yet closed.

## Release and competition actions (mostly no GPU)

- [x] Complete release packaging: dual-track README, pinned core/GPU dependency locks, MIT license, source link, sample records, limitations, exact reproduction commands, and isolated installation/fake-run/report verification. G1 is explicitly labeled numeric-only with request-derived backend labels. The video is tracked separately below.
- [ ] Record and upload a 2–3 minute D1 demo video using `tools/demo-silentpath.ps1` and the measured evidence; no video has been recorded yet.
- [x] Publish the release source branch `feat/silentpath-w7900` at commit `c72683a`.
- [ ] Submit **one** D1 project after signed-in form requirements and category eligibility are confirmed. The public live page says submissions are open; the project form remains inaccessible until signed in. Record the submission URL and timestamp if submitted.
- [ ] Complete confirmed low-effort XP activities (account/profile, eligible Academy courses, ROCm certification, community help) independently of GPU work. Check the private dashboard before relying on the September 10 XP table or the guessed Legend threshold.
- [ ] Only after D1 is submitted, choose the next distinct artifact from confirmed challenge rules: D2 OCR path/cost comparison using ROADREAD, or D4 support-matrix evidence if D2 yields no useful effect. Defer RAG, multi-agent triage, and GRPO until there is an independently useful underlying dataset and the relevant theme/final task is published.

## Operating rules for every notebook session

- Use `GPUWEBSKILL.md` and `tools/amd-gpu/gpu.js`. `status` does not start a pod; `py`/`sh` may auto-launch one. Keep a single browser-profile owner. If SSO asks for MFA, the account owner completes it in the visible login window.
- Start with a time-boxed question, expected artifact path, and absolute stop time. A three-hour daily quota is a wall-clock limit while the pod exists, including idle time; the existing `GpuBudget` does not protect it. Recheck live quota between long commands and keep at least 15 minutes for artifact retrieval and shutdown. Use `/workspace` for persistent data and model caches; monitor the 25 GB limit. Do not put credentials into archives, images, logs, or the public repo.
- Before shutdown, download the result bundle and verify file sizes/JSON readability locally. Always run `node gpu.js stop` and confirm `status: not_found`. Keep raw measurements; later summaries are derived artifacts.
- Stop optional experiments once the gate is answered. Prioritize the next release action over another sweep of the same arithmetic prompt.

## Stop/pivot rules

| Observation | Action |
|---|---|
| vLLM does not install or expose two valid paths on W7900D within Session A | Use the proven PyTorch SDPA route for D1. Revisit vLLM only on suitable hardware or after release. |
| No independent way to observe the selected kernel | Report `unresolved`; do not claim silent fallback. Investigate profiler/log capture before cost attribution by backend. |
| Repeated path timings overlap within variation | Lead with path visibility and unsupported-path detection. Report timing as inconclusive. |
| No answer-level divergence | Keep numeric drift as a numerical observation; drop the changed-invoice headline. |
| Dashboard disallows multiple submissions from one codebase | Submit one deep D1 and earn the remaining confirmed XP through other activities. |
| Final Legend requirements differ from the September 10 brief | Replan the prize path immediately from the current official rule; technical submission quality and XP are tracked separately. |

## Evidence and rule sources

- Project state: `aidlc-docs/registry.md`, `aidlc-docs/efforts/012-silentpath-cli-mcp-verification/effort-state.md`, `results/g1/`, `docs/SPEC.md`.
- Notebook operation: `GPUWEBSKILL.md`, `COMPUTE.md`.
- Official public event dates/individual format: https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge
- Official public live dashboard, XP/submission distinction: https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge/live
- Program details still requiring signed-in verification: `docs/hackathon-brief.md` (captured September 10; potentially stale).
