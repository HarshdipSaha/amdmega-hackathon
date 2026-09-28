# Spec: SILENTPATH

**Which path did your AMD GPU silently take, what did it cost you, and did it change your answer?**

Project for the **Lablab x AMD AI Academy Challenge**, Sept 1 - Dec 1 2026. Individual entry.
Written in the `to-spec` format. No issue tracker is configured for this repo, so this file is the tracked spec; the triage label would be `ready-for-agent`.

Status: `docs/STATUS.md`. Brief: `docs/hackathon-brief.md`. Council record: `docs/council/verdict.md`.

> **Evidence update (2026-09-28):** G1 has been run on one Qwen 2.5 1.5B arithmetic prompt on a W7900D. Flash and Math returned identical output text and token IDs, with a maximum chosen-token log-probability delta of `0.0086715`. Explicit Efficient Attention failed on this model's GQA shape. This narrow result does not establish silent fallback or predict OCR answer changes. D1 implementation is in progress; keep further answer-divergence claims as hypotheses until measured.

---

## Problem Statement

Someone runs a language model on an AMD GPU. They set no flags, read no release notes, and accept every default. Their stack quietly makes a decision on their behalf — which attention kernel to use, whether to use the GPU at all — and never tells them which decision it made.

Reports from AMD serving stacks describe cases where backend selection or GPU fallback was not apparent to users. These reports motivate SILENTPATH; they are background evidence, not measurements produced by this project:

- vLLM v0.18.2 gated `ROCM_ATTN` behind a flag defaulting to `False` and silently fell back to `TRITON_ATTN`. Prefill attention ran **5.5x slower** and total GPU time **2.5x worse**. There was no warning. It was found only because someone went digging with `rocprof`.
- ollama's own issue tracker states plainly that "GPU-to-CPU fallback happens silently with no user-visible warning" — layers reported as offloaded while the buffers actually landed in `CPU_Mapped`.
- lemonade shipped a **7x** regression via silent CPU fallback on gfx1201. LM Studio silently drops to Vulkan on unsupported targets. ROCm itself has loaded models onto the CPU while reporting a GPU device.

The project will investigate two questions; neither is assumed to be answered by the reports below.

**The first is a measurement question.** Published issue reports and benchmark writeups describe substantial slowdowns under some AMD backend and fallback configurations. SILENTPATH will measure GPU-seconds for its own specified workloads and configurations. The checked-in G1 run is one arithmetic prompt with one successful observation per backend; it does not establish a speed ratio or validate a general claim about token-count cost estimates.

**The second is a correctness question.** The tested Qwen prompt produced identical Flash/Math output text and token IDs with small log-probability differences. It did not test vLLM's backend set or OCR extraction. Whether other ROCm backends or workloads change a user-visible decision remains open; CUDA findings cannot settle that question for AMD.

A 2026 paper on Kernel Contracts states the gap precisely: when a matmul on AMD produces a different result than the same matmul on NVIDIA, "there is no formal artifact to arbitrate the dispute." It publishes a specification. Nobody has built the suite.

So the user's real problem is that they cannot answer three questions about their own inference run: *which path did I take, what did it cost, and did the path change my answer?*

## Solution

**SILENTPATH** makes the invisible decision visible, and then measures whether it mattered.

The system runs the same input through a matrix of configurations that a user would otherwise never control, and reports three things per configuration: the **path actually taken**, the **measured GPU-seconds** it consumed, and the **output it produced**. Then it diffs them.

The thesis it tests is falsifiable and stated in advance:

> For selected AMD configurations and workloads, runtime path selection may be difficult to observe; backend choice may affect measured GPU time and, for some inputs, decision-level output. SILENTPATH tests these claims by recording available path evidence, repeated GPU-time measurements, and output comparisons.

The demonstration that carries this is deliberately not the engineering. It is one sentence, and it needs no ROCm knowledge to land:

> **Hypothesis to test:** the same scanned invoice may return a different dollar amount under different attention settings. No checked-in OCR/VLM experiment currently demonstrates this answer divergence.

If an OCR/VLM experiment shows the extracted amount changes, that would be a decision-level result with practical consequences. Until then, treat it as a motivating hypothesis. The measured G1 result is narrower: one arithmetic prompt produced identical Flash/Math text and token IDs, despite log-probability differences. Prior reports of backend slowdowns motivate measurement, but this repository's G1 single-run timings do not establish a speed or cost advantage. The D1 value proposition is to measure and report paths and costs; performance claims require repeated measurements on a defined workload.

Six deliverables share one measurement core. Each is independently complete, independently demoable, and maps honestly onto one of the program's six bi-weekly themes. They are genuinely distinct artifacts, not one artifact relabelled six times; that distinction is deliberate and is discussed under *Further Notes*.

| # | Deliverable | Program theme |
|---|---|---|
| D1 | **Probe + MCP server** — reports which backend was selected and what it cost in measured GPU-seconds | 1: serving, MCP, tool-calling, GPU resource consumption |
| D2 | **Divergence harness on a vision-language OCR workload** — the invoice demo | 2: OCR with a multimodal model |
| D3 | **RAG advisor over a ROCm corpus** — explains *why* you are on a slow path, citing real docs and issues | 3: RAG on proprietary domain knowledge |
| D4 | **Support-matrix scraper** — builds the currently-undocumented map of which ROCm/vLLM version really supports which gfx target | 4: web-scraping agent that finds a software version number |
| D5 | **Multi-agent triage loop** — diagnose, hypothesize, patch config, re-measure, verify | 5: multi-agent software engineering |
| D6 | **Learned configuration policy** — GRPO-trained, rewarded on measured GPU-seconds subject to preserving output fidelity | 6: novel game + fine-tuning |

D6 may also serve as preparation if current official rules later confirm a final challenge involving RL or fine-tuning. That possibility is speculative: the public information checked on 2026-09-28 does not confirm the final challenge or its format. Course 6's GRPO material and AMD's *veRL: Production-Ready RL Post-Training on ROCm* (published 2026-09-07) provide technical context, not evidence about challenge requirements.

## Gating Verifications

These run in Week 1, before implementation. Each can independently invalidate the plan.

**G1 — Does the divergence exist?** Same model, same prompt, same seed, each available attention backend; dump logits; compare with `np.allclose` at several tolerances. Record not just numerical delta but whether the *decoded output* differs. This needs no profiler and is roughly 40 lines.
*If it fails:* the correctness half dies. Keep the cost half (D1, D3, D4, D5 survive; D2 becomes a cost demo). Thesis narrows to "token counts are not cost on AMD."

**G2 — Does the local AMD hardware work at all? — RESOLVED 2026-09-10: BLOCKED.**
Measured, not assumed. The machine is a Ryzen 5 5600H (Cezanne) with an **integrated** Radeon `DEV_1638` (gfx90c), 15.4 GB RAM, **no discrete NVIDIA GPU**, and **no Ryzen AI NPU** (XDNA starts at Ryzen 7040). WSL2 has Ubuntu but **no `/dev/kfd`**, no `/opt/rocm` and no `rocminfo`; `/dev/dri` exposes only D3D12 graphics passthrough, not ROCm compute. gfx90c was never officially ROCm-supported — the community workaround is `HSA_OVERRIDE_GFX_VERSION=9.0.0` — and ROCm-on-WSL targets discrete RDNA3/Instinct, not a Cezanne iGPU.

*Consequences, now binding on this spec:* there is **no local ROCm path**; every GPU measurement runs on the ~50 MI300X cloud hours. The Lemonade-SDK-on-NPU fallback named in earlier drafts **does not apply** — this machine has no NPU. The cross-vendor NVIDIA comparison is **not available** (see *Out of Scope*). Full evidence table in `docs/STATUS.md`.

*Why this is survivable:* the backend plurality the project studies exists only on CDNA, so a gfx90c iGPU would have had roughly one occupant in the backend zoo — the MI300X droplet is the correct instrument, not a consolation. The thesis is measurement-heavy and compute-light, so 50 GPU-hours is ample.

**G3 — Can one codebase serve multiple themes as distinct projects?** Ask the organizers in writing, and read the dashboard rules. Also read off the numeric Legend threshold and check the live leaderboard for how close anyone else is.
*If it fails:* the six-deliverable structure collapses to one or two submissions. Concentrate everything into D1+D2 as a single deep project and re-plan the XP path around courses and community.

## User Stories

1. As someone serving a model on an AMD GPU, I want to know which attention backend was actually selected, so that I am not silently running on a path 5x slower than the one I assumed.
2. As that same person, I want to be warned when my workload has silently fallen back to CPU or Vulkan, so that a 7x regression does not reach production unnoticed.
3. As an engineer evaluating AMD against NVIDIA, I want measured GPU-seconds per request rather than token counts, so that my cost comparison reflects what the hardware actually did.
4. As a finance-operations engineer extracting totals from scanned invoices, I want to know whether my extraction result depends on an inference setting I never chose, so that I can trust the numbers I post to a ledger.
5. As that same engineer, I want a reproducible report showing which inputs are sensitive to backend choice, so that I can route those documents to a verified configuration.
6. As a machine-learning engineer, I want to know whether two runs of my model that differ only in backend produce different logits, so that I can tell a genuine model change from an infrastructure artifact.
7. As a researcher, I want the divergence measured at decision level and not only at bit level, so that I can distinguish expected floating-point noise from an outcome that actually changed.
8. As a researcher, I want the null result published if the backends agree, so that the question is settled either way rather than left open.
9. As someone migrating a workload from NVIDIA to AMD, I want the same input run on both vendors with outputs compared side by side, so that I can decide whether the difference is acceptable before I migrate.
10. As that same person, I want a machine-readable record of any cross-vendor divergence, so that I have a concrete artifact to raise with a vendor rather than an anecdote.
11. As a developer on unfamiliar AMD hardware, I want to know whether my specific gfx target is genuinely supported by the ROCm and vLLM versions I have installed, so that I do not spend a day debugging an unsupported configuration.
12. As that developer, I want that support information gathered from real release notes and issue trackers rather than from folklore, so that it reflects what actually ships.
13. As a developer hitting a slow path, I want an explanation citing the actual ROCm documentation or GitHub issue responsible, so that I can act on it instead of guessing.
14. As a developer, I want a suggested configuration change and a measurement proving it helped, so that I am not trusting advice blindly.
15. As an agent author, I want the probe exposed over MCP as a tool, so that my own agent can ask what a candidate configuration will cost before committing to it.
16. As a cost-conscious operator, I want a policy that picks a configuration per workload to minimise measured GPU-seconds subject to preserving output fidelity, so that I get savings without silently changing answers.
17. As that operator, I want the policy to refuse configurations that change decisions even when they are faster, so that cost optimisation never trades away correctness.
18. As a hackathon participant on a fixed credit budget, I want every experiment resumable and cached, so that a crashed run does not burn irreplaceable GPU-hours.
19. As that participant, I want a hard spend ceiling enforced in the harness, so that I can reserve credit for later project work.
20. As a reviewer of this project, I want each deliverable to stand alone with its own README, demo and result, so that I can evaluate it without running the other five.
21. As a reviewer, I want the measurement methodology stated with repetition counts and variance, so that I can judge whether a reported speed difference is real.
22. As a maintainer of an AMD serving stack, I want a reproducible script attached to any divergence report, so that I can confirm the finding without reconstructing the setup.
23. As a future contributor, I want the measurement core separated from the workloads, so that adding a new workload does not require touching measurement code.
24. As the author, I want results recorded in a stable schema from day one, so that Week 2 measurements remain comparable with Week 10 measurements.
25. As the author racing a 9-day scoring lag, I want each deliverable shippable independently, so that nothing is blocked behind an unfinished later stage.

## Implementation Decisions

**One seam: the `RunRecord`.** Everything in the system is either a producer or a consumer of a single immutable record describing one inference run under one configuration. A producer runs a workload and emits a record. Every consumer — comparator, advisor, triage loop, policy — reads records and nothing else. This is deliberately the highest seam available, and it means the entire stack above measurement can be developed and tested with no GPU present.

A `RunRecord` carries: the configuration requested; the path **actually observed** (which is not assumed to equal the one requested — detecting that mismatch is the point); measured wall and GPU time; device telemetry; the workload input identifier; and the output, retained as both decoded text and raw logits where available.

**Path detection is observational, not declarative.** Reading back the environment variable that was set proves nothing, because the silent-fallback bug is precisely the case where the stack ignores it. Detection reads what the runtime reports it selected, corroborated by log capture and by device activity, and records a confidence level rather than asserting certainty. Where the three disagree, that disagreement is itself a finding and is preserved in the record.

**Telemetry via `amd-smi` and torch HIP events; rocprofiler-sdk is out.** rocprofiler-sdk has no first-class Python binding and requires a C++ tool injected via `LD_PRELOAD`. The council was unanimous that this consumes roughly two weeks for a need that hackathon-grade timing satisfies. `amd-smi` ships an official Python package; torch HIP events give per-region device timing. If a finding later demands per-kernel counters, that is a documented extension, not a dependency.

**Divergence is measured at three levels, and they are reported separately.** Bit-level equality; numerical distance over logits with tolerances stated; and decision-level difference — did the decoded answer, the extracted field, the retrieved document change? Only the third is a *finding*. Conflating the three is the error the Contrarian warned about, and the schema prevents it by construction.

**Cost is always measured, never estimated.** No component may derive cost from token counts. Where a token-based estimate is shown, it appears only as a baseline being refuted.

**Configuration matrix is data, not code.** Which backends, models, shapes and workloads to sweep lives in a config file, so the matrix can shrink to fit the credit budget without code changes. Every run is content-addressed and cached; re-running a completed cell is free.

**Spend ceiling is enforced in the harness.** The runner tracks cumulative GPU-hours against a configured cap and refuses to start a cell that would exceed it. Credits are not claimed until the harness runs unattended end-to-end on a local or trial configuration.

**Local-first, cloud-for-confirmation.** Development, iteration and the entire consumer layer run locally. The MI300X is used only for cells that genuinely require it — the backends that exist only on CDNA, and the final confirmation sweep. This follows directly from the 30-day credit expiry.

**Cross-vendor comparison is a workload, not a separate system — but it has no local hardware to run on.** The design intent stands: the same producer interface would run on an NVIDIA GPU and emit records in the same schema, making the cross-vendor result just a comparator query. The council's Expansionist rated this the highest-value artifact in the whole plan, precisely on the (mistaken) assumption that both vendors' hardware were already owned. G2 (2026-09-10) measured that this machine has no discrete GPU of any vendor. The seam is built regardless — it costs nothing to keep the door open — but running it requires a separately-funded NVIDIA instance and is moved to *Out of Scope* for this plan.

**The RAG corpus is built, not scraped ad hoc.** D4's scraper produces the corpus D3 consumes: ROCm release notes, vLLM release notes and the relevant GitHub issue threads, with source URL and retrieval date on every chunk. Advice without a citation is not shipped.

**D6's policy is small and its reward is measured.** Configuration selection is framed as an episodic decision problem over the record store — the "novel game" is choosing a configuration; the reward is negative measured GPU-seconds with a hard fidelity constraint that zeroes any action changing a decision-level output. Trained with GRPO via Unsloth on a small model, mirroring AI Academy Course 6, LoRA-scale, short episodes. If the credit budget cannot support training, a documented heuristic baseline plus the environment definition still constitutes the deliverable.

## Testing Decisions

**A good test here exercises external behaviour and never requires a GPU.** Because the `RunRecord` is the only seam, every consumer is tested against fixture records — including deliberately pathological ones. This is what makes a GPU-dependent project testable in CI.

- **Comparator** is tested against synthetic record pairs with known relationships: bit-identical, numerically close but decision-identical, and decision-divergent. The third must be reported as a finding and the first two must not. This is the single most important test in the project, because it is the test that stops the system from claiming a result it does not have.
- **Path detection** is tested against captured log fixtures from real runs, including the case where the requested backend and the observed backend disagree, and the case where the signals conflict and the record must carry low confidence rather than a guess.
- **Spend ceiling** is tested by simulating a matrix that exceeds the cap and asserting the runner refuses to start the offending cell rather than aborting mid-run.
- **Cache/resume** is tested by interrupting a matrix run and re-running it, asserting completed cells are not recomputed and results are identical.
- **Scraper** is tested against saved HTML fixtures, asserting extracted version/target pairs and that every record carries a source URL and retrieval date. Live network calls do not appear in tests.
- **RAG advisor** is tested for citation integrity: every claim in an answer must resolve to a chunk in the corpus. An uncited claim fails the test.
- **Statistical claims** require a stated repetition count and reported variance. A single-shot timing difference is never reported as a speed result — a direct response to the finding that AITER shows 2-16x higher measurement variability.

**Prior art for the test style:** the fixture-driven, no-hardware-in-CI approach mirrors the harness tests in `H:/augsepthacks/apartresearch` (`docs/superpowers/plans/2026-09-05-attest-harness.md`), where provider adapters were tested against recorded responses rather than live APIs.

## Out of Scope

- **Writing or optimising GPU kernels.** The kernel-generation space is saturated — 20+ papers in 2026, AMD's own GEAK and AgentKernelArena, and a $20M-funded startup. SILENTPATH measures kernels; it does not write them. This boundary is what keeps the project from competing with the sponsor's own research team.
- **Porting CUDA to HIP.** Already submitted by another participant on this hackathon page.
- **Fixing the divergence.** Characterising and reporting it is the contribution. Making the backends agree is a serving-stack change and out of reach in eleven weeks.
- **Batch-invariant kernel implementations.** Idea B from the council. Related and genuinely open, but a different, narrower project with a weaker demo.
- **Multi-GPU and tensor-parallel configurations.** Single-GPU only; the credit budget does not support 8x MI300X sweeps.
- **Training anything large.** LoRA-scale only, and only for D6.
- **A public website, hosted dataset, or preprint during the race.** Explicitly overruled by the council: none of it earns XP and all of it costs time. Reconsider after Dec 1.
- **Generic agent-framework scaffolding.** Agents appear only where they do measurable work (D5's triage loop). Roughly 79% of a prior AMD hackathon's submissions were agents; adding another is invisible.
- **Cross-vendor (NVIDIA) conformance measurement.** G2 (2026-09-10) confirmed this machine has no discrete GPU of any vendor, so there is no NVIDIA hardware to compare against. The producer seam is written so this is a documented future extension rather than a redesign, but running it needs a separately-funded NVIDIA instance and is not part of this plan.

## Further Notes

**On the six-deliverable structure and honesty.** The project plan proposes six artifacts built around one measurement core. Whether submissions sharing a codebase are eligible across multiple themes, and what degree of distinction is required, is unconfirmed; see [`hackathon-rules-2026-09-28.md`](hackathon-rules-2026-09-28.md). Treat the proposed six-deliverable structure as a product plan, not a confirmed eligibility strategy.

**On the risk that there is no finding.** Different kernels producing different floating-point values is expected; the project distinguishes numeric changes from decision changes. The checked-in G1 is a narrow numeric-only result for one arithmetic prompt, not evidence about invoice OCR. Whether backend choice changes an extracted field remains an open hypothesis. Cost or speed claims also remain open until repeated, comparable measurements support them; the single G1 timing observations do not establish them.

**On timing.** An earlier planning note assumed weekly Friday point updates and a first-three-to-Legend race, then estimated an effective Nov 21 deadline. The current public page and live dashboard reviewed on 2026-09-28 do not confirm the Legend threshold, prize ordering, or final-challenge rules. Do not use that estimated deadline for planning; check current participant-dashboard information instead.

**On the credit.** $100 ≈ 50 hours on one MI300X at $1.99/hr, on DigitalOcean GPU Droplets. Credits expire 30 days after being claimed. They are not claimed until the harness runs unattended end to end.

**On what the council overruled.** The Expansionist's cross-vendor conformance angle was judged the most defensible artifact in the whole plan and is retained, cheaply, because the seam makes it nearly free. The same advisor's proposals for a living dataset site, mid-race preprint and devrel outreach were rejected unanimously in peer review as scope that earns no XP. That judgement is recorded here so it is not silently reversed later.

**On the final challenge.** Earlier strategy notes assumed an RL/GRPO-shaped final challenge and scheduled D6 accordingly. The current public information checked on 2026-09-28 does not confirm the final challenge's existence, requirements, unlock timing, or duration. D6 remains a project deliverable; its relationship to any final challenge is unconfirmed.
