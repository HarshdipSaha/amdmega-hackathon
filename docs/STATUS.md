# STATUS — Lablab x AMD AI Academy Challenge (Sept 1 – Dec 1, 2026)

Last updated: 2026-09-10 (idea selected, spec written, implementation plan written and reviewed to revision 4; G2 measured and resolved as BLOCKED; no code written yet).
Spec: `docs/SPEC.md`. Brief: `docs/hackathon-brief.md`. Council: `docs/council/verdict.md`.

Project: **SILENTPATH** — which path did your AMD GPU silently take, what did it cost you, and did it change your answer?

## Where we are

| Phase | State | Notes |
|---|---|---|
| Understand hackathon | DONE | Full page captured with Playwright incl. collapsed XP tables and FAQ → `docs/hackathon-brief.md` |
| Research: AMD platform | DONE | Dev Cloud = DigitalOcean droplets, $1.99/hr MI300X, ~50 GPU-hrs, credits expire 30 days after claim; rocprofiler-sdk has no Python binding; GEAK/AgentKernelArena are AMD's own |
| Research: NVIDIA/market gap | DONE | Silent fallback is the most-repeated AMD failure mode; kernel-gen space saturated; CUDA→HIP agent already submitted here |
| Research: literature | DONE | All determinism/batch-invariance work is CUDA-only; no ROCm determinism work exists; Kernel Contracts publishes spec but no suite |
| Research: competition strategy | DONE | Building dominates XP ~7x; final gate likely GRPO-shaped; 9-day scoring lag; 6 submissions from 2,383 participants |
| Idea selection | DONE | LLM council: 5 advisors, 3 peer reviews, chairman verdict. Unanimous for Idea A, restructured as tiered submission factory. Ideas B and C rejected |
| Spec | DONE | `docs/SPEC.md` in `to-spec` format (problem, solution, 25 user stories, implementation/testing decisions, out of scope) |
| Implementation plan (core + D1) | DONE | `docs/superpowers/plans/2026-09-10-silentpath-core-and-d1.md`, 16 TDD tasks. Four revisions, each reviewed by reconstruct-and-run + mutation battery (rev 1: 3 tests failed; rev 2: two headline fixes still wrong; rev 3: 121 pass, 7 mutation survivors; rev 4 closes them) |
| **G1 — divergence kill-test** | **TODO — needs the cloud droplet** | G2 result means this cannot run locally. Same prompt/seed, each attention backend, dump logits, compare |
| **G2 — local hardware viability** | **DONE 2026-09-10 — BLOCKED** | Measured, not assumed. See "G2 measured result" below. No local ROCm path exists on this machine |
| **G3 — rules + finish line** | **TODO** | Ask organizers in writing re: one codebase across multiple themes. Read Legend XP threshold off dashboard. Check leaderboard |
| Onboarding XP | TODO | AMD account, connect lablab account, Discord (+50), complete AMD profile (+100) |
| Implementation plan | TODO | `writing-plans` skill, after gates pass |
| D1–D6 | TODO | See timeline below |

## G2 measured result — 2026-09-10, BLOCKED

Measured on the development laptop, not assumed. This is why the gate existed.

| Fact | Measured value | How |
|---|---|---|
| CPU | AMD Ryzen 5 5600H (Cezanne, Zen 3) | `Win32_Processor` |
| GPU | **AMD Radeon(TM) Graphics, integrated, `DEV_1638`** → gfx90c | `Win32_VideoController`, `Get-PnpDevice -Class Display` |
| Discrete NVIDIA GPU | **none — one display adapter total** | `Get-PnpDevice -Class Display` returned a single row |
| Ryzen AI / XDNA NPU | **none** — 5600H predates Phoenix; NPUs start at Ryzen 7040 | CPU model |
| RAM | 15.4 GB | `Win32_ComputerSystem` |
| WSL2 | present, Ubuntu + docker-desktop distros | `wsl -l -v` |
| ROCm in WSL | **absent** — no `/opt/rocm`, no `rocminfo` | `wsl -d Ubuntu` |
| `/dev/kfd` (ROCm compute node) | **absent** | `wsl -d Ubuntu -- ls -l /dev/kfd` |
| `/dev/dri` | present (`card0`, `renderD128`) — D3D12 passthrough for graphics, **not** ROCm compute | same |
| Windows HIP SDK | absent | no `C:\Program Files\AMD\ROCm` |
| torch | 2.2.2+**cpu**, `cuda.is_available()` False | `python -c` |

**Three findings that change the plan:**

1. **gfx90c is not ROCm-supported.** The last ROCm line with full GCN 5.1 support was 5.7, and gfx90c APUs were never officially included; the community workaround is `HSA_OVERRIDE_GFX_VERSION=9.0.0` to impersonate gfx900. Combined with the missing `/dev/kfd`, there is no ROCm compute path here at all — and ROCm-on-WSL's support matrix covers discrete RDNA3/Instinct, not a Cezanne iGPU.
2. **There is no local NVIDIA GPU.** The cross-vendor NVIDIA-vs-MI300X conformance artifact — which the council's Expansionist rated the most defensible thing in the whole plan, precisely because we appeared to own both sides — **is not available**. Either drop it or rent an NVIDIA hour separately.
3. **"Tier 0 runs locally, no cloud needed" was fiction**, exactly as the Contrarian and Executor both warned. Every GPU measurement collapses onto the ~50 MI300X cloud hours.

**Why this is survivable, and arguably better.** The backend plurality the project studies — `ROCM_ATTN`, `ROCM_AITER_FA`, `TRITON_MLA`, `AITER_MLA` — only exists on CDNA anyway. A gfx90c iGPU would have had roughly one occupant in the "backend zoo", which is the exact objection the Contrarian raised against Tier 0. The MI300X droplet is the *right* instrument, not a fallback. And the thesis is measurement-heavy but compute-light, so 50 GPU-hours is ample.

**Consequences:** claim the credit when G1 is scripted and ready to run unattended; all pure-Python tasks (1, 4–16) still run here unchanged; the cross-vendor angle moves to "out of scope unless separately funded."

## Live evidence found while measuring G2 — the thesis, on this laptop

Ollama 0.33.1 is installed and running on `127.0.0.1:11434`. Its own server log:

```
msg="dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1"
     id=0 library=Vulkan compute=0.0 name=Vulkan0 description="AMD Radeon(TM) Graphics"
msg="inference compute" id=cpu library=cpu name=cpu total="15.4 GiB" available="7.5 GiB"
msg="vram-based default context" total_vram="0 B"
```

It **detected** the AMD GPU through Vulkan, then dropped it and ran everything on CPU. `total_vram="0 B"`. The user is never told: the only notice is one INFO line in a log file nobody opens, and the opt-out env var (`OLLAMA_IGPU_ENABLE=1`) is undiscoverable unless you read that log.

This is SILENTPATH's exact thesis — *you are not on the path you think you are, and nothing tells you* — reproduced on consumer hardware before a line of the tool was written. It is a free, honest demo asset that needs no MI300X, and it widens the story from "AMD datacentre serving stacks" to "the whole AMD stack, including the laptop you already own."

**Measured, not just logged — the CPU-path baseline (`qwen2.5:0.5b`, seed 0, temp 0, 120 tokens, 3 repetitions on the primary `127.0.0.1:11434` instance):**

| Run | eval tokens | eval seconds | tok/s |
|---|---|---|---|
| 1 | 104 | 2.12 | 49.10 |
| 2 | 104 | 2.05 | 50.78 |
| 3 | 104 | 2.04 | 50.91 |
| **mean** | | | **50.26 tok/s** |

A second ollama instance was started on port 11500 with `OLLAMA_IGPU_ENABLE=1` to force the path ollama otherwise drops, purely to confirm the iGPU is real, not to leave a second server running: it logged `type=iGPU total="8.0 GiB"` where the default instance logged `total_vram="0 B"` — direct confirmation that the silent-drop path and the "actually try the GPU" path are observably different configurations on this exact machine. **The paired throughput comparison (iGPU vs the CPU baseline above) was started but interrupted before completion**, and the temporary instance was stopped afterward (primary instance on 11434 untouched throughout). Redo it if a same-machine before/after number is wanted:
```
$env:OLLAMA_IGPU_ENABLE=1; $env:OLLAMA_HOST="127.0.0.1:11500"; ollama serve
# then repeat the same /api/generate call against port 11500
```
gfx90c is unsupported for the ROCm path this project studies (see G2 table above), so this iGPU number would only ever support the *ollama/Vulkan* silent-fallback narrative, not the vLLM/ROCm backend-plurality thesis D1–D6 are built around. It is corroborating colour for the pitch, not a substitute for the MI300X measurement.

## Do first, in this order

1. **G1 — get the divergence result.** Needs the MI300X droplet now that G2 is BLOCKED. This is the load-bearing empirical claim under everything else. If backends agree bitwise, pivot rather than continuing to build on the correctness thesis.
2. **G3 email — fire it the same day.** Ten minutes, runs asynchronously. Whether one codebase can serve multiple themes as distinct projects decides the entire XP strategy.
3. **Onboarding XP (150) + Discord.** Free, immediate, and starts the scoring-lag clock.
4. **Do NOT claim the $100 credit.** Not until the harness runs unattended end to end (Task 14 Step 5 of the implementation plan).

## Key facts to not forget

- **Prizes are a race, not a ranking.** $2,500 / $1,500 / $1,000 to the *first three* to reach Legend. Legend = XP **plus** a final challenge that unlocks late.
- **Scoring lag up to 9 days** (Fridays, covering through the prior Wednesday). Effective creditable deadline ≈ **Nov 21**, not Dec 1.
- **XP per fully-decorated project ≈ 1,105**: demo 150, complete 200, open-source 200, built-with-AMD-tech 250, 3 milestones 300, repo 5. A course is 150. Never leave the five checkboxes unticked — most participants forfeit 500–700 XP per project by forgetting them.
- **Credits expire 30 days after claiming.** $100 ≈ 50 hrs on one MI300X at $1.99/hr. MI300X only — no MI325X/MI355X provisioned.
- **Dev Cloud is DigitalOcean GPU Droplets**: root SSH, Docker, and an unfirewalled public port, so a public OpenAI-compatible endpoint is ~30 minutes of work.
- **Field is nearly empty**: 2,383 approved, 6 completed submissions. The opponent is non-completion, not other entrants.
- **Act III collides**: AMD Developer Hackathon ACT III runs **Oct 12–18, 2026**, separate $5,000+ pool, and is worth +300 XP ("participate in an AMD hackathon"). Plan around it; do not be surprised by it.
- **Do not build**: kernel generators/optimisers (AMD's own GEAK + AgentKernelArena + a $20M startup), CUDA→HIP porting (already submitted here), generic agents (~79% of a prior AMD hackathon).
- Local dev box is **Windows 11**; vLLM/SGLang on ROCm are **Linux-only**.

## Timeline

Bi-weekly challenge drops estimated at ~Sep 1, Sep 15, Sep 29, Oct 13, Oct 27, Nov 10.

| Week | Dates | Target |
|---|---|---|
| W1 | Sep 10–16 | **G1, G2, G3.** Onboarding XP. Ship one deliberately small thing to calibrate the scoring lag |
| W2 | Sep 17–23 | **D1** — probe + MCP server (theme 1). First fully-decorated submission |
| W3 | Sep 24–30 | AI Academy courses 1/3/6 (they double as the tech for themes 1/5/6). Build divergence harness |
| W4 | Oct 1–7 | **D2** — OCR/VLM divergence, the invoice demo (theme 2). The narrative centrepiece |
| W5 | Oct 8–14 | Claim credits **only if** the harness is ready. First MI300X confirmation sweep |
| W6 | Oct 15–21 | **ACT III (Oct 12–18)** — +300 XP and a second prize pool. Enter with a SILENTPATH slice |
| W7 | Oct 22–28 | **D3** — RAG advisor over the ROCm corpus (theme 3) |
| W8 | Oct 29–Nov 4 | **D4** — support-matrix scraper (theme 4). Feeds D3's corpus |
| W9 | Nov 5–11 | **D5** — multi-agent triage loop (theme 5) |
| W10 | Nov 12–18 | **D6** — GRPO configuration policy (theme 6). The final-gate hedge |
| W11 | Nov 19–25 | **Hard creditable deadline ~Nov 21.** Final challenge when it unlocks. Buffer |
| W12 | Nov 26–Dec 1 | Reserve. Assume nothing shipped here gets credited in time |

Cross-vendor (NVIDIA vs MI300X) records are collected opportunistically throughout — the shared seam makes them nearly free — and folded into whichever deliverable is current.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Backends agree bitwise; no correctness finding | G1 in Week 1. Cost half of the thesis stands alone on documented 5.5x evidence. Null result is itself the first such measurement on AMD |
| Divergence exists but is only floating-point noise, never changes a decision | Report the three levels separately by construction; only decision-level counts as a finding. Never conflate them |
| Local AMD GPU unsupported under WSL2 | **Realized, not hypothetical**: G2 measured BLOCKED on 2026-09-10 (gfx90c iGPU, no NPU, no `/dev/kfd`). No local ROCm fallback exists on this machine — all GPU work moves to the MI300X droplet; pure-Python tasks (1, 4–16 of the implementation plan) are unaffected |
| Organizers reject one-codebase-many-themes | G3 in Week 1. Fallback: concentrate into D1+D2 as one deep project, re-plan XP around courses and community |
| Legend threshold unreachable at 15–25 hrs/week | Read the number off the dashboard immediately; recompute before committing to 6 deliverables |
| Final challenge takes longer than the remaining window once unlocked | D6 scheduled early so the GRPO skill exists before the gate opens |
| Credits expire unused, or burn out early | Do not claim until the harness runs unattended; hard spend ceiling enforced in the runner; content-addressed cache so no cell is recomputed |
| AITER's 2–16x measurement variability swamps timing results | Repetition counts and variance reported on every timing claim; no single-shot speed results |
| Act III (Oct 12–18) eats the D3/D4 window | Enter Act III with an existing SILENTPATH slice rather than a new build |
| Reviewer sees six submissions as one relabelled project | Each deliverable gets its own README, demo and result, independently useful. G3 confirms the organizers agree |
| Scope creep into kernel writing or a public dataset site | Both explicitly out of scope in the spec; the council overruled the latter unanimously |

## Council decision log

- **2026-09-10:** Council convened on Idea A (silent-path differential harness), Idea B (port batch-invariance research to ROCm), Idea C (GRPO-learned budget-aware agent pipeline). **Unanimous for A**, with B and C rejected outright — A is the only candidate that decomposes into independently-submittable artifacts, which is what a per-project XP race rewards. Peer review was unanimous that the Executor's response was strongest and the Expansionist's the biggest blind spot. Adopted: build the detector first but lead every demo with the OCR divergence story; drop rocprofiler-sdk for `amd-smi` + torch HIP events; retain the cross-vendor comparison (cheap, and the most defensible artifact) but reject the dataset site, mid-race preprint and devrel outreach as XP-free scope. Peer review caught that no advisor had sized the finish line — the Legend XP threshold and the final challenge's own duration remain unknown and are now G3.

## Open questions

- What is the numeric XP threshold for Legend, and is it reachable at 15–25 hrs/week? (Dashboard, after enrollment approval.)
- What does the final challenge actually require, and how long does it take once unlocked? True deadline = unlock date − completion time − 9 days.
- Are the XP line items per-project or once-only? "Create your **first** project" is worded once-only while the others are not, which supports the per-project reading — but the whole strategy hinges on it.
- How close is anyone else to Legend? (Live leaderboard — free information about whether this race is even contested.)
- Which local AMD GPU is in the machine, and is its gfx target on the WSL2-supported list?
- Does Act III participation credit the +300 XP "participate in an AMD hackathon" line item in this program?
