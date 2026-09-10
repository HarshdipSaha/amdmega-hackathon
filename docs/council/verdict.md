# LLM Council — Chairman's Verdict

Convened 2026-09-10. Inputs: `round1-advisors.md` (5 independent advisors), `round2-peer-reviews.md` (3 anonymous peer reviews).
Question: which of three candidate projects should a solo developer commit to for the 11 remaining weeks of the Lablab x AMD AI Academy Challenge, racing to be one of the first three to reach Legend rank.

---

## 1. Where the council agrees

Unanimous, 5 of 5 advisors, with no dissent in peer review:

- **Commit to Idea A.** Skip B and C entirely. Not one advisor argued for either alternative.
- **Idea A wins for a structural reason, not an intellectual one.** A is the only candidate that decomposes into several independently-submittable artifacts sharing one measurement core. B is one paper. C is one pipeline. In a contest that pays per decorated project, decomposability *is* the argument. The First Principles Thinker put it most bluntly: novelty earns zero XP.
- **Do not claim the $100 credit yet.** Credits expire 30 days after being claimed; the program runs to Dec 1. Claim them the day the benchmark harness is scripted and ready to run unattended. Three advisors raised this independently.
- **Ship early, ship weekly.** The 9-day reporting lag punishes anyone who back-loads. Get one small thing submitted in week 1 purely to calibrate how the lag actually behaves.
- **Verify the multi-theme submission rule before writing code.** Four of five advisors flagged this as the load-bearing assumption. All three reviewers agreed it is the single highest-leverage unknown.

## 2. Where the council clashes

**Which tier leads — and this is a real disagreement, not a misunderstanding.**

The Executor says Tier 0 (the backend detector) is the product and everything else is decoration. The Outsider says the opposite: the detector fails the stranger test, and the buried OCR demo — *the same scanned invoice returns a different dollar amount depending on a setting you never chose* — is the entire project. The Expansionist says neither, and wants the cross-vendor NVIDIA-versus-MI300X conformance artifact to lead.

They are not actually in conflict once you separate build order from narrative order. Tier 0 is the **engineering substrate** — nothing else can be measured without it, so it must be built first. The OCR divergence is the **narrative** — it is what a judge, a reader, or an AMD engineer will remember. Build the Executor's thing; present the Outsider's thing. The Expansionist's cross-vendor angle is genuinely the most defensible artifact of the three, but it is a *result*, not a starting point.

**Scope expansion — and here the council does not clash so much as overrule one member.**

The Expansionist proposed a living public dataset site, an arXiv preprint mid-race, and AMD devrel amplification. All three reviewers independently rejected this as anti-de-risking dressed as upside: none of it earns XP, all of it costs time, and the payout function is strictly "first three to Legend." Overruled for the duration of the race. Revisit after Dec 1, when the artifacts exist and the clock is gone.

**Whether the core result is even real.**

The Contrarian and the Outsider both warn the thesis may be a nothingburger: different kernels producing different bits is expected floating-point behaviour, not a finding. This is the sharpest objection raised against the recommended idea, and it is correct as stated. The finding only exists if divergence flips a **decision** — a digit in an extracted invoice total, a retrieved document, an answer. That is an empirical question with an unknown answer, and no amount of planning substitutes for measuring it.

## 3. Blind spots the peer review caught

The reviewers converged on one omission that all five advisors shared, approached from three angles:

- **Nobody sized the finish line.** The council optimised XP-per-project and projects-per-week without ever asking what Legend costs in XP, or whether that total is reachable at 15–25 hrs/week inside the lag-adjusted window. This is planning a race without knowing its distance.
- **Nobody sized the final gate itself.** If Legend requires XP *and* a final challenge that takes real time once unlocked, the true deadline is *unlock date − completion time − 9-day lag*, not simply "before Dec 1."
- **Nobody proposed a fallback** if the organizers rule that one codebase cannot be submitted across multiple themes.
- **Nobody suggested reading the leaderboard** to see how close other racers already are — free information about whether this race is even contested.

## 4. The recommendation

**Commit to Idea A, structured as a tiered submission factory, and gate the commitment behind three week-1 verifications.**

The strategic logic is sound and unanimously supported: in a field where 2,377 of 2,383 approved participants have shipped nothing, the opponent is non-completion, not other entrants. A decomposable project that produces a shippable artifact every two weeks beats a single deep artifact that lands in November.

But three facts could each independently invalidate the plan, and all three are cheap to check:

1. **Can one codebase legitimately serve multiple mini-challenge themes as distinct projects?** Ask the organizers in writing. If no, the XP multiplier vanishes and the calculus changes.
2. **Does ROCm actually work on the local AMD hardware under Windows/WSL2?** vLLM and SGLang on ROCm are Linux-only, and WSL2 supports a narrow gfx list. If the local card is not supported, "Tier 0 runs locally, no cloud needed" is fiction and the entire de-risking story collapses onto 50 cloud hours.
3. **Do the backends actually produce different numbers?** A roughly 40-line script — set `VLLM_ATTENTION_BACKEND` to each value, same prompt, same seed, dump logits, compare — either kills the thesis or hands over the headline result. It requires no profiler at all.

Two engineering calls the council settled: **drop rocprofiler-sdk** (LD_PRELOAD C++ injection will consume two weeks for a hackathon-grade need) in favour of the `amd-smi` Python package plus torch HIP events; and **build the detector first but lead every demo with the divergence story.**

**A chairman's addition the council did not address.** "Reskin one repo six times" is the wrong framing, and not only because it may be against the rules. Six genuinely distinct deliverables that share a measurement core is ordinary good engineering, and defensible to anyone who looks. The same artifact resubmitted six times with new labels is neither, and a reviewer who notices will discount all six. Build the former. The distinction should survive contact with a skeptical judge, which is exactly the test the organizer question in item 1 is really probing.

## 5. The one thing to do first

**Get the divergence result.** Before any spec, agent, MCP server, or GRPO work: same prompt, same seed, each attention backend, dump the logits, compare them. It is an afternoon's work and it is the load-bearing empirical claim under every tier above it. If the backends agree bitwise, pivot on day 11 rather than day 60.

Send the organizer question the same day — it costs ten minutes and runs asynchronously while the measurement happens.
