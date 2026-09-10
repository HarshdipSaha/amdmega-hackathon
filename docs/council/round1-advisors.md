# LLM Council — Round 1: Independent Advisor Responses

Convened 2026-09-10 on the choice of project for the Lablab x AMD AI Academy Challenge.
Five advisors answered the same framed question independently, with no knowledge of each other's answers.
Peer reviews: `round2-peer-reviews.md`. Verdict: `verdict.md`.

Anonymisation map used for the peer-review round (randomised so reviewers had no positional bias):

| Letter | Advisor |
|---|---|
| A | The Expansionist |
| B | The Executor |
| C | The Contrarian |
| D | The First Principles Thinker |
| E | The Outsider |

---

## The framed question

A solo developer, day 10 of 92, racing for cash in a program where $5,000 goes to the **first three** people to reach "Legend" rank (not the top three by XP). Legend requires XP plus a final challenge that unlocks late. XP is credited weekly with up to a 9-day lag. Six bi-weekly themes (serving/MCP/GPU-consumption, OCR, RAG, web-scraping, multi-agent SWE, novel game + fine-tuning). One fully-decorated project ≈ 1,105 XP versus 150 per course. 2,383 approved participants, 6 completed submissions.

Constraints: 15–25 hrs/week, ~11 weeks. $100 AMD credit ≈ 50 hrs on one MI300X (credits expire 30 days after claiming), plus a local NVIDIA GPU and a local AMD GPU / Ryzen AI laptop on Windows 11. Ambitious but de-risked. 

Three candidates: **A** = differential harness measuring which AMD kernel backend was silently selected, what it costs in GPU-seconds, and whether backends produce different numbers; **B** = port the batch-invariance/determinism research line to ROCm; **C** = budget-aware agentic pipeline with a GRPO-learned allocation policy.

---

## Response A — The Expansionist

Commit to A — but stop thinking of it as one project. The council is evaluating which single idea is best. Wrong frame. This is an XP race with a 9-day reporting lag and a final challenge that unlocks late, which means the binding constraint is *how many fully-decorated 1,105-XP artifacts you can ship before the lag window closes*. Idea A is the only candidate that is natively a **factory**: five tiers, each independently demoable, open-sourceable, AMD-tech-decorated, and mapped to a different bi-weekly theme. B and C are one project each. A is potentially five at ~1,105 XP apiece. That alone decides it.

**What's being undervalued: the dual-vendor local hardware.** Everyone is treating the differential as *across AMD backends*. The bigger artifact is **cross-vendor**: same model, same input, NVIDIA vs MI300X, numbers side by side. The Kernel Contracts paper literally says no artifact exists to arbitrate that dispute. He can build the first one because he owns both sides. That's not a hackathon demo — that's a conformance suite AMD's own team lacks and cannot easily produce (they don't publish NVIDIA comparisons).

**Adjacent upside nobody's costed:** ship it as a *living public dataset* — a continuously-updated divergence + gfx-support matrix site, fed by the Tier 3 scraper. That converts a one-shot submission into a citable resource, an arXiv preprint mid-race, and AMD devrel amplification. Visibility from the sponsor is the cheapest accelerant in a race where 2,377 of 2,383 entrants have shipped nothing.

**Sequencing:** don't claim the $100 credit yet — 30-day expiry, and Tiers 0–1 run local. Ship Tier 0 by week 2 to start the lag clock.

---

## Response B — The Executor

**Commit to A — but you have the tiers backwards. Tier 0 is the product; everything else is decoration.**

Monday morning, three things, in this order:

1. **Ask the organizers, in writing, today:** can one repo be submitted across multiple mini-challenge themes with different framing, or does each theme need a distinct project? Six decorated projects ≈ 6,600 XP; one repo reskinned six times is the entire race strategy or an instant disqualification. Ten minutes of email decides your next eleven weeks. Nobody in this brief has checked.

2. **Verify your local AMD box can actually run the thing.** vLLM/SGLang on ROCm are Linux-only and you're on Windows 11. ROCm-on-WSL2 supports a narrow gfx list. If your local AMD card isn't on it, "Tier 0 runs locally, no cloud needed" is fiction and your whole de-risking collapses onto 50 hours of MI300X. Run `rocminfo` under WSL before you write a line of code.

3. **Do NOT claim the $100 credit yet.** 30-day expiry from claim. Claim it the day your benchmark harness is scripted and ready to `bash run_all.sh`.

**What to actually build first:** the numerical divergence check needs no profiler at all. Set `VLLM_ATTENTION_BACKEND` to each of the four values, same prompt, same seed, dump logits, `np.allclose`. That's a 40-line script and a screenshot that either kills the idea or hands you the headline result by Friday of week 1. Get the finding before you build the agent, MCP server, or GRPO anything.

Drop rocprofiler-sdk entirely — LD_PRELOAD C++ injection will eat two weeks. `amd-smi` Python plus torch HIP events gets you GPU-seconds well enough for a hackathon.

Skip C. Skip B. Ship weekly; the 9-day lag punishes anyone who back-loads.

---

## Response C — The Contrarian

**Idea A's coverage claim is unverified and probably false, and everything rests on it.**

The whole case for A is "one deep artifact, six theme submissions." Nobody in this brief has read the rule that matters: does the XP engine award "complete project" + "open source" + "built with AMD" *repeatedly* for the same repo re-skinned six times? Bi-weekly mini-challenges almost always require a distinct deliverable, and reviewers who see the same harness six times will grade it once. If that's true, A doesn't cover six themes — it covers one, and the 1,105-XP-per-project math that makes A look efficient collapses. **Verify this before writing a line of code.** It is a 30-minute question with an 11-week payload.

**Tier 0 is not the safe fallback it's advertised as.** The local AMD GPU is not an MI300X. AITER, MLA backends, and most of vLLM's ROCm attention plurality don't exist on consumer RDNA/gfx110x targets. The backend zoo you claim to detect may have exactly one occupant locally. So the "always submittable, no cloud needed" floor is unproven, and the 50 cloud hours have to cover a model × shape × version × backend matrix *plus* repetitions for variance — that's the compute-light claim doing a lot of work it hasn't earned.

**Tier 1 risks being a nothingburger.** Different kernels producing different bits is expected floating-point behavior, not a finding. The finding only exists if divergence flips a *decision* — an OCR digit, a RAG answer. Nature may not cooperate. You cannot plan around a result you haven't seen.

**What's missed:** with 6 submissions from 2,383 people, your competitor is not other entrants, it's non-completion — and the 9-day lag plus late-unlocking final challenge means velocity and queue position, not depth, decide the cash. Also: don't claim the credit yet; 30-day expiry means claim it the week you actually need the MI300X.

---

## Response D — The First Principles Thinker

You're asking "which idea is best" when the payout function is "first three across a line." **Novelty earns zero XP.** Re-read the economics: XP comes from demo, complete, open-source, built-with-AMD, 3 milestones, repo. Not one of those rewards being interesting. Interestingness only matters for not being rejected and for the separate judged prizes — which are not the money you said you're racing for.

So the real question is: **what maximizes fully-decorated submissions per week?** The answer is a substrate that slices into six separately-submittable artifacts. Idea A is the only candidate that does this structurally — not because the thesis is better, but because backend-detection, OCR divergence, ROCm RAG, the version-matrix scraper, and the multi-agent loop are *already six repos* sharing one measurement core. B is one paper (one submission). C is one pipeline (one submission). At ~1,105 XP each, A is a 6x XP multiplier over B or C. That's the entire argument.

Three things being missed:

1. **The 9-day lag moves your deadline to ~Nov 21.** Everything shipped after that may not be credited before three people cross. Backload nothing.
2. **Verify before committing** that one codebase can be submitted to multiple mini-challenges as distinct projects. If it can't, A's whole advantage evaporates and this decision inverts. Also find the actual Legend XP number — you're planning a race without knowing the finish line's distance.
3. **Don't claim the $100 yet.** 30-day expiry. A is compute-light and local-first by design; claim it for the one week you genuinely need MI300X.

Commit to A. But commit to it as a submission factory, not as a research program.

---

## Response E — The Outsider

**The pitch fails the stranger test.** "Detects which kernel backend the stack silently selected and whether backends differ numerically" — I have no idea why I should care. Read it as an outsider: it's a diagnostic tool for a bug nobody knows they have, in a stack most people never configure. Compare with the buried Tier 2: *the same scanned invoice returns a different dollar amount depending on a setting you never chose*. That one sentence is the whole project. It needs no ROCm knowledge, it's alarming in three seconds, and it makes the boring backend-detector suddenly matter. You have the winning demo ranked fourth in your own tier list.

**Commit to A, but invert it.** Lead with the divergence artifact, treat the detector as supporting evidence, not the headline.

**What's being missed, and it's bigger than the idea choice:** the entire Tier 1–4 stack rests on an unverified assumption — that the backends *do* produce different numbers. If they agree to the last bit, your thesis evaporates in week 8. That check costs one afternoon on the local AMD GPU. Run it before writing another line of spec. If it's a null result, you pivot on day 11 instead of day 60.

Second miss: "maps to all six themes" is retrofitting, and I distrust it. You are describing one project wearing six hats. But XP is awarded per decorated project — six shallow projects beat one deep one that arrives late.

Third: 6 completed submissions out of 2,383. This is not a quality contest, it's an attendance contest with a 9-day reporting lag. Ship something small in week 1 to calibrate the lag before it can cost you the race.
