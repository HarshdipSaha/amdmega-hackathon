# LLM Council — Round 2: Anonymous Peer Reviews

Three reviewers each saw all five advisor responses, anonymised as A–E (mapping in `round1-advisors.md`), and answered the same three questions. They did not see each other's reviews.

**Result: unanimous on both questions 1 and 2.** All three ranked Response B (the Executor) strongest and Response A (the Expansionist) as the biggest blind spot. Their answers to question 3 converged on the same omission from a different angle each time.

---

## Reviewer 1

**1. Strongest: B.** It's the only one that converts analysis into an immediate, ordered action list with concrete commands (`rocminfo` under WSL, `VLLM_ATTENTION_BACKEND` + `np.allclose` script) and correctly flags the Windows/WSL/Linux-only risk that A, D, and E never mention — a real de-risking gap since the developer's local box runs Windows 11. It also gives the sharpest build-order call (kill rocprofiler-sdk, use amd-smi + torch events) with a reason.

**2. Biggest blind spot: A.** It leans hardest into the "submission factory" and "living dataset / arXiv preprint / AMD devrel amplification" upside without ever flagging the load-bearing assumption underneath it — whether the platform even allows one repo to count as six submissions. C, D, and E all surface that risk explicitly; A never does, despite building its entire multiplier argument on it. It also never addresses the Windows/local-AMD-viability question B and C raise.

**3. All five missed:** nobody asks **what "Legend" actually costs and whether the XP timing math closes** — the 9-day lag plus late-unlocking final challenge means the real question is "how many decorated projects fit before ~day 83, and does even a maximal-A strategy generate enough total XP to reach Legend at all," not just relative speed. Also nobody proposes a fallback if the six-submission premise is rejected by organizers — no one says what A becomes as a single strong project.

---

## Reviewer 2

**1. Strongest: B.** It's the only response that converts every claim into an immediate, falsifiable action with a deadline — file the multi-theme question today, run `rocminfo` under WSL before writing code, ship the 40-line `np.allclose` divergence check by Friday of week 1. It also correctly reorders the tiers (divergence result before agent/MCP/GRPO scaffolding) and gives the sharpest concrete kill-test for rocprofiler-sdk (drop it, use amd-smi + torch HIP events). D and A make the same "verify multi-theme submission" point but leave it as a to-do; B turns it into a script.

**2. Biggest blind spot: A.** It proposes an arXiv preprint, a "living public dataset" site, and AMD devrel amplification as accelerants — none of which earns XP or cash, and all of which cost time a 15–25 hr/week solo dev racing a 9-day lag cannot spare. It's optimizing for academic/visibility upside in a contest whose payout function (per D) is strictly "first three to Legend." A is also the only response that never questions whether the six-theme mapping is real before spending on it.

**3. All five missed:** nobody asked **what the "final challenge" near the end actually requires or how long it takes to complete once unlocked**. If Legend needs XP *and* a final challenge that itself takes non-trivial time, the real deadline is "final-challenge-unlock date minus completion time minus 9-day lag," not simply Nov 21 (D's number). Racing for cash without sizing that last gate is planning blind on the actual finish line, not just its distance.

---

## Reviewer 3

**1. Strongest: B.** It's the only one that converts the brief into falsifiable, dated actions: verify multi-theme submission rules by email today, run `rocminfo` under WSL2 before writing code (directly testing whether "Tier 0 runs locally" is even true on Windows), and ship a 40-line `VLLM_ATTENTION_BACKEND` + `np.allclose` script by Friday week 1 to get the headline finding before building anything else. It also correctly drops rocprofiler-sdk for `amd-smi`, resolving the stated tooling friction. Most execution-ready.

**2. Biggest blind spot: A.** It asserts the "one harness = six submittable projects" multiplier as settled fact ("That alone decides it") without ever flagging it as an unverified rules question — the exact thing B, C, and D independently identify as the load-bearing assumption. A then piles on scope (public dataset site, arXiv preprint, AMD devrel outreach) for a 15–25 hr/week solo dev on a hard 11-week clock — that's anti-de-risking dressed up as upside.

**3. What all five missed:** none checked whether the required Legend XP total is even **reachable** at 15–25 hrs/week within the lag-adjusted window (~Nov 21 per D) — they optimize project count and XP-per-project but never do the arithmetic against a known finish line. None discuss a fallback if organizers reject repo-reskinning (pivot budget to Idea C), and none mention **checking the leaderboard to see how close other racers already are** to Legend.
