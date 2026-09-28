# Effort 012 audit trail

## 2026-09-28 — Resume and execute W7900D plan

- **Authorization:** The user asked to execute the written GPU plan. That instruction authorized implementation and the GPU sessions; no external submission was authorized or performed.
- **Scope:** Continued existing Effort 012; did not rerun inception because the project baseline exists. Selected the PyTorch SDPA route after vLLM was deferred under the plan’s stop/pivot guidance.
- **Construction:** Implemented producer, CLI, MCP query functions, workload matrices, evidence capture, and corrected absent log-probability output to `null`. Commits: `08b530a`, `800ab18`, `a2895a9`.
- **Verification:** GPU-free CLI smoke succeeded. The full suite previously reported 161 passing tests; a fresh rerun is pending for this status update. GPU sessions A/B/B2/C produced the evidence bundle under `docs/evidence/silentpath-w7900/`.
- **Interpretation:** Forced Flash and Math were distinguished by profiler operators. No positive silent fallback was observed. Reversed-order decode ratios reproduced at 1.292x and 1.299x for the two prompts; only the short sequence diverged, first at generated token 65. No chosen-token log-probabilities were recorded. These are workload-specific observations.
- **Operations:** The W7900 notebook artifacts were retrieved and the pod was stopped (`not_found`). `/workspace` space constraints required model weights and large traces to use pod scratch; full archives were retrieved and hashed.
- **Decision:** Keep the effort `in-progress`: clean-checkout packaging, project-root README reconciliation, short demo, final branch review/publication, and competition submission remain. No dashboard-only prize mechanics are claimed as verified.

## 2026-09-28 — AI-DLC status reconciliation

- Added this requirements delta and audit trail, updated the effort state, and aligned the registry and project status with measured evidence.
- Approval gates were satisfied by the user’s direct instruction to execute the plan. No new scope requiring a separate approval gate was introduced.
- Completion remains open until the release requirements above are verified; the user has not requested a pause or abandonment.
- Fresh verification at this stage: `python -m pytest -q` completed with **161 passed in 15.96s**.

## 2026-09-28 — Release packaging and pinned reproduction

- Replaced the ROADREAD-only root README with a dual-track SILENTPATH/ROADREAD entry page, added MIT `LICENSE`, pinned SILENTPATH dependency files, and added the installable `silentpath` CLI entry point.
- Added explicit model revision propagation to both Transformers loads and pinned it in the W7900 configs. The new test failed before implementation, then passed afterward.
- Added a PowerShell demo walkthrough and ran it successfully: four fake records, two comparisons, and both MCP query functions reading the same store.
- Fresh full suite: **162 passed in 20.19s**. An isolated Python 3.12 environment built and installed the wheel, invoked the console command outside the repository, ran four fake cells, and rendered the report.
- The public Lablab live page currently displays “Submissions open”; that does not establish the signed-in form or D1 category eligibility. A local submission draft is at `docs/submission-d1-draft.md`; no LabLab platform draft or submission was created. The timed recording script is at `docs/demo-script.md`; video and account-specific review remain open.
- Published the feature branch to `origin/feat/silentpath-w7900` at `c72683a`, so the README and draft source links resolve to the release contents. The participant account remains signed out in the visible browser.
