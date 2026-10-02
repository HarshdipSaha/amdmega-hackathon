# Effort 013: SOURCEBOUND (Mini-Challenge 3, RAG with exact citations)

## Metadata

- **Effort ID:** `013`
- **State:** `complete` (release checks and checkpointed notebook rehearsal complete; competition submission remains user-owned)
- **Updated:** 2026-10-02
- **Branch:** `feat/sourcebound-mc3` (pushed; NOT merged to master)
- **Spec:** `docs/MINI_CHALLENGE_3_SPEC.md`. **Plan:** `docs/superpowers/plans/2026-10-01-sourcebound-mc3.md` (22 tasks, reviewed and approved).
- **Memory:** `mc3-rag-spec-status.md` in the Claude project memory.

## Done (Tasks 1-18, 20; 21 started)

- Code, tests, CI, remote GPU driver: 237 tests pass on the repo (162 old + 75 new). CI `MC3 contract (CPU)` green: unit job plus grader-isolation job (root, no network, DAC_OVERRIDE dropped; kit 10/10).
- GPU results on W7900D (details in `results/mc3-session1.md` and `results/mc3-e4-holdout-scale.md`):
  - 4B dev reader: kit 10/10, dev 64/64. 8B: dev 63/64, kit 10/10. 9B: dev 62/64, kit 10/10.
  - **Release reader chosen: Qwen3-VL-8B-Instruct** (rev `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`); embedder Qwen3-Embedding-0.6B (rev `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`).
  - Holdout (run ONCE, 8B): 32/32. Frozen at tag `mc3-freeze-1` (later commits only touch CI and the deadline guard).
  - Scale (300 files): initial run indexed in 63.7 s with a 22.7 s maximum question; `SB_MIN_CALL_S` default was raised 6 -> 10. The checkpointed fresh-worker re-measurement was 63.65 s index and 15.303 s maximum question.
- Release `rc1`: GitHub Actions publish and image-check jobs passed. The notebook used the documented local-layer fallback because the temporary `crane` download was not reliable; kit rehearsal was 10/10 with 19.4 GiB peak VRAM, supervisor recovery returned `E7731`, and fresh scale timing at `SB_MIN_CALL_S=10` was 63.65 s index / 15.303 s worst query / 21.01 GiB peak VRAM.
- Secret `MC3_IMAGE` set in GitHub (`docker.io/harshdipsaha/sb-r7k2-private`, no tag). The Docker Hub repo must stay PUBLIC for the grader. Never commit the reference.

## Completion notes

1. The image reference still must be submitted by the user on the LabLab MC3 form; no submission is claimed here.
2. The branch remains `feat/sourcebound-mc3` and is not merged to master.

## Practical notes for the next session

- GPU: `export HEADLESS=1`; if `NOT_SIGNED_IN`, the USER runs `! node login.js` from `tools/amd-gpu`. Quota left after this session about 2 h; ALWAYS end with `node tools/amd-gpu/remote-mc3.js end`.
- Each new pod session needs: `sync` (run twice; the first call after a launch can fail), `model Qwen/Qwen3-Embedding-0.6B embedder tmp`, `data eval_mc3/data/dev`, `setup`, then `worker-start` (the pod cannot drop DAC_OVERRIDE, so no `nodac`).
- Never put the process name `sourcebound` in a `gpu.js sh` command (self-kill); use `eval_mc3/worker_ctl.sh`.
- `eval_mc3/data/` and `results/*.jsonl` are gitignored; regenerate data with `python eval_mc3/generate_corpus.py` (seeds: dev 101-104, holdout 901-902, scale 2001-2025).
- Do not tune on the holdout again; any change after `mc3-freeze-1` that affects answers needs a fresh holdout (new seeds).
