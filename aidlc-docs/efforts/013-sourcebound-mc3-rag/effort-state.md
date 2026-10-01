# Effort 013: SOURCEBOUND (Mini-Challenge 3, RAG with exact citations)

## Metadata

- **Effort ID:** `013`
- **State:** `in-progress` (release rc1 pushing; rehearsal and submission remain)
- **Updated:** 2026-10-01
- **Branch:** `feat/sourcebound-mc3` (pushed; NOT merged to master)
- **Spec:** `docs/MINI_CHALLENGE_3_SPEC.md`. **Plan:** `docs/superpowers/plans/2026-10-01-sourcebound-mc3.md` (22 tasks, reviewed and approved).
- **Memory:** `mc3-rag-spec-status.md` in the Claude project memory.

## Done (Tasks 1-18, 20; 21 started)

- Code, tests, CI, remote GPU driver: 237 tests pass on the repo (162 old + 75 new). CI `MC3 contract (CPU)` green: unit job plus grader-isolation job (root, no network, DAC_OVERRIDE dropped; kit 10/10).
- GPU results on W7900D (details in `results/mc3-session1.md` and `results/mc3-e4-holdout-scale.md`):
  - 4B dev reader: kit 10/10, dev 64/64. 8B: dev 63/64, kit 10/10. 9B: dev 62/64, kit 10/10.
  - **Release reader chosen: Qwen3-VL-8B-Instruct** (rev `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`); embedder Qwen3-Embedding-0.6B (rev `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`).
  - Holdout (run ONCE, 8B): 32/32. Frozen at tag `mc3-freeze-1` (later commits only touch CI and the deadline guard).
  - Scale (300 files): index 63.7 s, worker start 18 s; max question 22.7 s, so `SB_MIN_CALL_S` default raised 6 -> 10 (NOT yet re-measured on GPU).
- Secret `MC3_IMAGE` set in GitHub (`docker.io/harshdipsaha/sb-r7k2-private`, no tag). The Docker Hub repo must stay PUBLIC for the grader. Never commit the reference.

## Remaining

1. **Check the rc1 release run.** Pushing `release/rc.env` triggers `.github/workflows/mc3-release.yml` (dispatch is impossible: workflow is not on master). Run: `gh run list --workflow mc3-release.yml --limit 3`, then `gh run view <id> --log-failed`. Earlier failures were YAML/env-file bugs, all fixed; the run after commit `2ea4cc9` is the first to reach `publish.sh`. Expected: `published tag rc1`, then `check_image.py` prints `"ok": true`. Possible new failures: `crane mutate --workdir`/`--cmd` flags, runner disk, Docker Hub push size/timeouts (lower `LAYER_MAX` in `release/build_layers.py`), base image has an ENTRYPOINT (publish.sh stops on purpose). To retrigger, change `release/rc.env` (comment lines are allowed) and push.
2. **Task 21 Step 4: rehearse on the pod.** `export MC3_IMAGE='docker.io/harshdipsaha/sb-r7k2-private:rc1'` then `node tools/amd-gpu/remote-mc3.js sync` and `node tools/amd-gpu/remote-mc3.js rehearse` (use `rehearse local` if the pod cannot reach Docker Hub; re-download the model with `model Qwen/Qwen3-VL-8B-Instruct reader tmp` first because `/root/models` is ephemeral). Expect kit 10/10, torch `+rocm`, liveness query answers `E7731`, supervisor alive. Also confirm the 8B at `SB_MIN_CALL_S=10` keeps max question under about 25 s (rerun `suite eval_mc3/data/scale scale-final2` after a fresh `worker-start`).
3. If the image size fails the check (`size_ok`), fall back to the 4B reader (8.9 GB; it scored 64/64 dev) by changing `release/rc.env`.
4. **Submit.** The user submits the image reference on the lablab.ai MC3 form (deadline unknown, programme ends 2026-12-01). Update `docs/STATUS.md` (Task 22) and the registry row.
5. Optional: if the branch should be merged to master, ask the user first.

## Practical notes for the next session

- GPU: `export HEADLESS=1`; if `NOT_SIGNED_IN`, the USER runs `! node login.js` from `tools/amd-gpu`. Quota left after this session about 2 h; ALWAYS end with `node tools/amd-gpu/remote-mc3.js end`.
- Each new pod session needs: `sync` (run twice; the first call after a launch can fail), `model Qwen/Qwen3-Embedding-0.6B embedder tmp`, `data eval_mc3/data/dev`, `setup`, then `worker-start` (the pod cannot drop DAC_OVERRIDE, so no `nodac`).
- Never put the process name `sourcebound` in a `gpu.js sh` command (self-kill); use `eval_mc3/worker_ctl.sh`.
- `eval_mc3/data/` and `results/*.jsonl` are gitignored; regenerate data with `python eval_mc3/generate_corpus.py` (seeds: dev 101-104, holdout 901-902, scale 2001-2025).
- Do not tune on the holdout again; any change after `mc3-freeze-1` that affects answers needs a fresh holdout (new seeds).
