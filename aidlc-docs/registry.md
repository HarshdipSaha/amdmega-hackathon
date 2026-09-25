# AI-DLC Effort Registry: ROADREAD (AMD AI Developer Challenge)

| Effort ID | Ref / Name | Scope / Milestones | Status | Key Deliverables / Metrics |
|---|---|---|---|---|
| `001` | `scaffold-normalization-rules` | Tasks 1, 2, 3 | `complete` | Editable package, normalization engine, domain rules (US banner scrubbing, Chinese serial I/O, repeat-char safety) |
| `002` | `decode-ipc-pipeline-worker` | Tasks 4, 5, 6, 7, 8 | `complete` | Multi-format image decoder, zero-import thin client (`app.py`), localhost IPC protocol, adaptive escalation pipeline, resident model worker, ROCm Qwen engine |
| `003` | `eval-remote-cloud-tooling` | Tasks 9, 10 | `complete` | 10 sample extractor, process-isolated evaluation runner (`run_eval.py`), Playwright remote cloud GPU orchestrator (`remote.js`), hardware gates benchmark (`gates.py`) |
| `004` | `gpu-gates-setup-smoke` | Tasks 11, 12 | `complete` | GPU Session 1: 21.25 GB root size (`bake_9b` tier qualified), BF16 verified, dependencies locked, 4B snapshot downloaded; GPU Session 2: **10/10 (100%) exact match**, 0.565s p50 |
| `005` | `synthetic-dataset-dev-eval` | Task 13 | `complete` | 120-image dev/holdout splits with 6 degradations; GPU Session 3: **109/120 (90.8%) exact match**, 0.515s p50, 0 violations |
| `006` | `container-packaging-release` | Task 14 | `complete` | Multi-stage Dockerfile, entrypoint daemon, automated 60 GiB & 11-layer container validation script |
