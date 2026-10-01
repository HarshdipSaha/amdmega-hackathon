# MC3 GPU session 1 (2026-10-01, W7900D gfx1100, torch 2.13.0+rocm10.0.0, transformers 5.17.0)

Reader: Qwen3-VL-4B-Instruct (dev). Embedder: Qwen3-Embedding-0.6B (/root/models, ephemeral: /workspace has <1 GB free).
Worker ready in 26.5 s. `setpriv` cannot drop DAC_OVERRIDE on the pod, so the mode-000 hazard is covered by unit tests + CI;
on the pod, run_suite deletes that file instead.

| Run | Strict | Notes |
|---|---|---|
| kit, baseline | 9/10 | Q9 chain under-cited (value quote only) |
| dev (4 corpora x 16), baseline | 53/64 | 7 under-cited chains; 4 false answers = readable chmod-000 file (root) |
| dev, hazard fix | 57/64 | 7 under-cited chains; the link-quote prompt rule did not change the 4B's behaviour |
| dev, row-key citation rule | **64/64** | kit 10/10; max 3.5 s/question, index <= 4.7 s, peak VRAM 12.07 GiB, 0 violations |

The row-key rule (gates.py) was derived from the dev failures, so dev is now development data.
The holdout (seeds 901, 902) is untouched. Remaining: Task 18 release reader sweep (8B, 9B), Task 20 holdout once + scale timing,
Task 21 release. Pod ended; results/*.jsonl are gitignored.
