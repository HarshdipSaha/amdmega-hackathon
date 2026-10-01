# MC3 reader sweep (E4), holdout and scale timing (2026-10-01, W7900D)

| Reader | Dev (64) | Kit (10) | max s/question | peak VRAM |
|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | 64 | 10 | 3.4 | 12.1 GiB |
| Qwen3-VL-8B-Instruct | 63 (1 false refusal) | 10 | 10.1 | 20.0 GiB |
| Qwen3.5-9B (thinking off) | 62 (2 false answers) | 10 | 8.8 | 21.3 GiB |

Dev is saturated (row-key rule was tuned on it), so it cannot separate the readers. Release reader chosen: **Qwen3-VL-8B-Instruct**
(spec default; stronger OCR margin for the harder hidden corpus; fits image size; latency well inside 30 s). Frozen at tag mc3-freeze-1.

Holdout (seeds 901, 902; run once, 8B): **32/32**, max 7.5 s, peak 19.9 GiB.

Scale (300 files, 50 images transcribed, fresh worker, 8B): index 63.7 s, worker start 18 s (about 82 s of the 600 s startup budget);
max question 22.7 s (one ambiguous question used 3 reader attempts; others 1.7-7.6 s). Accuracy there is not meaningful (25 companies share
wording). Fix: SB_MIN_CALL_S default 6 -> 10 so no attempt starts unless it can finish inside the 25 s budget.
