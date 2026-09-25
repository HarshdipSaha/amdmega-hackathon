# Development Set Baseline Evaluation (120 Images)

**Model:** `Qwen/Qwen3-VL-4B-Instruct` (BF16, SDPA attention, greedy decoding, non-thinking)  
**Run Tag:** `dev-q3vl4b-p1`  
**Dataset:** 120 images across 6 domain slices with synthetic degradations (blur, noise, low light, glare, skew, motion blur)  
**Hardware:** AMD Radeon Pro W7900D (48 GiB VRAM, `gfx1100`)  

---

## 1. Summary Scorecard

```json
{
  "correct": 109,
  "total": 120,
  "by_slice": {
    "us_plate": [19, 20],
    "cn_plate": [19, 20],
    "word_sign": [18, 20],
    "speed_sign": [20, 20],
    "advisory_plaque": [20, 20],
    "warning_sign": [13, 20]
  },
  "p50_s": 0.515,
  "max_s": 0.665,
  "max_overhead_s": 0.103,
  "violations": 0
}
```

- **Overall Exact Match Accuracy:** 109/120 (90.8%)
- **Speed Limit Signs:** 20/20 (100.0%)
- **Advisory Plaques:** 20/20 (100.0%)
- **US License Plates:** 19/20 (95.0%)
- **Chinese License Plates:** 19/20 (95.0%)
- **Word Signs:** 18/20 (90.0%)
- **Warning Signs:** 13/20 (65.0%)
- **p50 Latency:** 0.515 s
- **Max Latency:** 0.665 s (against 30.0 s ceiling)
- **Client IPC Overhead:** 0.103 s (< 110 ms)
- **Peak VRAM Allocated:** 9.76 GB
- **Harness Violations / Timeouts:** 0

---

## 2. Analysis and Error Breakdown

1. **Perfect Performance on Numbers and Speeds:** Both `speed_sign` and `advisory_plaque` achieved 100% accuracy across all degradation types without inventing units or misreading digits.
2. **License Plate Resilience:** 95% accuracy on both US and Chinese plates, successfully preserving Chinese province characters and handling glare/blur.
3. **Multiline Warning Signs:** Misses occurred almost exclusively on severe glare/skew artifacts where multi-word lines were occluded or truncated in the base view.
4. **Latency Budget:** Sub-second per-image inference (0.515s p50) leaves enormous headroom for the 30-second per-image deadline.
