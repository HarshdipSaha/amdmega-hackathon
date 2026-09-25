# Smoke Test Evaluation Notes (10 Brief Reference Samples)

**Model:** `Qwen/Qwen3-VL-4B-Instruct` (BF16, SDPA attention, greedy decoding, non-thinking)  
**Execution Environment:** Resident worker on AMD Radeon Pro W7900D (48 GiB VRAM)  
**Evaluator Invocation:** Process-isolated `python app/app.py --input-image ...` via TCP IPC  

---

## 1. Summary Scorecard

```json
{
  "correct": 10,
  "total": 10,
  "by_slice": {
    "us_plate": [3, 3],
    "cn_plate": [2, 2],
    "word_sign": [2, 2],
    "speed_sign": [1, 1],
    "warning_sign": [1, 1],
    "advisory_plaque": [1, 1]
  },
  "p50_s": 0.565,
  "max_s": 0.665,
  "max_overhead_s": 0.095,
  "violations": 0
}
```

- **Exact Match Accuracy:** 10/10 (100.0%)
- **p50 Latency:** 0.565 s (against 30 s per-image hard limit)
- **Max Latency:** 0.665 s
- **Client IPC Overhead:** 0.095 s (< 100 ms)
- **Peak VRAM Allocated:** 9.75 GB (out of 48 GiB capacity)
- **Harness Violations / Timeouts:** 0

---

## 2. Sample-by-Sample Analysis

| Image | Format | Domain Slice | Ground Truth | Prediction | Latency | Overhead | Result |
|---|---|---|---|---|---|---|---|
| `image_01.png` | PNG | US Plate (clean) | `7ABC123` | `7ABC123` | 0.608 s | 0.095 s | Correct |
| `image_02.png` | PNG | Chinese Plate (clean) | `京A·12345` | `京A·12345` | 0.573 s | 0.089 s | Correct |
| `image_03.jpg` | JPEG | US Plate (angled / NY slogan) | `JHT 2951` | `JHT 2951` | 0.665 s | 0.094 s | Correct |
| `image_04.png` | PNG | US Plate (motion blur) | `5XYZ891` | `5XYZ891` | 0.565 s | 0.082 s | Correct |
| `image_05.jpg` | JPEG | Chinese Plate (low light / glare) | `沪B·88888` | `沪B·88888` | 0.627 s | 0.088 s | Correct |
| `image_06.png` | PNG | Word Sign (clean) | `STOP` | `STOP` | 0.505 s | 0.083 s | Correct |
| `image_07.tiff` | TIFF | Word Sign (noisy) | `STOP` | `STOP` | 0.521 s | 0.085 s | Correct |
| `image_08.jpg` | JPEG | Speed Sign (multiline) | `SPEED LIMIT 65` | `SPEED LIMIT 65` | 0.589 s | 0.087 s | Correct |
| `image_09.png` | PNG | Warning Sign (multiline) | `ROAD WORK AHEAD` | `ROAD WORK AHEAD` | 0.515 s | 0.077 s | Correct |
| `image_10.tiff` | TIFF | Advisory Plaque (numeric) | `35` | `35` | 0.465 s | 0.086 s | Correct |

---

## 3. Key Observations

1. **Format Agility:** PNG, JPEG, and uncompressed/compressed TIFF images all decode cleanly with first-frame and EXIF orientation handling.
2. **Domain Rules:**
   - Sample 3 NY slogan "EXCELSIOR" and state header were automatically filtered, preserving only the plate registration `JHT 2951`.
   - Sample 5 repeated 8s (`沪B·88888`) were strictly preserved without heuristic truncation.
   - Sample 10 numeric plaque `35` returned purely `35` without hallucinating "MPH".
3. **Inference Latency:** Sub-second per image (0.46s – 0.66s), providing massive margin under the 30-second evaluator budget.
