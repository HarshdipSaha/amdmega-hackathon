# Baseline Requirements: ROADREAD (AMD AI Challenge — Mini-Challenge 2)

## 1. Executive Summary
ROADREAD is a vision-language OCR and transcript normalization pipeline built for the AMD AI Developer Program Mini-Challenge 2 (LabLab.ai). The solution transcribes unseen, severely degraded images of vehicle license plates and road signs into exact string format, evaluated under strict automated harness constraints on AMD ROCm compute.

---

## 2. Functional Requirements

### 2.1 Domain Categories
1. **US License Plates (`us_plate`):** Extract uppercase registration letters and digits. Strip state headers, slogans, DMV URLs, and frame banners while preserving vanity plate letters.
2. **Chinese License Plates (`cn_plate`):** Extract leading Chinese province characters, uppercase series letters, standard dot separator (`·`), and alphanumeric serials. Convert full-width characters and serial-only `I`/`O` characters to `1`/`0`. Preserve all repeated characters.
3. **Word-Based Signs (`word_sign`):** Extract printed uppercase words top-to-bottom joined by single spaces.
4. **Speed Limit Signs (`speed_sign`):** Extract exact wording (e.g. `SPEED LIMIT 65`).
5. **Advisory Plaques (`advisory_plaque`):** Extract numeric speed advisory values (e.g. `35`) without adding extraneous units such as MPH.
6. **Multiline Warning Signs (`warning_sign`):** Extract all printed uppercase text line-by-line joined with single spaces (e.g. `ROAD WORK AHEAD`).

### 2.2 Normalization Engine (`roadread.normalize`)
- Evaluator normalization strictly applies uppercase conversion and drops whitespace and punctuation `[\s\-\.·_]`.
- All other punctuation and Unicode characters must remain intact.

---

## 3. Non-Functional Requirements (NFRs)

| Constraint | Specification Limit | System Design Target |
|---|---|---|
| Base Container Image | `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0` | Exact match |
| Container Image Size | <= 60 GiB (64,424,509,440 bytes) uncompressed | ~30 GB (`bake_9b` tier qualified) |
| Hardware Accelerator | AMD Radeon Pro W7900D (48 GiB VRAM, `gfx1100`) | Native ROCm PyTorch execution |
| Per-Image Latency | 30.0 seconds hard timeout | < 1.0 second (measured ~0.51s–0.56s) |
| Evaluator Startup | 600.0 seconds budget | < 30.0 seconds (measured ~23s) |
| Total Run Latency | 600.0 seconds for 10 test images | < 10.0 seconds |
| Client Process Overhead | Standard library thin client | < 100 ms per invocation |
| Peak VRAM | 1.0 GiB to 48.0 GiB | ~9.76 GB allocated |
