# Session 1 Gates and Environment Verification

**Date:** 2026-09-25T09:44:40Z  
**Hardware Platform:** AMD Radeon Pro W7900D (`gfx1100`, RDNA3)  
**Host & Runtime:** Ubuntu 26.04, Python 3.14.4, PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333  

---

## 1. Measured Gate Benchmarks

| Metric | Measured Value | Threshold / Target | Status |
|--------|----------------|-------------------|--------|
| Base Image Unpacked Root Size | 21.25 GB (`21,254,608,884` bytes) | < 60 GiB | Passed |
| Container Image Headroom | 39.95 GB (`39,948,675,084` bytes) | >= 20 GB (for 9B bake) | `bake_9b` Tier Qualified |
| GPU Architecture | `gfx1100` (RDNA3) | AMD ROCm supported | Passed |
| Total VRAM Reported | 48 GiB (49,136 MB) | 48 GiB | Passed |
| BF16 Matmul Relative Error | `0.00308` | < 0.02 | Passed (`bf16_ok: true`) |
| Hugging Face Mirror Throughput | ~91.3 MB/s (`91,337,742` B/s) | > 50 MB/s | Passed |
| Free Persistent Workspace | 26.84 GB | 25 GB NFS volume | Passed |

---

## 2. Dependency Lock

Dependencies installed to `/workspace/pylib` (unshadowed, preserving ROCm PyTorch 2.13.0 in `/opt/venv`):
- `transformers==5.17.0`
- `tokenizers==0.23.2`
- `safetensors==0.8.0`
- `huggingface_hub==1.33.0`
- `regex==2026.9.10`

---

## 3. Model Snapshot

- Checkpoint: `Qwen/Qwen3-VL-4B-Instruct`
- Local directory: `/workspace/models/Qwen3-VL-4B-Instruct`
- Symlink: `/workspace/models/current`
- Total weight footprint: 8.3 GB
