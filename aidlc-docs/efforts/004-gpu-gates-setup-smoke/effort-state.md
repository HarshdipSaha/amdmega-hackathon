# Effort 004: GPU Session 1 & 2 (Gates, Dependency Lock, Model Snapshot, Smoke Test)

- **Status:** `complete`
- **Scope:** Tasks 11, 12 of implementation plan
- **Completed At:** 2026-09-25T09:51:52Z
- **Commits:**
  - `65b5eec`: feat: complete session 1 hardware gates and smoke evaluation with 10/10 exact match

## Requirements Delta
- Execute hardware gate benchmarks on AMD Radeon Pro W7900D (`gfx1100`, RDNA3, 48 GiB VRAM).
- Verify base image layer budget and uncompressed footprint (21.25 GB root used, `bake_9b` tier qualified).
- Verify BF16 matrix multiply accuracy (`bf16_ok: true`, rel error `0.00308`).
- Freeze and lock unshadowed ROCm Transformers environment into `app/requirements.lock`.
- Download `Qwen/Qwen3-VL-4B-Instruct` snapshot (8.3 GB).
- Execute smoke evaluation on 10 reference challenge images.

## Verification
- Reference Sample Accuracy: **10/10 (100.0%) exact match**.
- Inference Latency: p50 **0.565s**, max **0.665s** (against 30s limit).
- Client Overhead: **0.095s** (< 100 ms).
- Peak VRAM: 9.75 GB.
- Quota Management: Pod cleanly terminated with 8,683s quota preserved.
