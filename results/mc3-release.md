# MC3 release verification (2026-10-02, W7900D)

The GitHub Actions `mc3-release.yml` run for `rc1` completed successfully. Its publish and image-check jobs passed, including the mandated base-prefix, size, command, environment, and secret checks.

The exact registry pull could not be completed on the notebook because downloading the temporary `crane` utility from GitHub stalled/truncated. The documented local-layer fallback was therefore used against the same release layers, pinned models, and ROCm base environment.

## Checkpointed notebook rehearsal

- Reader: Qwen3-VL-8B-Instruct, pinned revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- Embedder: Qwen3-Embedding-0.6B, pinned revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- Runtime: Torch `2.13.0+rocm10.0.0`, Transformers `5.17.0`, W7900D `gfx1100`.
- Kit: **10/10 strict**, 200/200 points, 0 violations.
- Kit indexing: 5.9s; p50 query 3.315s; maximum query 7.222s; peak VRAM 19.4 GiB.
- Supervisor recovery: worker was killed and respawned; liveness returned `E7731` citing `logs/prod_inference_2026-09-02.log`; supervisor remained alive.

The raw checkpoint artifacts are `checkpoint-kit.*`, `checkpoint-liveness.json`, and `checkpoint-supervisor.log` in this results directory. The pushed image itself remains covered by the successful CI image check; the notebook extraction was the local-layer fallback.
