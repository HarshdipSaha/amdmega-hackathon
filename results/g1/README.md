# SILENTPATH Gate 1 (G1) Divergence Kill-Test Results

**Hardware:** AMD Radeon Pro W7900D (48 GiB VRAM, `gfx1100`, RDNA3)  
**Software:** ROCm 10.0.0, PyTorch 2.13.0, Transformers 5.17.0, Python 3.14.4  
**Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Prompt:** Multi-step arithmetic reasoning invoice (`prompts/g1.txt`)  
**Verdict:** **`NUMERIC-ONLY`** (`max_abs_logprob_delta = 8.6715e-03`)  

---

## 1. Summary of Attention Paths

| Backend Requested | Backend Observed | Status | Output Decoded | Token IDs | Generation Latency | VRAM Peak |
|---|---|:---:|---|---|---:|---:|
| `FLASH_ATTENTION` | `PyTorch SDPBackend.FLASH_ATTENTION` | **Success** | `1064.875` | `[16, 15, 21, 19, 13, 23, 22, 20, 151645]` | 1.902 s | 3.18 GB |
| `MATH` | `PyTorch SDPBackend.MATH` | **Success** | `1064.875` | `[16, 15, 21, 19, 13, 23, 22, 20, 151645]` | 1.611 s | 3.18 GB |
| `EFFICIENT_ATTENTION` | `None` | **Failed / Incompatible** | — | — | — | — |

---

## 2. Key Findings

### Finding A: Numeric-Only Divergence (Identical Tokens, Divergent Logprobs)
- Both `FLASH_ATTENTION` and `MATH` generated the exact discrete numerical result: `"1064.875"`.
- However, the underlying token log-probabilities differed across attention kernels:
  - First token (ID 16): Flash `-1.0496` vs Math `-1.0578` ($\Delta = 0.0081$)
  - Fourth token (ID 19): Flash `-2.1310` vs Math `-2.1223` ($\Delta = 0.0087$)
  - **Maximum Absolute Logprob Delta:** $8.67 \times 10^{-3}$
- Under the spec decision tree:
  > *NUMERIC-ONLY: Logprobs differ but every decoded output is identical. Proceed, but the thesis narrows to cost. Report the null honestly — still the first such measurement on AMD.*

### Finding B: Silent Incompatibility in `EFFICIENT_ATTENTION` on GQA Architectures
- When `EFFICIENT_ATTENTION` is explicitly requested, ROCm PyTorch aborts execution with:
  ```
  RuntimeError: No available kernel. Aborting execution.
  UserWarning: For dense input, both fused kernels require query, key and value to have the same num_heads. 
  Query.sizes(): [1, 12, 100, 128], Key sizes(): [1, 2, 100, 128], Value sizes(): [1, 2, 100, 128] instead.
  ```
- Because Qwen uses Grouped-Query Attention ($12$ query heads vs $2$ key/value heads), `EFFICIENT_ATTENTION` is mathematically unsupported by the fused kernel on ROCm.
- Under unconstrained PyTorch SDPA dispatching, requests for efficient attention silently fall back to `MATH` or `FLASH_ATTENTION`.
