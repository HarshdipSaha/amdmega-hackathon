# SILENTPATH Gate 1 (G1) Divergence Kill-Test Results

**Hardware:** AMD Radeon Pro W7900D (48 GiB VRAM, `gfx1100`, RDNA3)  
**Software:** ROCm 10.0.0, PyTorch 2.13.0, Transformers 5.17.0, Python 3.14.4  
**Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Prompt:** One multi-step arithmetic prompt (`prompts/g1.txt`; one prompt, one successful run per backend)
**Verdict:** **`NUMERIC-ONLY`** (`max_abs_logprob_delta = 8.6715e-03`)  

---

## 1. Summary of Attention Paths

| Backend requested | Independent runtime path evidence | Status | Output decoded | Token IDs | Generation latency | VRAM peak |
|---|---|:---:|---|---|---:|---:|
| `FLASH_ATTENTION` | Not independently captured; runner label came from the requested setting | **Success** | `1064.875` | `[16, 15, 21, 19, 13, 23, 22, 20, 151645]` | 1.902 s | 3.18 GB |
| `MATH` | Not independently captured; runner label came from the requested setting | **Success** | `1064.875` | `[16, 15, 21, 19, 13, 23, 22, 20, 151645]` | 1.611 s | 3.18 GB |
| `EFFICIENT_ATTENTION` | `None` | **Failed / Incompatible** | — | — | — | — |

---

## 2. Key Findings

### Finding A: Numeric-Only Difference (Identical Tokens, Different Logprobs)
- Both `FLASH_ATTENTION` and `MATH` generated the exact discrete numerical result: `"1064.875"`.
- However, the underlying token log-probabilities differed across attention kernels:
  - First token (ID 16): Flash `-1.0496` vs Math `-1.0578` ($\Delta = 0.0081$)
  - Fourth token (ID 19): Flash `-2.1310` vs Math `-2.1223` ($\Delta = 0.0087$)
  - **Maximum Absolute Logprob Delta:** $8.67 \times 10^{-3}$
- Under the spec decision tree:
  > *NUMERIC-ONLY: Logprobs differ but every decoded output is identical. Proceed, but the thesis narrows to cost. Report the null honestly — still the first such measurement on AMD.*

### Finding B: Explicit Efficient Attention Failed on This GQA Configuration
- When `EFFICIENT_ATTENTION` is explicitly requested, ROCm PyTorch aborts execution with:
  ```
  RuntimeError: No available kernel. Aborting execution.
  UserWarning: For dense input, both fused kernels require query, key and value to have the same num_heads. 
  Query.sizes(): [1, 12, 100, 128], Key sizes(): [1, 2, 100, 128], Value sizes(): [1, 2, 100, 128] instead.
  ```
- The warning reports Qwen's Grouped-Query Attention shape (12 query heads vs 2 key/value heads); the explicitly requested fused path failed for this configuration.
- This forced-backend failure does not show what unconstrained PyTorch SDPA dispatch would select, and it does not demonstrate a silent fallback. That behavior needs a separate run that observes runtime dispatch independently.

## 3. Measurement limits

This is one arithmetic prompt and one successful generation per backend. It establishes identical Flash/Math output for this test and the reported log-probability difference; it does not establish how often other prompts diverge, an OCR answer change, or a performance/cost advantage. Generation and wall times are single observations.

The checked-in `gate/g1_transformers.py` assigns `backend_observed` from the requested backend on successful runs. Those labels are therefore not independent runtime observations. The failed Efficient Attention record has no observed backend.
