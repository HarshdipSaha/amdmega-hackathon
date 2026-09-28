# Effort 008: SILENTPATH Scaffolding & G1/G2 Hardware Viability Gates

## Metadata
- **Effort ID:** `008`
- **Reference:** `silentpath-scaffold-and-gates`
- **Tasks Covered:** Tasks 1, 2, 3 of SILENTPATH Plan
- **State:** `complete`
- **Timestamp:** 2026-09-27

---

## Scope & Implementation Details
1. **Task 1: Project Scaffolding**
   - Configured `pyproject.toml` with Pydantic v2, PyYAML, FastMCP optional dependencies.
   - Set up test harness, `.gitignore`, and package layout (`silentpath/`, `gate/`, `tests/`).
2. **Task 2: Hardware Viability Gate (G2)**
   - Implemented `gate/g2_hardware.py` and `tests/test_g2_parse.py` (3/3 passing).
   - Validated AMD GPU target identification parsing `rocminfo` and identifying target architecture `gfx1100` (Radeon Pro W7900D).
3. **Task 3: G1 Divergence Kill-Test Benchmark**
   - Deployed benchmark script onto live AMD Radeon Pro W7900D cloud GPU via Playwright headless automation.
   - Tested SDPA attention backends on Qwen2.5-1.5B-Instruct:
     - `FLASH_ATTENTION`: Success (1.902s, output: `1064.875`).
     - `MATH`: Success (1.611s, output: `1064.875`).
     - `EFFICIENT_ATTENTION`: Failed as expected on ROCm with head mismatch (`Query.sizes(): [1, 12, 100, 128]`, `Key/Value sizes(): [1, 2, 100, 128]`) proving kernel-level GQA rejection.
   - G1 Classification Verdict: **`NUMERIC-ONLY`** (`max_abs_logprob_delta = 8.6715e-03`, token sequences bit-for-bit identical).
   - Persisted raw logs and `verdict.json` to `results/g1/` and committed to git.
