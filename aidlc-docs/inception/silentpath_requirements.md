# Baseline Requirements: SILENTPATH (AMD Hackathon Main Challenge)

## 1. Executive Summary & Core Thesis
SILENTPATH investigates whether AMD ROCm attention backends (native PyTorch SDPA, FlashAttention, Math, Efficient Attention, Triton, AITER) silently diverge in numerical output, execution cost, or runtime fallback behavior under identical input prompts and greedy decoding.

The core question: **Which path did your AMD GPU silently take, what did it cost you, and did it change your answer?**

The project delivers:
1. An empirical kill-test (G1) evaluating backend output divergence on AMD hardware.
2. A reproducible measurement harness and benchmark suite (`silentpath`).
3. An orientation-independent cost and fallback comparator.
4. FastMCP server tools (D1 deliverable) enabling AI agents to introspect ROCm attention path integrity and performance costs.

---

## 2. Functional Requirements

### 2.1 The `RunRecord` Seam
- Every execution under a specific backend configuration must produce an immutable, content-addressed `RunRecord`.
- Record identity `record_id` must be deterministic based strictly on input specification `(config, workload_id)` — never on outputs — ensuring idempotent caching and seamless resume.
- Path introspection must differentiate between `requested` and `observed` backends via positive evidence, classifying fallbacks without conflating string aliases.

### 2.2 Three-Level Divergence Classification
- **IDENTICAL:** Bit-identical token IDs and chosen logprobs across backends.
- **NUMERIC-ONLY:** Identical token IDs and decoded text, but nonzero delta in chosen logprobs. This represents expected floating-point kernel differences and is strictly excluded from being reported as a "discovery".
- **DECISION-DIVERGENT:** Different token IDs or extracted discrete decisions. This constitutes a genuine finding.

### 2.3 Budget & Spend Ceiling
- Strict metering of billable cell wall time (including process start, engine initialization, and model load) against an immutable quota ceiling.
- Dual-barrier enforcement: pre-flight estimation check (`check_or_raise`) and post-execution backstop (`raise_if_over`).

---

## 3. Non-Functional Requirements (NFRs)

| Metric | Target / Constraint |
|---|---|
| Target Hardware | AMD Radeon Pro W7900D (48 GiB VRAM, `gfx1100`, RDNA3) / AMD Instinct CDNA |
| Execution Environment | PyTorch 2.13+ with ROCm 10 / Linux, with mock producer testability on Windows |
| Zero-GPU Testability | 100% of core modules (comparator, store, runner, budget, reporting, MCP) testable without a GPU |
| Daily Quota Safety | Zero background leakage; cloud sessions stopped immediately upon task completion |
| Test Coverage | Comprehensive unit tests for every module prior to hardware deployment |
