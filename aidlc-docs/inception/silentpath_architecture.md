# Baseline Architecture: SILENTPATH (Differential Attention Harness & Telemetry)

## 1. System Topology & Seam Architecture

```
+──────────────────────────────────────────────────────────────────────────+
| Configuration Matrix (YAML)                                              |
|      │                                                                   |
|      ▼ expand()                                                          |
| Matrix Cells [backend × workload]                                        |
|      │                                                                   |
|      ▼ run_matrix(budget, estimate_seconds)                              |
| +──────────────────────────────────────────────────────────────────────+ |
| | Runner Core (`silentpath/runner.py`)                                  | |
| |   ├── RecordStore (`silentpath/store.py`, append-only JSONL, cache)    | |
| |   └── GpuBudget (`silentpath/budget.py`, pre/post-flight ceiling)     | |
| +───────────────────────────────────┬──────────────────────────────────+ |
|                                     │                                    |
|              ┌──────────────────────┴──────────────────────┐             |
|              ▼                                             ▼             |
|      [Hardware Producer]                           [Mock Producer]       |
|  `vllm_subprocess` / `g1_transformers`       `silentpath/producers/fake`  |
|  (Linux ROCm / W7900D GPU)                   (Zero-GPU CI / Dev machine) |
|              │                                             │             |
|              └──────────────────────┬──────────────────────┘             |
|                                     │                                    |
|                                     ▼                                    |
|                    RunRecord (The Seam, Pydantic v2)                     |
|                                     │                                    |
|         ┌───────────────────────────┼───────────────────────────┐        |
|         ▼                           ▼                           ▼        |
|  3-Level Comparator         Report Generator              FastMCP Server |
|  (`silentpath/compare.py`)  (`silentpath/report.py`)      (`mcp_server`) |
|  - IDENTICAL                - Orientation-independent     - probe_path   |
|  - NUMERIC                  - Fallback flags              - check_fallbk |
|  - DECISION (Finding)       - Cost ratios (admissible)    - sweep_eval   |
+──────────────────────────────────────────────────────────────────────────+
```

---

## 2. Component Directory Structure

1. **`silentpath.record`:** Defines `Confidence`, `ObservedPath`, `CostSample`, `Output`, and `RunRecord`. Implements deterministic hashing for cell memoization.
2. **`silentpath.compare`:** Strict divergence evaluator ensuring floating-point logprob drift is classified as `NUMERIC` and never conflated with a decision-level discrepancy.
3. **`silentpath.store`:** Content-addressed, append-only JSONL store resistant to corrupted lines and process interruptions.
4. **`silentpath.budget`:** Hardware cost tracking metering cell wall time versus inference generation time.
5. **`silentpath.matrix`:** Sweep definition parser expanding YAML matrices into independent execution cells.
6. **`silentpath.producers`:** Protocol-based abstraction with `FakeProducer` for deterministic testing and real GPU sub-process execution.
7. **`silentpath.report`:** Tabulation engine producing orientation-independent cost ratios, variance statistics, and fallback summaries.
8. **`silentpath.telemetry`:** Device-level instrumentation capturing HIP events and AMD-SMI VRAM metrics.
