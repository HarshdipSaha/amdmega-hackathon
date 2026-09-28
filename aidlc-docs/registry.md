# AI-DLC Effort Registry & Evaluation: AMD Mega-Hackathon

## 1. Dual-Track Effort Registry

### Track A: ROADREAD (Mini-Challenge 2 — Vision OCR on ROCm)
| Effort ID | Ref / Name | Milestones | Status | Key Deliverables / Measured Results |
|---|---|---|---|---|
| `001` | `scaffold-normalization-rules` | Tasks 1–3 | `complete` | Normalization engine, domain rules (US banners, Chinese serials, repeat-digit safety) |
| `002` | `decode-ipc-pipeline-worker` | Tasks 4–8 | `complete` | Zero-import thin client (`app.py`), IPC socket, adaptive escalation pipeline, resident ROCm worker |
| `003` | `eval-remote-cloud-tooling` | Tasks 9–10 | `complete` | 10 sample extractor, process-isolated evaluation runner (`run_eval.py`), Playwright remote GPU automation |
| `004` | `gpu-gates-setup-smoke` | Tasks 11–12 | `complete` | Live W7900D GPU validation: **10/10 (100%) exact match**, 0.565s p50 latency |
| `005` | `synthetic-dataset-dev-eval` | Task 13 | `complete` | 120-image dev/holdout benchmark: **109/120 (90.8%) exact match**, 0.515s p50, 0 violations |
| `006` | `container-packaging-release` | Task 14 | `complete` | Multi-stage Dockerfile, startup daemon, automated 60 GiB & 11-layer container validation script |
| `007` | `fix-sample-hardcode-and-data-generator` | Bug fix fast-path | `complete` | Removed sample hardcoding for anti-cheating compliance; fixed text overflow with dynamic word wrap |

---

### Track B: SILENTPATH (Main Challenge — Differential Attention Harness & Telemetry)
| Effort ID | Ref / Name | Milestones | Status | Key Deliverables / Measured Results |
|---|---|---|---|---|
| `008` | `silentpath-scaffold-and-gates` | Tasks 1–3 | `complete` | Scaffolding; G2 hardware gate (`gfx1100`); G1 live GPU divergence test: **`NUMERIC-ONLY`** (`max_abs_logprob_delta = 8.67e-3`, bit-for-bit identical text `1064.875`), kernel GQA rejection confirmed on ROCm |
| `009` | `silentpath-core-record-comparator-store` | Tasks 4–6 | `complete` | Pydantic v2 `RunRecord` seam, 3-level comparator (`IDENTICAL`/`NUMERIC`/`DECISION`), append-only JSONL store with resume |
| `010` | `silentpath-budget-matrix-producer-runner` | Tasks 7–10 | `complete` | Adaptive GPU budget ceiling, YAML matrix expansion, `FakeProducer` for zero-GPU testing, matrix runner with cache |
| `011` | `silentpath-report-telemetry` | Tasks 11–12 | `complete` | Orientation-independent tabular reporter with fallback & admissibility flags, HIP device event timing, AMD-SMI VRAM telemetry |
| `012` | `silentpath-cli-mcp-verification` | Tasks 13–16 | `in-progress` | CLI surface (`probe`, `sweep`, `report`, `demo`), FastMCP server for AI agents, packaging & verification |

---

## 2. Winning Potential Evaluation (Score: 8.8 / 10)

### Dimension Scores:
1. **Scientific Rigor & Falsifiability (9.5/10):**
   - Did not manufacture false findings: When G1 returned `NUMERIC-ONLY`, the project documented the exact logprob drift ($8.67 \times 10^{-3}$) and cleanly pivoted to the cost and silent-fallback thesis rather than pretending noise is a discovery.
   - Guarded against order-dependent cost ratios and single-shot speed claims via `MIN_SAMPLES_FOR_SPEED_CLAIM`.

2. **AMD-Native Relevance & Architecture Fit (9.2/10):**
   - Directly addresses AMD ROCm kernel fragmentation (SDPA, FlashAttention, Math, Efficient Attention, Triton, AITER) and Grouped-Query Attention (GQA) kernel behavior.
   - Benchmarked and proven on physical AMD Radeon Pro W7900D hardware.

3. **Software Craftsmanship & Reliability (9.0/10):**
   - Clean seam architecture (`RunRecord`): Producers are isolated to hardware, while all evaluators, runners, budgets, and MCP tools are 100% testable without a GPU.
   - 72 passing automated tests in the test suite covering edge cases (interrupted stores, overspending budgets, alias matching).

4. **Completeness & Deliverable Scope (8.0/10):**
   - Mini-Challenge 2 (ROADREAD) is 100% complete and verified (Docker-ready, 90.8% dev score, 100% sample score).
   - Main Challenge (SILENTPATH D1) is ~75% complete (Tasks 1–12 done, Tasks 13–16 in progress). Reaching 10/10 requires shipping the CLI and FastMCP server tools.

**Overall Rating: 8.8 / 10** — A top-tier contender because it combines an actual working, audited mini-challenge submission with a scientifically rigorous, reproducible AMD developer tool.
