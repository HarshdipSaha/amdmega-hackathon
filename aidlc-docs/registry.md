# AI-DLC Effort Registry & Evaluation: AMD Mega-Hackathon

## 1. Dual-Track Effort Registry

### Track A: ROADREAD (Mini-Challenge 2 — Vision OCR on ROCm)
| Effort ID | Ref / Name | Milestones | Status | Key Deliverables / Measured Results |
|---|---|---|---|---|
| `001` | `scaffold-normalization-rules` | Tasks 1-3 | `complete` | Normalization engine, domain rules (US banners, Chinese serials, repeat-digit safety) |
| `002` | `decode-ipc-pipeline-worker` | Tasks 4-8 | `complete` | Zero-import thin client (`app.py`), IPC socket, adaptive escalation pipeline, resident ROCm worker |
| `003` | `eval-remote-cloud-tooling` | Tasks 9-10 | `complete` | 10 sample extractor, process-isolated evaluation runner (`run_eval.py`), Playwright remote GPU automation |
| `004` | `gpu-gates-setup-smoke` | Tasks 11-12 | `complete` | Live W7900D GPU validation: **10/10 (100%) exact match**, 0.565s p50 latency |
| `005` | `synthetic-dataset-dev-eval` | Task 13 | `complete` | 120-image dev/holdout benchmark: **109/120 (90.8%) exact match**, 0.515s p50, 0 violations |
| `006` | `container-packaging-release` | Task 14 | `complete` | Multi-stage Dockerfile, startup daemon, automated 60 GiB & 11-layer container validation script |
| `007` | `fix-sample-hardcode-and-data-generator` | Bug fix fast-path | `complete` | Removed sample hardcoding for anti-cheating compliance; fixed text overflow with dynamic word wrap |

---

### Track B: SILENTPATH (Main Challenge — Differential Attention Harness & Telemetry)
| Effort ID | Ref / Name | Milestones | Status | Key Deliverables / Measured Results |
|---|---|---|---|---|
| `008` | `silentpath-scaffold-and-gates` | Tasks 1-3 | `complete` | Scaffolding; G1 on one Qwen 2.5 1.5B arithmetic prompt: Flash and Math produced identical text (`1064.875`) and token IDs, with max chosen-token log-probability delta `0.0086715`; explicit Efficient Attention failed on this GQA shape. The run does not establish silent fallback; `backend_observed` is request-derived in `gate/g1_transformers.py`. |
| `009` | `silentpath-core-record-comparator-store` | Tasks 4-6 | `complete` | Pydantic v2 `RunRecord` seam, 3-level comparator (`IDENTICAL`/`NUMERIC`/`DECISION`), append-only JSONL store with resume |
| `010` | `silentpath-budget-matrix-producer-runner` | Tasks 7-10 | `complete` | Adaptive GPU budget ceiling, YAML matrix expansion, `FakeProducer` for zero-GPU testing, matrix runner with cache |
| `011` | `silentpath-report-telemetry` | Tasks 11-12 | `complete` | Orientation-independent tabular reporter with fallback & admissibility flags, HIP device event timing, AMD-SMI VRAM telemetry |
| `012` | `silentpath-cli-mcp-verification` | Tasks 13-16 | `in-progress` | PyTorch SDPA producer, installable `silentpath` CLI, store-backed MCP functions, pinned W7900D evidence, isolated install verified, release branch published; recorded demo and signed-in competition submission remain |

---

## 2. Evidence-based status (2026-09-28)

No judged-potential score is available from the evidence in this repository; the former `8.8 / 10` rating and dimension scores were unsupported and are withdrawn. Current measured status:

- **SILENTPATH D1:** The SDPA producer, installable CLI, store-backed MCP functions, package/license/dependency files, and W7900D evidence are implemented. A fresh isolated package install and fake run/report succeeded. Effort 012 remains in progress for the recorded demo and signed-in competition submission.
- **G1:** One Qwen 2.5 1.5B arithmetic prompt on the W7900D yielded identical Flash/Math output text and token IDs, with maximum chosen-token log-probability delta `0.0086715`. This is numeric-only for that test. Efficient Attention errored on the GQA shape; this is not evidence of silent fallback. The G1 implementation populates `backend_observed` from the request.
- **ROADREAD:** Its W7900D smoke and OCR benchmark results remain separately documented: 10/10 exact match in the smoke set and 109/120 exact match in the 120-image dev/holdout set. These measurements do not validate SILENTPATH's attention-path claims.
- **Performance/cost:** Five-repeat W7900D runs found Math about 1.29-1.36x slower than Flash for these Qwen2.5-1.5B decode prompts; a reversed-order fresh-session repeat reproduced ratios of 1.292x and 1.299x. The short prompt diverged at generated token 65 in both sessions; the long prompt matched. This is workload-specific. No positive silent fallback was observed; default dispatch remains unresolved by the comparator. See `docs/evidence/silentpath-w7900/README.md`.
