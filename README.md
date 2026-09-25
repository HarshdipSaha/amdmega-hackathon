# AMD Mega Hackathon — ROADREAD (Mini-Challenge 2)

## Overview

This repository contains the complete implementation, research, automated GPU tooling, and evaluation harness for Mini-Challenge 2 of the AMD AI Developer Program Hackathon (hosted on LabLab.ai).

The project, titled ROADREAD, is an autonomous, high-precision license plate and road sign recognition engine specifically engineered and optimized for AMD ROCm hardware (AMD Radeon Pro W7900D, 48 GiB VRAM, gfx1100 architecture).

---

## Challenge Summary and Objectives

The objective of Mini-Challenge 2 is to extract exact text transcriptions from unseen, severely degraded road imagery across six challenging operational domains:

1. United States vehicle license plates (filtering state banners, slogans, and DMV URLs while retaining registration numbers).
2. Chinese vehicle license plates (retaining leading province characters, letters, and repeated serial characters without destructive heuristic truncation).
3. Word-based road signs (such as STOP or ONE WAY).
4. Worded speed limit signs (such as SPEED LIMIT 65).
5. Number-only advisory plaques (such as 35 without adding invented units like MPH).
6. Multiline warning and work-zone signs (such as ROAD WORK AHEAD, transcribed strictly top-to-bottom and joined by single spaces).

Scoring is evaluated on 10 hidden test images (20 points each, 200 points total) using strict normalized exact-string match:
- Normalization: uppercase conversion, elimination of all whitespace, and removal of specific punctuation (`-`, `.`, `·`, `_`).
- Zero character error tolerance: a single missing or mistranscribed character invalidates the score for that image.

---

## Hardware and Execution Constraints

Submissions must operate strictly within the mandated container and compute boundaries:
- Base Image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`
- Image Size Gate: Maximum uncompressed image size of 60 GiB (64,424,509,440 bytes).
- GPU Hardware: AMD Radeon Pro W7900D (48 GiB VRAM, RDNA3 gfx1100).
- Precision: BF16 / FP16 execution (FP8 is unsupported on RDNA3 matrix hardware).
- Latency Budgets:
  - Container startup: under 600 seconds.
  - Per-image inference: hard ceiling of 30 seconds (internal pipeline budget set to 25–27 seconds).
  - Overall evaluation: 10 images completed within 600 seconds total post-startup.
- GPU Memory Budget: 1 GiB to 48 GiB peak memory (sampled every 3 seconds by the evaluator).

---

## Architectural Design

### 1. Thin Client Architecture (`app/app.py`)
Importing heavy libraries like PyTorch and Transformers on ROCm incurs several seconds of interpreter overhead per execution. To meet the 30-second deadline reliably, `app/app.py` is written purely in standard library Python without importing PyTorch or Transformers. It acts as a lightweight client communicating via localhost TCP socket to a persistent background worker.

### 2. Resident Model Worker (`roadread/worker.py`)
The model is loaded into GPU VRAM once during container initialization and warmed up on dummy tensors. The worker serves sequential inference requests over IPC, eliminating cold-load latency on individual test images.

### 3. Adaptive Escalation Pipeline (`roadread/pipeline.py`)
- First pass: Fast inference on a bounded view (1 MP max).
- Disagreement / Suspect Trigger: If the first transcription fails domain sanity checks (such as invalid Chinese registration format or unexpected character length), a single high-resolution escalation reread is triggered on original pixels or an upscaled view.
- Non-sampling greedy decoding with thinking mode disabled (`enable_thinking=False`) to prevent unbounded chain-of-thought token generation.

### 4. Robust Domain Post-Processing (`roadread/rules.py`)
- Full-width ASCII and Unicode separator normalization (converting dot variants to standard middle dot `·`).
- Serial-only I/O mapping for Chinese registrations (mapping I to 1 and O to 0 strictly inside the serial, leaving province characters and Latin signs untouched).
- Curated US state banner and slogan filtering without stripping valid vanity plate letters.
- Guarantee against repeated-character deduplication (preserving patterns such as repeated 8s).

### 5. Automated Cloud GPU Driver (`tools/amd-gpu/`)
Complete headless Playwright automation driving the AMD AI Notebooks hub and JupyterLab REST/WebSocket API directly from local tooling. Supports automated environment synchronization, dependency installation, Hugging Face weight downloads, gate benchmarking, and evaluation execution.

---

## Repository Structure

```
├── app/
│   ├── app.py                     # Evaluator entry point (thin client, stdlib only)
│   ├── requirements.txt           # Candidate dependencies
│   └── requirements.lock          # Frozen ROCm environment dependencies
├── roadread/
│   ├── __init__.py                # Package initialization
│   ├── normalize.py               # Official challenge evaluation normalization
│   ├── rules.py                   # Domain transcription rules and format checks
│   ├── decode.py                  # EXIF-aware image decoding and view generators
│   ├── prompts.py                 # Structured VLM instructions
│   ├── protocol.py                # Localhost TCP IPC protocol
│   ├── pipeline.py                # Adaptive single-escalation inference pipeline
│   ├── engine_fake.py             # Deterministic mock engine for CPU unit testing
│   ├── engine_qwen.py             # ROCm PyTorch engine for Qwen3-VL / Qwen3.5
│   └── worker.py                  # Resident IPC model worker
├── eval/
│   ├── extract_samples.py         # PyMuPDF sample extractor from challenge brief
│   ├── run_eval.py                # Process-isolated evaluation runner
│   ├── gates.py                   # Session-1 hardware and bandwidth verification gates
│   └── samples/                   # 10 reference sample images with gold answers
├── tests/
│   ├── test_normalize.py          # Normalization test suite
│   ├── test_rules.py              # Domain rules and repeat-character safety tests
│   ├── test_decode.py             # Format decoding and view generation tests
│   ├── test_protocol_app.py       # IPC protocol and thin-client unit tests
│   ├── test_pipeline.py           # Pipeline escalation and selection tests
│   └── test_eval.py               # Evaluator harness tests
├── tools/amd-gpu/
│   ├── gpu.js                     # Core Playwright driver for AMD JupyterLab API
│   ├── remote.js                  # Remote workflow orchestrator for sync, gates, and eval
│   ├── browser.js                 # Shared CDP browser runner
│   └── login.js                   # Authenticated session manager
├── docker/
│   ├── Dockerfile                 # Multi-stage production container build
│   ├── entrypoint.sh              # Container startup and worker orchestrator
│   └── check_container.sh         # Pre-flight container verification script
├── docs/                          # Specifications, research audits, and execution plans
└── pyproject.toml                 # Package definition and pytest configuration
```

---

## Implementation Progress and Current Task

Implementation is structured in 14 strictly gated milestones:

### Local Milestones (Tasks 1 to 10)
- Task 1: Scaffold Python project and configuration — Completed
- Task 2: Evaluator normalization engine (`roadread/normalize.py`) — Completed
- Task 3: Domain transcription rules and safety checks (`roadread/rules.py`) — Completed
- Task 4: Image decoding, EXIF orientation, and view scaling (`roadread/decode.py`) — Completed
- Task 5: IPC protocol and zero-dependency thin client (`app/app.py`) — Completed
- Task 6: Inference pipeline with mock engine (`roadread/pipeline.py`) — Completed
- Task 7: Resident background model worker (`roadread/worker.py`) — Completed
- Task 8: Transformers ROCm engine (`roadread/engine_qwen.py`) — Completed
- Task 9: Evaluation harness and brief sample extraction (`eval/run_eval.py`) — Completed
- Task 10: Remote GPU driver and hardware gate benchmarks (`tools/amd-gpu/remote.js`) — Completed

### Remote AMD GPU Milestones (Tasks 11 to 13)
- Task 11: GPU Session 1 — Hardware benchmark gates, ROCm environment lock, and model snapshot download — Completed
- Task 12: GPU Session 2 — Reference 10-sample verification and latency profiling (10/10 100% exact match, p50 0.565s, 9.75GB VRAM) — Completed
- Task 13: GPU Session 3 — 120-image development benchmark, model tier sweep, and locked holdout evaluation — Next


### Release Packaging (Task 14)
- Task 14: Docker container build, base layer verification, and anonymous registry deployment

---

## Local Development and Testing

To install local development dependencies and run the unit test suite:

```bash
# Install package in editable mode with development dependencies
python -m pip install -e ".[dev]"

# Run full pytest test suite
python -m pytest -v
```

---

## License

All challenge development and submission artifacts are licensed under the MIT License.
