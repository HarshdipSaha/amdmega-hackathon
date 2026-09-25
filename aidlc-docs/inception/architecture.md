# Baseline Architecture: ROADREAD (System Architecture & Component Design)

## 1. Architectural Topology

```
+-----------------------------------------------------------------------+
| Docker Container (rocm/pytorch:rocm10.0_ubuntu26.04_py3.14)           |
|                                                                       |
|  [Evaluator CLI]                                                      |
|         │                                                             |
|         ▼ (invoked per image)                                         |
|  app/app.py (Thin Client, standard library only, 0ms torch load)      |
|         │                                                             |
|         ▼ Localhost TCP Socket (Port 47811, Newline JSON)             |
|  roadread/worker.py (Persistent Background Worker, Pre-warmed)        |
|         │                                                             |
|         ├── roadread/decode.py (EXIF, Alpha Composite, View Bounding) |
|         ├── roadread/pipeline.py (Adaptive Single-Escalation Engine)  |
|         ├── roadread/engine_qwen.py (BF16 Qwen3-VL on ROCm PyTorch)   |
|         └── roadread/rules.py (Chinese Format, Banner Scrubbing)      |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

## 2. Core Components

1. **Thin Evaluator Client (`app/app.py`):**
   - Implements standard library Python only.
   - Zero PyTorch or Transformers imports to eliminate multi-second module load latency.
   - Communicates with resident worker via `roadread/protocol.py`.

2. **Resident GPU Worker (`roadread/worker.py`):**
   - Persistent daemon initialized at container entry.
   - Loads VLM weights into GPU VRAM once; executes pre-warmup pass.
   - Exposes Unix/TCP localhost socket with PID tracking.

3. **Adaptive Escalation Pipeline (`roadread/pipeline.py`):**
   - Pass 1: Bounded primary view (1.0 MP max).
   - Validation Gate: Evaluates candidate against domain rules (e.g. valid Chinese plate structure).
   - Escalation Pass: Triggers single higher-resolution or upscaled reread if primary output is suspect or uncertain under the timing budget.

4. **Domain Transcription Rules (`roadread/rules.py`):**
   - Standardizes full-width ASCII and separator characters (`·`).
   - Maps `I`->`1` and `O`->`0` inside Chinese plate serial numbers without modifying provincial characters or Latin signs.
   - Removes US state banners while strictly preserving vanity plates.
   - Guarantees zero repeated-character deduplication.

5. **Headless Cloud GPU Driver (`tools/amd-gpu/`):**
   - Playwright-driven browser automation for JupyterLab REST and WebSocket APIs on notebooks.amd.com.
   - Handles environment synchronization, gate benchmarking, dependency locking, model downloading, and evaluation execution.
