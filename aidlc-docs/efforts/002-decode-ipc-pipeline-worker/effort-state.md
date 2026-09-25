# Effort 002: Image Decoding, IPC Protocol, Pipeline, Worker, and Engine

- **Status:** `complete`
- **Scope:** Tasks 4, 5, 6, 7, 8 of implementation plan
- **Completed At:** 2026-09-25T09:35:46Z
- **Commits:**
  - `a61adea`: feat: image decoding and reread views
  - `3d59f12`: feat: thin evaluator client and IPC protocol
  - `bd6f88e`: feat: bounded pipeline with single conditional escalation
  - `e67660c`: feat: resident worker with ready file
  - `dd53be3`: feat: Qwen transformers engine

## Requirements Delta
- Decode PNG, JPEG, TIFF (including 16-bit and multipage first-frame) with EXIF transposition and alpha compositing.
- Create zero-dependency thin evaluator client (`app/app.py`) without PyTorch imports to prevent cold-load overhead.
- Implement newline JSON IPC socket communication (`roadread/protocol.py`).
- Implement adaptive single-escalation pipeline (`roadread/pipeline.py`) with cross-view verification.
- Implement persistent worker (`roadread/worker.py`) with ready file signaling and PID tracking.
- Implement ROCm PyTorch VLM engine (`roadread/engine_qwen.py`) with BF16 greedy non-thinking generation.

## Verification
- Unit Tests: 31 unit tests passing across all core modules.
- Sub-process module isolation test confirming zero PyTorch/Transformers imports in client runtime.
