# MC3 scale timing (2026-10-02, W7900D)

Fresh worker, pinned 8B reader, `SB_MIN_CALL_S=10`, 300-file scale corpus:

- Index: **63.65s**; 300 files; 725 segments; 50 images transcribed.
- Worst query: **15.303s**; p50 3.996s.
- Peak VRAM: **21.01 GiB**.
- Contract violations: **0**.

The scale corpus intentionally combines 25 companies with repeated question wording, so its 13/16 strict accuracy is not an accuracy claim. It is timing evidence only.
