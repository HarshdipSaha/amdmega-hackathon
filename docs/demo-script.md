# SILENTPATH D1 demo recording script (about 2:20)

Record one terminal window and the evidence README. Keep the W7900D result segment onscreen long enough to read. The fake run demonstrates the software flow only; label it as fake. Do not present the short-run timings as a general speedup.

## 0:00-0:15 — Project

Show the root README. Say: “SILENTPATH checks which PyTorch SDPA attention operator ran on AMD ROCm and keeps its evidence beside the output. ROADREAD is a separate OCR project in this repository.”

## 0:15-0:45 — GPU-free path

Run `./tools/demo-silentpath.ps1`. Show the `run`, `report`, `which_path`, and `summarise_store` output. Say: “This is the fake producer. It verifies the CLI, record store, comparison report, and queries without claiming anything about GPU execution.”

## 0:45-1:05 — Actual operator evidence

Open `docs/evidence/silentpath-w7900/README.md` and one selected Session C profiler trace/operator log. Say: “On the W7900D, profiler operators distinguished forced Flash Attention from Math. Default dispatch showed Flash in the profiler, while the current comparator leaves Default-to-Flash unresolved.”

## 1:05-1:40 — Repeated decode result

Show the decode results in the evidence README. Say: “For two prompts with `max_new_tokens=128` and five repetitions per cell, Math was 1.30 to 1.36 times slower in the first session. A fresh reversed-order session measured 1.292 and 1.299. The short generated sequence first differed at token 65 in both sessions; the long sequence matched.”

## 1:40-2:05 — Limits

Say: “These results describe these prompts and this pinned model only. We collected no chosen-token log-probabilities for the decode runs, observed no positive silent fallback, and make no general answer-quality or speed claim. The separate G1 arithmetic run was numeric-only, with identical `1064.875` text and a maximum log-probability delta of `0.0086715`; its backend labels were request-derived.”

## 2:05-2:20 — Reproduction

Show the README reproduction commands and evidence directory. Say: “The configs pin the Qwen model revision. Run the fake workflow locally, or follow the documented commands on the pinned AMD ROCm image to reproduce the W7900D matrix.”

## Capture checklist

- [ ] Record at 1080p or higher with readable terminal text.
- [ ] Keep the model revision, repeated-run ratios, and limits visible in the evidence segment.
- [ ] Do not show credentials, private dashboard data, or unrelated desktop windows.
- [ ] Export MP4, check playback and duration, and add the public video URL to the submission draft.
