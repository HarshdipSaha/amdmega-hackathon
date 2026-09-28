# SILENTPATH W7900D evidence (2026-09-28)

This bundle records local gate work and four W7900D notebook sessions (A, B, the fresh reversed-order B2 repeat, and C). Source changes are on branch `feat/silentpath-w7900`, commits `08b530a` (producer/CLI/MCP), `800ab18` (timing/path evidence fixes), and `a2895a9` (missing log-probability delta is reported as unavailable). The GPU image, ROCm/PyTorch versions, and pinned model revision are in [environment.json](environment.json).

## What the GPU showed

| Workload | Repetitions | Observed path | Result |
|---|---:|---|---|
| Gate A invoice, one generated token | 1 | Default → Flash; forced Flash → Flash; forced Math → Math | All outputs were one token; this is path evidence only. |
| Prefill, two prompt lengths, one generated token | 5 per cell | Flash and Math operators were distinct; default dispatched to Flash | Outputs matched. Math/Flash wall-time ratios were 1.26–1.32 in Session B and 1.32–1.40 in the demo sweep. Treat these very short runs as preliminary. |
| Decode, two prompt lengths, max_new_tokens=128 | 5 per cell | Forced Flash and Math matched their profiler operator names | Math was slower: 1.30–1.36× in Session B. In the fresh, reversed-order repeat it was 1.292× (short) and 1.299× (long). |
| Short decode output | Same prompt, seed, model, and generation settings | Flash vs Math | Token IDs first differed at position 65 in both sessions; the long-prompt output was identical. No chosen-token log-probabilities were collected, so a log-probability delta is unavailable. |

The forced-path attribution comes from PyTorch profiler operator names and saved Chrome traces. Confidence is `reported` (one evidence source). No positive silent fallback was observed. Default dispatch was observed using Flash Attention, but the comparator marks `DEFAULT` versus Flash as unresolved because it does not yet treat default dispatch as an alias of a concrete backend. The output difference is limited to this generated sequence workload; it does not establish a general answer-quality or invoice-correctness claim.

## Run records and reports

- [Session A: records and report](session-a/records.jsonl) · [report](session-a/report.json) · [config](session-a/config.yaml)
- [Session B prefill: records and report](session-b-prefill/records.jsonl) · [report](session-b-prefill/report.json) · [config](session-b-prefill/config.yaml)
- [Session B decode: records and report](session-b-decode/records.jsonl) · [report](session-b-decode/report.json) · [config](session-b-decode/config.yaml)
- [Fresh-session reversed-order repeat: records and report](session-b2-decode/records.jsonl) · [report](session-b2-decode/report.json) · [config](session-b2-decode/config.yaml)
- [Session C CLI demo: records and report](session-c-demo/records.jsonl) · [report](session-c-demo/report.json) · [config](session-c-demo/config.yaml)

The Session C folder includes three representative Chrome traces, compressed losslessly as `.trace.json.gz`, and their raw profiler operator logs for default, forced Flash, and forced Math on the short prefill workload. Decompress a trace to its original `.trace.json` filename to open it in Chrome tracing. The SHA-256 hashes of the uncompressed traces are `90233012a945e3bcd5b2265ad7cf54426c49e88522df04cfb539e2ec2d09f031` (default), `1208c6757e1248468803fe3917d81d99a72a943f26ca08c87a73251b15a6d21e` (Flash), and `6a7fd092768f5e130082a0dbf3a3c7d8f950d36004548e0d453d994e80993733` (Math). Full raw traces for every cell are retained in the ignored local archives under `results/`; their SHA-256 checksums are in [archive-sha256.json](archive-sha256.json). Decode traces are large and remain in those local archives.

## Reproduction and limits

Run from the repository root with the W7900D notebook environment and the model snapshot from revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` available locally. The evidence configs name the cached local model path used in that session; substitute the path to that pinned snapshot if needed.

```bash
python -m silentpath.cli run --matrix configs/w7900-prefill.yaml --out runs/w7900-prefill --cap-hours 0.5 --producer sdpa
python -m silentpath.cli report --out runs/w7900-prefill
```

The `which_path` and `summarise_store` functions were called against the Session C store and returned the same six records, observed operators, artifact names, and comparison summary. MCP fixture tests also pass; a networked MCP transport session was not launched. vLLM was not tested. The W7900 workspace was nearly full, so this run staged the pinned weights and large traces in pod scratch and retrieved all artifacts before shutting down.
