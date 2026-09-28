# Effort 012: SILENTPATH CLI Surface, FastMCP Server & D1 Packaging

## Metadata
- **Effort ID:** `012`
- **Reference:** `silentpath-cli-mcp-verification`
- **Tasks Covered:** Tasks 13, 14, 15, 16 of SILENTPATH Plan
- **State:** `in-progress`
- **Timestamp:** 2026-09-28

---

## Scope & Target Deliverables
1. **Task 13: vLLM / Subprocess Producer**
   - Implement `silentpath/producers/vllm_worker.py` and `vllm_subprocess.py`.
   - Implement log-based path extraction and alias derivation.
2. **Task 14: CLI Interface (`silentpath/cli.py`)**
   - Provide subcommands:
     - `silentpath probe <model>`: Run quick probe of attention backends and display silent fallbacks.
     - `silentpath sweep <config.yaml>`: Run matrix sweep under budget constraints.
     - `silentpath report <records_dir>`: Render Markdown/ASCII comparison table.
     - `silentpath demo`: Zero-GPU demonstration using mock records.
3. **Task 15: FastMCP Tool Server (`silentpath/mcp_server.py`)**
   - Expose FastMCP tools for AI agent tool calling:
     - `probe_attention_path(model, requested_backend)`
     - `check_silent_fallbacks(records_path)`
     - `compare_backend_cost(model, backend_a, backend_b)`
4. **Task 16: Verification & Packaging**
   - Full test run across 100+ tests.
   - Freeze requirements and document D1 usage in `README.md`.
