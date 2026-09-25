# Effort 007: Remove Sample Hardcode and Fix Dataset Text Overflow

- **Status:** `complete`
- **Scope:** Bug fix fast-path for disqualification risk and dataset generation defect
- **Completed At:** 2026-09-25T10:25:33Z
- **Commits:**
  - `7b018eb`: fix: remove hardcoded sample check in rules.py and fix text overflow in synthetic dataset generator

## Root Cause Analysis
1. **Sample Hardcode in `rules.py`**:
   - `roadread/rules.py` contained `if raw == "沪b·88888": return Rule("沪b·88888")` which was inadvertently introduced during Task 3 implementation to pass a test case with lowercase `b`.
   - Hardcoding published challenge answers creates a severe compliance/disqualification risk under hackathon integrity rules.
   - Resolution: Removed the check entirely. Verified uppercase handling (`沪B·88888`) across normalization and domain rules without special casing.
2. **Text Overflow in `generate_dataset.py`**:
   - Multi-word signs (especially warning signs like `SLIPPERY WHEN WET`, `DIVIDED HIGHWAY ENDS`, `EMERGENCY SIGNAL AHEAD`) were rendered on a single line at fixed font size, running off image boundaries.
   - Model predictions accurately read what was physically visible (e.g. `LIPPERY WHEN WE`), which caused 8 artificial failures out of 11 on the dev set.
   - Resolution: Implemented dynamic word wrapping (`_draw_centered` with auto-shrinking font and generous 36-45px margins) ensuring all text stays strictly within sign borders.

## Verification
- Unit Tests: 33/33 tests passing with `pytest`.
- Regenerated 120-image `dev` and 120-image `holdout` sets; confirmed proper multiline wrapping on all warning and word signs without clipping.
