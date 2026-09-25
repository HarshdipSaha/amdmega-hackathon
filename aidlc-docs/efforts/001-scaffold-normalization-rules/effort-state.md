# Effort 001: Scaffold, Evaluator Normalization, and Domain Rules

- **Status:** `complete`
- **Scope:** Tasks 1, 2, 3 of implementation plan
- **Completed At:** 2026-09-25T09:20:07Z
- **Commits:**
  - `f23d51f`: chore: scaffold roadread package
  - `9fd814a`: feat: evaluator normalization
  - `f1eaef0`: feat: plate/sign domain rules with repeat-safe tests

## Requirements Delta
- Setup editable package `roadread` with Pytest configuration.
- Implement strict challenge normalization (uppercase, drop whitespace and `[\s\-\.·_]`).
- Implement domain transcription rules:
  - Chinese plate structure validation and serial-only `I`/`O` replacement.
  - US state banner and slogan scrubbing with vanity plate safety.
  - Invariant preservation for repeated characters (e.g. `88888`).

## Verification
- Unit Tests: 11 tests passing (`test_normalize.py`, `test_rules.py`).
- 500-sample randomized property test for Chinese plate length preservation.
