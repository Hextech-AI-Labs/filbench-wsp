# 05 — Implementation Status (code-complete checkpoint)

Date: 2026-11-18. All code below is implemented; no new experimental results claimed.

## Complete
- [x] Fail-closed edit harness (schema / applier / matched random controls) — unchanged semantics for old plans
- [x] Scripts 01–05 (inspect, splits, eval, score, propose/ingest)
- [x] 06 language-contrastive diagnostics + packet v2 renderer (+ --allow-tied-rows)
- [x] 07 counterfactual controls (sign-flip, layer-shuffle)
- [x] 08 in-memory K-plan screening + feedback packet renderer
- [x] scale_tied_rows operator: tie-preserving shared delta, bitwise tie verification,
      absent-lm_head handling, matched random control that also preserves the tie,
      per-experiment enable flag (--allow-tied-rows at ingest/screen; default OFF)
- [x] Configurable eval batch size (eval_adapter + eval_frozen.yaml)
- [x] tests/test_tied_rows.py

## Before first new run
- [ ] pytest (full suite, incl. test_tied_rows.py)
- [ ] Re-run baseline once with new batch_size (parity counters change with batching)
- [ ] Extend 06 seed corpora via --fil-file / --en-file (few hundred sentences each)

## Experiment conditions now supported
zero-shot v1 | zero-shot v2 (measured profile) | v2 + tied rows | K-plan screen + feedback |
controls: matched random, sign-flip, layer-shuffle