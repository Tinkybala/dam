# Final Test Continuation Checkpoint — 2026-09-05

This repository record intentionally excludes credentials, private keys, API
tokens, hostnames, account names, and machine-specific absolute paths.

## Frozen experiment state

- Final training source commit: `b2f4d6b8222f9f5a9afd0633f54a235f50e52c69`
- Stage F evidence commit: `2ed6a14`
- Stage F: complete
- Model metrics schema gate: 19/19 passed
- Trainable-run GPU attach gate: 18/18 passed
- Automatic retry, timeout, and manual intervention during training: none
- Stage G fixed ensemble: not yet generated
- Final metric summary: not yet generated
- Formal training outputs remain preserved outside Git.

Stage F must not be rerun. Continuation starts with the read-only Stage G
preflight and then derives the already locked percentile-rank ensemble with BPR
weight `0.7` and weighted-NeuMF weight `0.3`.

## Reproducibility helpers

| File | Purpose | SHA-256 |
|---|---|---|
| `ops/final_19run_campaign.sh` | Exact launcher used for the completed 19-run Stage F campaign | `FDC2E1A8472F86D5FE13B8A015F6A4F15CE38FFFE1A050089867B99D650E5F9A` |
| `ops/final_stage_g_h.py` | Fail-closed Stage G preflight, fixed ensemble derivation, and three-seed summary | `98D2FE145FB05C6521D2201A751A386F71E1A350A41C669F8837AC3823D1460D` |

The Stage G–H helper refuses to overwrite an existing ensemble or summary,
requires the frozen source commit, verifies all 19 metrics, checks six
prediction inputs and their row counts, and evaluates only the predeclared
weight `0.7`. It does not train a model or search ensemble weights.

## Access boundary

The previous one-time SSH credential was revoked and deleted. A newly issued
one-time credential is required for continuation and must remain outside this
repository. No credential material is required by either helper.
