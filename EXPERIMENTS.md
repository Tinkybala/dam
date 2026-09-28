# Experiment records

This file records repository-facing experiment checkpoints. Generated datasets,
predictions, checkpoints, and raw result directories remain outside Git.

## 2026-09-28 — Locked MovieLens 1M transfer completed

Candidate-lock commit `c5f903579c81d9a58e051714c27e2a9483946ca9` and
final-run commit `61863193e11650cdd579a102f172a51582502552` were deployed as
separate releases. Data integrity, two CUDA smoke jobs, 13/13 formal model runs,
12/12 trainable CUDA gates, three fixed ensembles, 10 deterministic anonymous
cases and a no-bulk metadata archive passed. Formal runs did not retry, time out,
change parameters or fall back to CPU. The final ensemble reached
0.632039 ± 0.003126 sampled-candidate test NDCG@10 and
0.838692 ± 0.003120 HR@10.

See [the detailed transfer report](evidence/movielens_transfer_results_20260928.md),
[machine-readable summary](evidence/movielens_transfer_summary_20260928.json),
and [anonymous case record](evidence/movielens_case_study_20260928.json).

## 2026-09-26 — Methods, setup and parameter-table draft

Added implementation-matched model/loss/ensemble equations, dataset and candidate
construction, checkpoint selection, and final parameter tables. All 19 archived
run configurations match both the archived YAML and the current final configs.
Verified the local raw-rating checksum against the Anime manifest, reconciled
run populations and split counts, and recovered best epochs from validation
histories. Source and environment hashes are recorded in
`evidence/anime_methods_audit_20260926.json`. BPR/NCF reference metadata was checked
against the author-deposited primary records.

See [Methods/setup](docs/32_LIJIE_METHODS_AND_EXPERIMENTAL_SETUP.md) and
[parameter tables](docs/33_LIJIE_FINAL_PARAMETER_TABLES.md). No model code,
experiment configuration, dataset or prediction was changed in this writing step.

## 2026-09-26 — Anime report figures from archived results

Prepared model-comparison, component/fusion and validation-only alpha figures in
PNG/SVG/PDF, with full-precision source tables and hashes. Cross-checked 19 model
metrics against the verified final metadata archive and recomputed mean/sample
SD. The actual BPR fusion component is kept separate from the standalone BPR
baseline. Phase D alpha results remain separate from full-user final test data.

See [figure captions and interpretation](docs/30_LIJIE_ANIME_REPORT_FIGURES.md).
No retraining, new parameter selection, MovieLens execution or GPU use occurred.

The v2 revision adds paired differences against standalone BPR, replaces the
repeated component bars with all 12 seed/component/metric fusion gains, and
marks the NDCG-selected alpha on both parameter panels. Figures now use 7.4-inch
layouts, 300-dpi PNGs and vector exports; v1 files are preserved. An English
[results/discussion draft](docs/31_LIJIE_ANIME_RESULTS_DISCUSSION_DRAFT.md) maps
claims to evidence and records missing cases/transfer results. The updated
analysis suite passes with 62 tests; the archived plotting input is unchanged.

## 2026-09-26 — MovieLens local preparation, GPU pending

MovieLens 1M adapter and artifact integrity gate implemented; 57 tests pass.
Local frozen artifacts contain 6,034 users, 3,125 movies, 562,308 training
positives, and 100 validation/test candidates per user. All 13 transferred
configs keep test evaluation disabled. No production training or test metrics
were run. Candidate-lock commit and CUDA preflight remain pending.

See [preparation evidence](evidence/movielens_local_preparation_20260926.md) and
its accompanying JSON for provenance, hashes and the next execution gate.

## 2026-09-02 — Anime full-user preflight benchmark

### Status

- Purpose: final-run feasibility and provenance check
- Evaluation mode: validation only
- Formal test status: sealed; no recorded metric contains a `test` field
- Source commit: `7ed1c2b6f0c789b6e1c20c52c6e116b9bdf07426`
- Release identifier: `dam-7ed1c2b6f0c789b6e1c20c52c6e116b9bdf07426`
- Metadata archive SHA-256:
  `d78fa8b91418162ae3b5c0476452b5d44e9c13155f8641b2a43412f38530fb7e`

### Frozen evaluation context

- Dataset: Anime Recommendations Database, positive threshold `rating >= 7`
- Eligible users: 60,384
- Warm items: 7,223
- Training positives: 5,087,394
- Split seed: 42
- Candidate protocol: one validation positive plus 99 fixed unseen negatives
- Training seed: 42
- Trainable-model benchmark budget: one epoch
- Hardware: NVIDIA RTX A6000

### Validation metrics

These are one-epoch feasibility measurements, not final model results and not a
new model-selection stage.

| Model | NDCG@10 | Hit Rate@10 | Total runtime |
|---|---:|---:|---:|
| Popular | 0.505679 | 0.770684 | 1.7 s |
| BPR | 0.657300 | 0.904528 | 118.3 s |
| GMF | 0.500919 | 0.764590 | 47.0 s |
| MLP | 0.611773 | 0.871688 | 49.2 s |
| NeuMF | 0.654878 | 0.898682 | 49.7 s |
| Weighted NeuMF | 0.653950 | 0.897721 | 49.9 s |

### Acceptance evidence

- Six of six planned runs completed.
- Every metric records the frozen source commit and 60,384 selected users.
- Every embedded benchmark configuration matches its locked final configuration
  after excluding the permitted benchmark-only changes: output directory,
  epochs, and early-stopping patience.
- All five trainable models completed one epoch on CUDA.
- Logs contain no traceback, CUDA out-of-memory error, non-finite loss, or
  timeout.
- GPU monitoring covered 320 one-second samples on GPU 0. Peak observed memory
  was 590 MiB of 46,068 MiB; peak utilization was 45%.
- The repository test suite passed: 24 tests.

### Reproducibility note

The server used PyArrow 17.0.0 and the local verification environment used
PyArrow 25.0.1. Parquet byte hashes therefore differed. The local verification
record reports matching canonical logical-content signatures, row counts,
schemas, and fixed-position samples for all nine prepared artifacts. Reports
must distinguish physical serialization identity from logical-content
equivalence.

### Interpretation boundary

The sampled metrics rank one held-out positive among 100 candidates. They must
not be presented as full-catalog performance, catalog coverage, or final test
evidence. The locked final configurations remain test-sealed at this checkpoint.

## 2026-09-05 — Anime locked final experiment

### Status

- Final training source commit:
  `b2f4d6b8222f9f5a9afd0633f54a235f50e52c69`
- The locked 19-run campaign completed without retry, timeout, or manual
  intervention during training.
- All 19 metrics passed the schema gate; all 18 trainable runs passed the GPU
  attach gate.
- The pre-declared `w=0.7` ensemble was generated from the locked component
  outputs without further weight search. It ranked first on both final metrics:
  `0.799658 ± 0.000259` NDCG@10 and `0.956335 ± 0.000843` HR@10.
- Six ensemble artifacts and the final summary passed their schema gates.
- The metadata archive matched locally and remotely by SHA-256, the final
  report was committed, and the one-time SSH credential was removed locally
  and from the server.
- The campaign is complete and archived. Stage F must not be rerun, and the
  test result must not be used to restart tuning.

The credential-free gate record is in
[`evidence/final_stage_f_gate_summary_2026-09-04.md`](evidence/final_stage_f_gate_summary_2026-09-04.md).
The sanitized continuation checkpoint and helper hashes are in
[`evidence/final_test_continuation_checkpoint_2026-09-05.md`](evidence/final_test_continuation_checkpoint_2026-09-05.md).
Operational helpers are retained under [`ops/`](ops/); generated metrics,
predictions, model checkpoints, server coordinates, and access credentials stay
outside Git.

The complete stage-by-stage explanation is in
[`docs/EXPERIMENT_PIPELINE_OVERVIEW.md`](docs/EXPERIMENT_PIPELINE_OVERVIEW.md),
and the final results are in
[`evidence/final_result_report_2026-09-05.md`](evidence/final_result_report_2026-09-05.md).
