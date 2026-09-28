# MovieLens 1M locked transfer results — 2026-09-28

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: frozen GPU run and reproducibility validation
- Verification Status: ARCHIVED_VERIFIED for the 13 model runs, 3 fixed ensembles,
  and the local metadata archive; inference beyond this protocol is qualified
- Owner: Lijie model-based recommender workstream
- Version Label: movielens_transfer_final_v1

## Locked execution and provenance

- Dataset: [GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/),
  downloaded from the official distribution and checked against its publisher MD5.
- Candidate-lock commit: `c5f903579c81d9a58e051714c27e2a9483946ca9`.
- Final-run source commit: `61863193e11650cdd579a102f172a51582502552`.
- The 13 final configs differ from candidate-lock configs only in
  `evaluate_test: false` → `true`. A test assertion was updated to require
  all 13 configs to be unsealed; the model and data code did not change.
- Raw archive SHA-256:
  `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`.
- `ratings.dat` SHA-256:
  `506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f`.
- Final summary SHA-256:
  `7d1e76e655bbafce82b32def7eb70b6e13bb873a97a8818eda4afec47e3c8cab`.
- Final metadata archive SHA-256:
  `8cf604fe5f9492e1fffb96f889cdc3d5b9fa99df01ba264ab72a9715bc338de7`.
- Final campaign script SHA-256:
  `ff0ae7f4d81956d4238b92d5aaa5b62cd106072185267fdb0443bdb8a335e08a`.
- Fixed postprocessor SHA-256:
  `9d1cc62a041e728e255b23207bac1ca131026ab3d9cbe2d4ccd7fa4ede50b0c4`.
- Case-selection script SHA-256:
  `0730114721ad8689ccdc1a0411eacb371ad9ae4f157fd43cef600b11b9aca1ea`.

The source commit was deployed as an archive to its own server release. The
official dataset and nine prepared artifact files passed checksums and a read-only
schema/negative-exclusion gate. Local test suite: 62 passed. Two one-epoch smoke
jobs (BPR and Weighted NeuMF, seed 42) ran on CUDA with `evaluate_test: false`;
their metrics had validation only and recorded the candidate-lock commit. Formal
runs began only after the smoke and artifact gates.

The server campaign completed 13/13 model runs: Popular once, then BPR, GMF,
NeuMF and Weighted NeuMF at seeds 42/43/44. All 12 trainable runs recorded
`device: cuda`, GPU negative sampling, and the final-run commit; the monitor
captured processes on both RTX A6000 GPUs. Every metrics file contains both
validation and test results for 6,034 eligible users. No worker exited with an
error, retry, timeout, CPU fallback or non-finite loss. Logs and the completion
marker were checked. The test configurations were not revised after seeing
formal scores.

## Evaluation protocol

MovieLens uses rating ≥ 4 as positive and keeps lower ratings in the observed
exclusion set. The frozen positive 5-core/split has 6,034 eligible warm users,
3,125 warm movies and 562,308 training positives. Every user has one validation
positive, one test positive and 99 fixed unseen negative candidates in each
split. NDCG@10 is primary; Hit Rate@10 is secondary. Each number below is from
the **sampled 100-candidate test**, not a full-catalog or online evaluation.
Three-seed standard deviations describe variation across training seeds on one
fixed split and are not confidence intervals.

## Final sampled-candidate results

| System | Seeds | Test NDCG@10 mean ± sample SD | Test HR@10 mean ± sample SD | Mean recorded run time |
|---|---:|---:|---:|---:|
| Popular | 1 | 0.355596 (one run) | 0.580709 (one run) | 0.21 s |
| GMF | 3 | 0.507165 ± 0.003409 | 0.758314 ± 0.004067 | 17.95 s |
| NeuMF | 3 | 0.599754 ± 0.003445 | 0.825710 ± 0.001963 | 27.10 s |
| BPR | 3 | 0.620558 ± 0.003101 | 0.825820 ± 0.009887 | 107.42 s |
| Weighted NeuMF | 3 | 0.623203 ± 0.003328 | 0.831013 ± 0.005221 | 21.39 s |
| Fixed 0.7 BPR + 0.3 Weighted NeuMF | 3 | **0.632039 ± 0.003126** | **0.838692 ± 0.003120** | Derived; no extra model training |

Run time is per-model total time as recorded by the training command, including
its validation and test scoring. The final ensemble also requires both trained
components and rank blending; the table is not a controlled serving-latency
benchmark. BPR's longer time partly reflects its locked 24-negative setting and
different realized epoch counts. Popular has no gradient training.

## Transfer finding and component check

The common-model order by mean NDCG is the same as in Anime: ensemble,
Weighted NeuMF, BPR, NeuMF, GMF, Popular. MLP was run on Anime only and is not
part of this transfer matrix. On MovieLens, the fixed ensemble exceeds BPR by
0.011481 NDCG and 0.012872 HR, and exceeds Weighted NeuMF by 0.008836 NDCG and
0.007679 HR. For both metrics and both components, each of the three paired
seed differences is positive. The blend weight was carried over from Anime
without searching on MovieLens validation or test results.

The direction of some secondary effects changed. Weighted NeuMF has higher mean
NDCG than BPR in both datasets, but its HR is slightly lower than BPR on Anime
and higher on MovieLens. The shared pattern is therefore the ranking of systems
by primary NDCG and the positive fusion gain, not an identical effect for every
metric. Anime and MovieLens have different users, movies, scales and positive
thresholds; their absolute metric magnitudes should not be read as a direct
measure of dataset difficulty.

## Deterministic success and failure cases

Seed 42 cases were selected using the smallest eligible source user IDs in each
predeclared category, up to two per category. The public result replaces IDs
with case labels. Rank refers to the held-out positive among 100 fixed sampled
test candidates. Histories shown in the JSON are from training positives only.

| Anonymous case | Category | Held-out movie | BPR rank | Weighted NeuMF rank | Ensemble rank |
|---|---|---|---:|---:|---:|
| Case 01 | Rank-one success | *Driving Miss Daisy* (1989) | 1 | 1 | 1 |
| Case 03 | Boundary hit | *The Good Earth* (1937) | 14 | 11 | 10 |
| Case 05 | Fusion miss | *Glengarry Glen Ross* (1992) | 19 | 8 | 16 |
| Case 07 | BPR hit, weighted miss | *Arlington Road* (1999) | 8 | 12 | 8 |
| Case 09 | Weighted hit, BPR miss | *Glengarry Glen Ross* (1992) | 19 | 8 | 16 |

Case 05 and Case 09 are the same selected profile because the failure and
component-disagreement rules overlap. The boundary example shows a case in which
blending moved a held-out positive from ranks 14/11 to rank 10. Case 05 shows a
counterexample: the weighted component alone hits the top ten, but blending
drops the same item to rank 16. These examples illustrate ranking behavior; they
were selected after the locked test and did not change model choice or weights.
The fully anonymous ten-case record includes two examples per category plus
history and top-ten sampled-candidate titles. It is not a full-catalog case study.

## Artifacts and remaining scope

- Versionable, no-bulk summary:
  `evidence/movielens_transfer_summary_20260928.json`.
- Anonymous cases: `evidence/movielens_case_study_20260928.json`.
- Local server archive (ignored by Git):
  `results/movielens/deploy/final/final_metadata_6186319.tar.gz`.
- Report figures: `results/movielens/report_20260928/model_comparison.*` and
  `component_ablation.*`; the prior Anime validation alpha graph serves as the
  parameter-analysis figure.
- Figure code: `ops/build_transfer_figures.py`. The figure manifest hashes the
  frozen Anime and MovieLens inputs, script and six image/vector outputs.

The metadata archive has 13 model logs, 13 source configs, 16 metrics files
(13 model runs plus three derived ensembles), monitor data and analysis scripts.
It excludes raw ratings, checkpoints and prediction parquet files. The archive
was transferred once and its local SHA-256 matched the server SHA-256.

One non-experiment operational error occurred before smoke: a Windows-to-Bash
script pipe appended a carriage return to the raw-data path during the first
artifact-audit command. Reissuing that read-only command with a clean path
passed; no training was attempted before the successful audit. There were no
formal experiment retries. Final group report, contribution statement and
source-only ZIP still require team assembly and submission checks.
