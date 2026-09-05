# LIJIE Anime Final Result Report — 2026-09-05

## Verification status

- Status: `ARCHIVED_VERIFIED`
- Branch: `lijie`
- Final-run source commit: `b2f4d6b8222f9f5a9afd0633f54a235f50e52c69`
- `final_summary.json` SHA-256: `B9F34504DA2ADF564E23ED17B3215A2D582E2B164F13FC436393EE6EA8DA2DDB`
- Metadata archive SHA-256: `F8FBE19C4B8BACB4D8EF728155F1D5717C5E9C0E2680C922A60A3E22B7E6C74A`
- Server release: `/mnt/hdd2/houlijie/sc4020_data_mining/releases/dam-b2f4d6b8222f9f5a9afd0633f54a235f50e52c69`
- Server archive: `/mnt/hdd2/houlijie/sc4020_data_mining/final-test-metadata-b2f4d6b8.tar.gz`
- Local archive: `D:\Projects\SC4020_data_ming\tmp\server_results_20260905_final\final-test-metadata-b2f4d6b8.tar.gz`

Stage F completed with 19/19 schema gates, 18/18 GPU attach gates, and zero worker exit failures. Stage G–H generated the locked percentile-rank ensemble with `w=0.7`; the six ensemble artifacts and `final_summary.json` passed `STAGE_G_H_SCHEMA_GATE=PASS`.

## Sampled-candidate test results

The evaluation uses one held-out positive item and 99 sampled negatives per eligible warm user. Values below are the analyzed test results supplied after reviewing `final_summary.json`.

| System | Test NDCG@10 | Test HR@10 |
|---|---:|---:|
| Ensemble `w=0.7` | **0.799658 ± 0.000259** | **0.956335 ± 0.000843** |
| Weighted NeuMF | 0.783825 ± 0.000137 | 0.946862 ± 0.003018 |
| BPR | 0.770767 ± 0.000330 | 0.952703 ± 0.000526 |
| NeuMF | 0.765100 ± 0.000851 | 0.947094 ± 0.000734 |
| GMF | 0.716820 ± 0.002421 | 0.935043 ± 0.001161 |
| MLP | 0.716791 ± 0.000522 | 0.936037 ± 0.000538 |
| Popular | 0.507031 | 0.771761 |

BPR ensemble components reached `0.789854 ± 0.000248` NDCG@10 and `0.950053 ± 0.001238` HR@10, but remain ensemble components rather than standalone ranking entries under the pre-declared protocol.

## Main findings

- The locked ensemble is highest on both NDCG@10 and HR@10, with all three seeds winning against the comparison systems.
- Ensemble versus BPR component: `+0.009804` NDCG@10 and `+0.006282` HR@10.
- Ensemble versus Weighted NeuMF: `+0.015832` NDCG@10 and `+0.009473` HR@10.
- Ensemble versus standalone BPR: `+0.028891` NDCG@10 and `+0.003632` HR@10.
- All listed ensemble gains are positive for seeds 42, 43, and 44.
- Validation-to-test gap for the ensemble is only `−0.000475` NDCG@10 and `−0.000861` HR@10.
- The ensemble reduces Weighted NeuMF's HR seed SD from `0.003018` to `0.000843`.

## Interpretation boundaries

- Three seeds provide descriptive stability evidence only; the reported SDs are not strong statistical-significance claims.
- This is a sampled-candidate test, not an evaluation against all 7,223 warm items and not an online click-through estimate.
- Warm-user filtering and the fixed 1:99 candidate ratio define the generalization boundary; aggregate scores do not imply identical performance for every user.
- The 11/11 statistical-fallacy audit found no test-after-selection, repeated-tuning, causal overclaim, or regression-to-the-mean issue.

## Reproducibility and cleanup

- The metadata archive was created once and its local SHA exactly matched the server SHA.
- The one-time SSH key was deleted locally and removed from the server after archive verification.
- Stage G/H outputs and the complete server release remain in place for future inspection; no further tuning or rerun is authorized by this record.

## Execution note

The first helper invocation generated the six ensemble files and the summary. A later accidental duplicate invocation was stopped by the helper's `refusing to overwrite` guard; it did not alter those artifacts. The archived `stage-g-h.log` therefore records that later guard refusal, while artifact existence plus the independent schema gate establish the successful generated outputs.
