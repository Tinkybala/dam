# Candidate lock full-user preflight — results

## Material Passport

- ID: `anime-candidate-lock-full-user-preflight-2026-09-03`
- Type: validation-only code experiment result and reproducibility record
- Status: PASS; preflight complete
- Runbook: `22_LIJIE_CANDIDATE_LOCK_FULL_USER_PREFLIGHT_RUNBOOK.md`
- Preflight/source commit: `e4869c504c7fc376abc98649c9343517c1a63721`
- Source archive SHA-256: `37D512EC4D1DDA8838CCCD7C90D3561B6E3FC0BB11A97B73F8670FF868EF8F00`
- Metadata archive SHA-256 (server and local): `89633BAC0AE1B5C126B924554747783DAF34FC85AAC2FE41385D7115C6C6DA3F`
- Server release: `/mnt/hdd2/houlijie/sc4020_data_mining/releases/dam-e4869c504c7fc376abc98649c9343517c1a63721`
- Local metadata: `D:\Projects\SC4020_data_ming\tmp\server_results_20260903_candidate_lock_preflight\candidate-lock-preflight-metadata-e4869c50.tar.gz`
- Users/candidates: `60,384` users and `6,038,400` candidates
- Device/sampling: explicit CUDA; `gpu_sampling: true`
- `evaluate_test`: `false`; test was not read

## Model metrics

| Component | NDCG@10 | Hit Rate@10 | Total runtime (s) | Training runtime (s) | Epochs |
|---|---:|---:|---:|---:|---:|
| BPR ensemble component | 0.7360814346 | 0.9477179385 | 91.3947 | 86.5134 | 1 |
| Weighted NeuMF (`alpha=0.5`) | 0.7146354068 | 0.9295177530 | 20.7750 | 15.9226 | 1 |
| Percentile-rank ensemble (`w=0.7`) | 0.7351599315 | 0.9473536036 | — | — | — |

## Resource and schema checks

- GPU 0 peak: 900 MiB / 46,068 MiB, 87% utilization; 91 samples.
- GPU 1 peak: 812 MiB / 46,068 MiB, 91% utilization; 91 samples.
- Startup GPU checks showed the Python training children on GPU 0 and GPU 1; `nvidia-smi` reports those child PIDs while the shell `$!` values belong to their `timeout` parents.
- Both training return codes: `0`.
- Both metrics: 60,384 users, 1 epoch, CUDA, finite, correct commit, and no `test` key.
- Ensemble: fixed weight `0.7`, 60,384 users, 6,038,400 candidates, `validation_only: true`.
- Logs: no traceback, OOM, non-finite, failed, or timeout matches.
- Local repository: branch `lijie`, clean after pushed commit; final configs unchanged and remain test-sealed.
- FYP directory `/mnt/hdd2/houlijie/trellis2` was not touched.

## Stop point

The candidate-lock full-user preflight is complete. No 30-epoch run, seed 43/44 run, final config edit, or formal test run was started. The one-time SSH key remains available and must be deleted only after an explicit user command.
