# Final Stage F Gate Summary — 2026-09-04

This evidence file intentionally contains no concrete test metric values.

- Branch: `lijie`
- Final-run source commit: `b2f4d6b8222f9f5a9afd0633f54a235f50e52c69`
- Server release: `/mnt/hdd2/houlijie/sc4020_data_mining/releases/dam-b2f4d6b8222f9f5a9afd0633f54a235f50e52c69`
- Results directory: `/mnt/hdd2/houlijie/sc4020_data_mining/releases/dam-b2f4d6b8222f9f5a9afd0633f54a235f50e52c69/results/final`
- Launcher SHA-256: `FDC2E1A8472F86D5FE13B8A015F6A4F15CE38FFFE1A050089867B99D650E5F9A`

## Gates

- `CAMPAIGN_COMPLETE`: present
- `schema_gate=PASS`: 19/19
- `gpu_attach=PASS`: 18/18
- Worker GPU 0 exit code: `0`
- Worker GPU 1 exit code: `0`
- Abnormal keyword scan (`traceback`, `OOM`, `non-finite`, `NaN`, `failed`, `timeout`): 0 matches
- Retry: none
- Timeout: none
- Manual intervention during Stage F: none; the approved launcher replacement occurred before launch.

## Device-level monitor peaks

Values are device-level peaks from `gpu-monitor.csv` and may include concurrent processes on the shared server.

| GPU | Peak memory | Peak utilization |
|---|---:|---:|
| 0 | 13,335 MiB | 100% |
| 1 | 13,333 MiB | 100% |

Stage G and the fixed `w=0.7` ensemble were not run. The formal results remain in the server release for continuation.
