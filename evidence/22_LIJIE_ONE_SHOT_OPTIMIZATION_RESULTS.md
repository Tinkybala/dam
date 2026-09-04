# Anime one-shot optimization campaign — Phase F results

## Material Passport

- ID: `anime-oneshot-phase-f-2026-09-03`
- Type: validation-only code experiment result and reproducibility record
- Status: complete; permanent validation tuning stop reached
- Source commit used by the server release: `3e874b8b6a940365ba13a636fd84cba99c549246`
- Corrected local plan commit: `787484f36ac9c3764868f63fc79299fbaf7ed191`
- Server release: `/mnt/hdd2/houlijie/sc4020_data_mining/releases/dam-3e874b8b6a940365ba13a636fd84cba99c549246`
- Dataset/split: fixed Anime development split; 10,000 development users; 1,000,000 sampled candidates
- `evaluate_test`: `false` for every run
- GPU sampler: `true` for every run
- Raw summaries: `D:\Projects\SC4020_data_ming\tmp\server_results_20260902_oneshot\phase_f_candidate_summary.json`, `phase_f_controls_10k_summary.json`
- Gate artifact: `D:\Projects\SC4020_data_ming\tmp\server_results_20260902_oneshot\phase_f_gate.json`

## Valid paired results

The candidate summary contained six earlier full-user control rows from the interrupted setup. Those rows were explicitly rejected. The gate uses only the corrected 10,000-user control rerun and the six matching 10,000-user candidate rows.

| Track | Control mean NDCG@10 | Candidate mean NDCG@10 | Delta | Control mean HR@10 | Candidate mean HR@10 | Decision |
|---|---:|---:|---:|---:|---:|---|
| BPR | 0.7568308717 | 0.7823573282 | +0.0255264565 | 0.9442666667 | 0.9426333333 | Retain locked BPR; HR guard fails (−0.0016333333) |
| weighted NeuMF (`alpha=0.5`) | 0.7432154620 | 0.7670270249 | +0.0238115629 | 0.9347000000 | 0.9399333333 | Phase F gate passes |

All 12 valid trials were `complete`, finite, and validation-only. No CUDA OOM, traceback, or final-test read occurred.

## Ensemble confirmation

Using the already selected Phase E rank blend (`weight=0.7` for BPR and `0.3` for weighted NeuMF), the three paired validation-only ensembles were:

| Seed | NDCG@10 | HR@10 |
|---:|---:|---:|
| 42 | 0.7867671812 | 0.9494 |
| 43 | 0.7908172628 | 0.9497 |
| 44 | 0.7907210042 | 0.9482 |
| Mean | 0.7894351494 | 0.9491 |

The ensemble exceeds the BPR candidate mean by `+0.0070778212` NDCG and `+0.0064666667` HR, and exceeds the weighted-NeuMF candidate mean by `+0.0224081245` NDCG and `+0.0091666667` HR. It therefore passes the promotion rule.

## Stop/cleanup decision

- Validation-only tuning is permanently stopped as required by the campaign manual.
- No final configs or training code were changed during this close-out, per user instruction; all existing final configs remain `evaluate_test: false`.
- The promoted result is recorded as the frozen validation-only ensemble (`weight=0.7`). A formal test run remains a separate, explicitly authorized step.
- The one-time SSH public-key entry and local private/public key files were removed after the final remote process/key checks and local artifact verification.
