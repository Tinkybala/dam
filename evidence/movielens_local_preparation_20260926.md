# MovieLens local preparation — 2026-09-26

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run (local ETL and verification only)
- Verification Status: VERIFIED for local preparation; GPU execution NOT RUN
- Owner: Lijie model-based recommender workstream
- Version Label: movielens_local_preparation_v1

## Completed

Implemented the MovieLens 1M adapter, dataset-aware preparation CLI, and a
read-only artifact validator. The adapter accepts `UserID::MovieID::Rating::Timestamp`,
checks integer IDs and the 1–5 rating range, and discards timestamps. Internally,
movie IDs use the legacy `anime_id` column so the existing model pipeline remains
compatible. This is a seeded random split, not a chronological evaluation.

Ratings >= 4 are positive; lower ratings still exclude observed movies from
negative sampling. Preparation uses iterative positive 5-core filtering,
leave-two-out with warm-item repair, split seed 42, and 99 unseen negatives.
Existing non-empty artifact directories now fail rather than being overwritten.
Each generated parquet file has a row count and SHA-256 in the manifest.

Official source: [GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/).
The downloaded archive passed its publisher-provided MD5 check. Full source URL,
download time, byte counts, SHA-256 hashes, and artifact manifest are recorded in
`evidence/movielens_local_preparation_20260926.json`.

| Local data gate | Result |
|---|---:|
| Raw ratings | 1,000,209 |
| Eligible warm users | 6,034 |
| Warm movies | 3,125 |
| Training positives | 562,308 |
| Validation positives | 6,034 |
| Test positives | 6,034 |
| Candidates per user per split | 100 |
| Candidate rows per split | 603,400 |
| Development users | 6,034 (all eligible users, no duplication) |
| Artifact files verified | 9/9 |

The independent artifact gate checked raw and artifact hashes, finite integer
values, user/item mappings, disjoint splits, warm held-out items, label-to-holdout
identity, one positive per user, candidate counts, and exclusion of all observed
ratings. Test files were accessed only for preparation/integrity checks; no
MovieLens test predictions or metrics were computed.

## Configuration and tests

- 13 validation-only configs: Popular once; BPR, GMF, NeuMF, Weighted NeuMF at
  seeds 42/43/44.
- All 13 have `evaluate_test: false`; all 12 trainable configs require `cuda`
  and `gpu_sampling: true`.
- BPR inherits the Anime ensemble-component configuration, as required by the
  transfer plan. Other models inherit their corresponding Anime final configs.
- Tests compare all non-dataset hyperparameters against the exact source files.
  The transferred ensemble remains fixed at 0.7 BPR + 0.3 Weighted NeuMF.
- `py -m pytest`: **57 passed** (original suite plus 16 new cases).
- Tests cover malformed input, repeatable preparation, low-rating exclusion,
  Anime defaults, overwrite refusal, artifact corruption, incorrect held-out
  labels, config migration, and sealed-test access during synthetic validation.
- `git diff --check`: passed; Windows line-ending notices only.

Changes are in the local working tree, based on commit
`4d2bd1fbfe5700fa1e5288f46cd9a1a76dacd5f5`. **There is no new candidate-lock commit
yet.** Source/config file hashes record this exact local preparation snapshot;
they do not replace the required release commit before GPU execution.

## Reproduce or inspect locally

Run from the repository root. The preparation command requires a new or empty
output directory; the current artifact directory is already populated.

```powershell
py -m src.prepare --dataset movielens-1m --ratings data/movielens-1m/ratings.dat --output artifacts/movielens-1m-r4 --positive-threshold 4 --core-size 5 --seed 42 --negative-count 99 --development-user-count 10000
py -m src.validate_artifacts --artifacts artifacts/movielens-1m-r4 --ratings data/movielens-1m/ratings.dat
py -m pytest
```

Local logs: `results/movielens/preparation/prepare.log` and
`results/movielens/preparation/artifact_gate.json`. Raw data, the official README,
download provenance and ZIP are under `data/movielens-1m/`; frozen artifacts are
under `artifacts/movielens-1m-r4/`. These directories remain Git-ignored.

## GPU handoff

1. Review and commit the source, tests, configs and evidence as the MovieLens
   candidate-lock commit. Deploy a release of that exact commit when a GPU is
   available. Do not run formal experiments from the old base commit.
2. Transfer the local data/artifacts outside Git, or fetch the same official
   archive on the server. If transferring artifacts, verify every file against
   the saved manifest. If regenerating them, use these same CLI settings and
   audit the resulting manifest; serialization hashes can depend on dependencies.
3. Run `python -m src.validate_artifacts` against the server copy. Prepare one-epoch
   BPR and Weighted NeuMF smoke configs from seed 42, changing only epochs,
   patience and output directory, with test still sealed.
4. Run the CUDA smoke gate from the existing closeout runbook. No production
   MovieLens model has been trained locally, and GPU availability is not assumed.
5. Only after that gate passes, create a final-run commit changing only
   `evaluate_test` to true, then run the formal campaign and fixed ensemble.

The original runbook's Stage A and local Stage B checks are now implemented;
data acquisition and artifact checks from Stage C have been completed locally.
Server deployment, CUDA attachment, smoke tests, final-run commit, formal
training, ensemble evaluation, figures and final report remain pending.

## Execution notes

Python's first HTTPS download failed because its local certificate chain could
not be verified. The archive was obtained with Windows `Invoke-WebRequest`
without disabling TLS verification, then matched against the official checksum.
No ETL or training experiment crashed, timed out or was retried. The two ETL/audit
commands had 300-second hard timeouts and exited successfully. No lab server or
GPU was accessed, and existing Anime artifacts/results were not regenerated.
