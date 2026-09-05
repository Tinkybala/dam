# Model-based Top-N recommendation

This branch contains lijie's model-based collaborative-filtering workstream. The
first implemented milestone is the reproducible data and evaluation contract for
the Anime Recommendations Database.

Repository-facing experiment checkpoints are recorded in
[`EXPERIMENTS.md`](EXPERIMENTS.md). Generated datasets, predictions,
checkpoints, and raw result directories remain outside Git.

Hash-recorded final-run operational helpers are stored in [`ops/`](ops/).
They contain no access credentials or machine-specific connection settings.

## Setup

Use Python 3.11 or newer in a virtual environment:

```bash
python -m pip install -e ".[dev]"
```

On the A6000 CUDA 12.1 server, install the pinned environment with:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -c constraints/server-cu121.txt -e ".[dev]"
```

Prepare the Anime split and fixed sampled-evaluation candidates:

```bash
python -m src.prepare \
  --ratings /path/to/rating.csv \
  --output artifacts/anime-r7 \
  --positive-threshold 7 \
  --core-size 5 \
  --seed 42 \
  --negative-count 99 \
  --development-user-count 10000
```

The command removes every ambiguous duplicated user-item pair, constructs an
iterative positive k-core, holds out one validation and one test positive per
user, repairs the split so every held-out item remains visible in training, and
persists identical candidates for all models. All observed interactions,
including ratings below the positive threshold and `-1`, are excluded from
negative candidates. The command also persists zero-based user and item ID
mappings derived only from the finalized training-visible universe.
The seed-42 development-user sample is persisted as
`development_users.parquet`; every development model consumes this exact file.

Generated datasets and experiment artifacts are intentionally ignored by Git.

Run the initial baselines after preparation:

```bash
python -m src.train --config configs/anime_popular.yaml
python -m src.train --config configs/anime_bpr_dev.yaml
```

The development BPR configuration uses a seed-42 sample of 10,000 users, rejects
all observed interactions during negative sampling, uses four negatives per
positive for up to 15 epochs, and selects its checkpoint using validation
NDCG@10. Test candidates are scored only after checkpoint selection.
Development configurations omit `evaluate_test`; add
`evaluate_test: true` only to a locked final configuration.

Validate and run the equal-budget Stage A sweep with:

```bash
python -m src.tune --plan configs/tuning/anime_stage_a.yaml --dry-run
python -m src.tune --plan configs/tuning/anime_stage_a.yaml
```

The sweep assigns one trial to each configured GPU, records a complete generated
config and log per trial, and writes a validation-ranked `summary.json`. A rerun
reuses completed results only when their recorded configuration is identical;
it refuses to overwrite a stale trial. The Stage A plan allocates nine trials to
each of BPR, GMF, MLP, and NeuMF by crossing three embedding sizes with three
learning rates.

After selecting and copying the winning NeuMF settings into the weighted NeuMF
base config, run the `configs/tuning/anime_alpha.yaml` sweep for
`alpha = 0, 0.5, 1, 2`. Its `alpha = 0` result is the required standard-NeuMF
control. Neither tuning plan permits test evaluation.

Stage A winners are copied into `configs/selected/`. Run the equal-budget L2
stage with:

```bash
python -m src.tune --plan configs/tuning/anime_stage_b_l2.yaml --dry-run
python -m src.tune --plan configs/tuning/anime_stage_b_l2.yaml
```

The per-model Stage B winners are recorded in `configs/selected/*_stage_b.yaml`.
The neural dropout check is:

```bash
python -m src.tune --plan configs/tuning/anime_stage_c_dropout.yaml
```

After the dropout decision, compare identical negative-sampling choices for all
four principal models:

```bash
python -m src.tune --plan configs/tuning/anime_stage_d_negatives.yaml
```

Once Stage D is fixed, rerun confidence weighting under that selected protocol:

```bash
python -m src.tune --plan configs/tuning/anime_alpha_final.yaml
```
