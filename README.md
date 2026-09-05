# Anime Recommendation System

This repository contains the recommendation-system work for our SC4020 Data
Analytics and Mining project. We use the Anime Recommendations Database to
compare several collaborative-filtering models and produce Top-10 anime
recommendations.

The final model is an ensemble of BPR and Weighted NeuMF. BPR contributes 70%
of the final percentile-rank score and Weighted NeuMF contributes 30%.

## Results

The final evaluation contains 60,384 users and 7,223 warm items. Each test case
ranks one held-out positive item against 99 fixed negative items.

| Model | NDCG@10 | Hit Rate@10 |
|---|---:|---:|
| BPR + Weighted NeuMF | **0.799658 ± 0.000259** | **0.956335 ± 0.000843** |
| Weighted NeuMF | 0.783825 ± 0.000137 | 0.946862 ± 0.003018 |
| BPR | 0.770767 ± 0.000330 | 0.952703 ± 0.000526 |
| NeuMF | 0.765100 ± 0.000851 | 0.947094 ± 0.000734 |
| GMF | 0.716820 ± 0.002421 | 0.935043 ± 0.001161 |
| MLP | 0.716791 ± 0.000522 | 0.936037 ± 0.000538 |
| Popular | 0.507031 | 0.771761 |

These results use sampled candidates rather than the entire catalog. They
measure offline ranking performance for known users and known items, so they do
not cover cold-start users or online recommendation quality.

## What we tested

We compared the following methods:

- Popularity baseline
- Bayesian Personalized Ranking (BPR)
- Generalized Matrix Factorization (GMF)
- Multi-Layer Perceptron (MLP)
- Neural Matrix Factorization (NeuMF)
- Confidence-weighted NeuMF
- A fixed BPR and Weighted NeuMF ensemble

Model selection and hyperparameter tuning used validation results only. The
test set remained unused until the models, random seeds and ensemble weight had
been fixed. The final campaign completed all 19 planned runs across two RTX
A6000 GPUs.

A more detailed explanation of the experiment is available in
[docs/EXPERIMENT_PIPELINE_OVERVIEW.md](docs/EXPERIMENT_PIPELINE_OVERVIEW.md).
The exact final results and checks are recorded in
[evidence/final_result_report_2026-09-05.md](evidence/final_result_report_2026-09-05.md).

## Demo

The Streamlit demo lets the viewer choose one of 20 anonymous users, see a
small part of that user's anime history, and generate ten unseen anime
recommendations. The interface can be switched between Chinese and English.

Anime posters are included for presentation only. They are not used by the
models and do not affect the ranking.

```bash
python -m pip install -e ".[demo]"
python -m streamlit run demo/app.py
```

The demo requires a local `demo_bundle` containing the frozen checkpoints and
prepared metadata. This directory is excluded from Git. Setup details are in
[demo/README.md](demo/README.md).

## Reproducing the data split

Python 3.11 or newer is required.

```bash
python -m pip install -e ".[dev]"

python -m src.prepare \
  --ratings /path/to/rating.csv \
  --output artifacts/anime-r7 \
  --positive-threshold 7 \
  --core-size 5 \
  --seed 42 \
  --negative-count 99 \
  --development-user-count 10000
```

The preparation step removes ambiguous duplicate ratings, constructs an
iterative 5-core, holds out one validation item and one test item per user, and
creates the fixed evaluation candidates used by every model.

Run the tests with:

```bash
python -m pytest
```

The complete experiment commands and server environment are documented in
[EXPERIMENTS.md](EXPERIMENTS.md).

## Repository structure

```text
configs/      experiment configurations
demo/         Streamlit interface and offline inference
docs/         explanations and runbooks
evidence/     final reports, checks and hashes
ops/          final-run helper scripts
src/          data preparation, models, training and evaluation
tests/        automated tests
```

Generated datasets, checkpoints, predictions, cached posters and credentials
are not stored in this repository.
