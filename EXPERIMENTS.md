# Running the experiments

Run these commands from the repository root after the [installation steps](README.md#quick-start).
The tests need no datasets. Full experiments need the original data and can be expensive on CPU.

## Datasets

- **Anime:** obtain `anime.csv` and `rating.csv` from the [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database).
- **MovieLens 1M:** extract `ratings.dat` and `movies.dat` from [GroupLens](https://grouplens.org/datasets/movielens/1m/).

Use this local layout; `data/` and `artifacts/` are ignored by Git:

```text
data/
  anime.csv
  rating.csv
  ml-1m/
    ratings.dat
    movies.dat
```

The shared preparation keeps positive ratings (Anime ≥ 7; MovieLens ≥ 4), applies iterative positive 5-core filtering,
and holds out one positive for validation and one for test per user.
Each held-out positive is ranked against 99 fixed unobserved candidates.
Every observed rating is excluded from the negative pool, including low or unrated observations.

## Content-based filtering

```bash
python -m jupyterlab
```

Open the notebooks in this order:

| Dataset | Preparation and evaluation | Optional visualization |
| --- | --- | --- |
| Anime | `content_collab/anime_dataset/content_based_shared_protocol.ipynb` | `content_collab/anime_dataset/3_embedding_visualization.ipynb` |
| MovieLens | `content_collab/movielens_dataset/1_data_preparation.ipynb`, then `2_content_based_movielens.ipynb` in the same folder | `content_collab/movielens_dataset/3_embedding_visualization.ipynb` |

Set `ANIME_DATASET_DIR` to the directory containing `anime.csv` and `rating.csv`,
and `MOVIELENS_1M_DIR` to the extracted `ml-1m` folder before starting Jupyter.

```powershell
# PowerShell, from the repository root
$env:ANIME_DATASET_DIR = (Resolve-Path data).Path
$env:MOVIELENS_1M_DIR = (Resolve-Path data/ml-1m).Path
python -m jupyterlab
```

```bash
# Bash, from the repository root
export ANIME_DATASET_DIR="$PWD/data"
export MOVIELENS_1M_DIR="$PWD/data/ml-1m"
python -m jupyterlab
```

Select C1–C4 or M1–M4 using validation NDCG@10; the selected configurations are C4 and M2.
The older Anime cleaning notebook is exploratory and is separate from the reported shared-protocol experiment.

## User-user neighbourhood

These scripts use fixed artifact folders. Preparation refuses to overwrite a non-empty output folder.

```bash
python -m src.prepare --ratings data/rating.csv --output artifacts/user_user_knn_anime
python -m src.run_user_user_knn
python -m src.run_user_user_knn_test

python -m src.prepare --dataset movielens-1m --ratings data/ml-1m/ratings.dat --output artifacts/user_user_knn_movielens
python -m src.run_user_user_knn_movielens
python -m src.run_user_user_knn_test_movielens
```

Validation evaluates K = 5, 10, 20, 50; test uses the selected K = 50.
Outputs are under each artifact folder's `user_user_knn_results/` directory.
Anime evaluates the fixed 10,000 development users while retaining all training users as possible neighbours.
MovieLens evaluates all eligible users.
The formal scripts use binary positive interactions; `UserUserKNN` and [earlier examples](examples/README.md) also illustrate rating-vector recommendations.

## Item-item and model-based methods

Prepare the common artifact folders:

```bash
python -m src.prepare --ratings data/rating.csv --output artifacts/anime-r7
python -m src.prepare --dataset movielens-1m --ratings data/ml-1m/ratings.dat --output artifacts/movielens-1m-r4
```

Run the deterministic item-item models:

```bash
python -m src.train --config configs/final/anime_item_item_final.yaml
python -m src.train --config configs/final/movielens_item_item_final.yaml
```

Model-based examples:

```bash
python -m src.train --config configs/final/anime_bpr_final_seed42.yaml
python -m src.train --config configs/movielens/movielens_bpr_seed42.yaml
```

Final neural/BPR configurations use CUDA. For a small CPU smoke run, copy a YAML in the same directory,
then set `device: cpu`, `gpu_sampling: false`, `max_users: 100`, `epochs: 2` and a separate `output_dir`.
Keep `evaluate_test: false` for exploratory runs. Config paths are relative to the YAML file.
Such a smoke run checks execution; it does not reproduce the recorded full-data scores.

`metrics.json` records scores; `model.pt` stores the best validation checkpoint for trainable models.
Prediction tables are saved as Parquet files. Model-based results use seeds 42, 43 and 44.

The final Anime ensemble uses `anime_bpr_ensemble_component_final_seed*.yaml`,
not the separate standalone BPR baseline, together with Weighted NeuMF.
It combines per-user percentile ranks with weights 0.7 and 0.3.
MovieLens uses the transferred BPR configuration and the same blend.
`src.ensemble` evaluates validation blends; the original final-test helpers are in `ops/`.
The [result summary](docs/RESULTS.md) records the original full-data runs.

## Checks

```bash
python -m pytest
```

This includes shared preparation/ranking checks, model and demo checks, item-item tests, and MovieLens content-feature tests.
It does not download data, run full notebooks or repeat the GPU experiments.
