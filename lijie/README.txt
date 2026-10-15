SC4020 Project 1 - Lijie Hou: model-based recommendation

This folder is my part of the group submission. It contains the model code,
two dataset configs, and these instructions. Add the other members' code
before making the final group submission ZIP.

SETUP AND DATA
Use Python 3.11 or 3.12. From this folder:
  python -m pip install -r requirements.txt

Provide the original datasets separately:
  data/rating.csv    Anime: user_id, anime_id, rating
  data/ratings.dat   MovieLens 1M: user::movie::rating::timestamp

Prepare the fixed splits:
  python prepare.py --ratings data/rating.csv --output artifacts/anime-r7
  python prepare.py --dataset movielens-1m --ratings data/ratings.dat --output artifacts/movielens-1m-r4

The defaults use rating >= 7 (Anime) or >= 4 (MovieLens), a 5-core,
one validation and one test positive per user, and 99 unseen negatives.
Preparation uses split seed 42. Observed items are excluded from negatives.

RUN A MODEL
Choose the dataset config, model, and training seed:
  python train.py --config configs/anime.yaml --model weighted_neumf --seed 42
  python train.py --config configs/movielens.yaml --model bpr --seed 42

Anime models: popular, bpr, gmf, mlp, neumf, weighted_neumf, bpr_ensemble.
MovieLens models: popular, bpr, gmf, neumf, weighted_neumf.
The two YAML files retain the report's hyperparameters. For three-seed
experiments, repeat with --seed 43 and --seed 44; no extra YAMLs are needed.
Popular is deterministic and only needs one run. Full training defaults to
CUDA; add --device cpu if needed. Paths in YAML are relative to configs/.

For a short CPU check, use --quick (100 users, 2 epochs, validation only):
  python train.py --config configs/movielens.yaml --model bpr --quick
  python train.py --config configs/movielens.yaml --model weighted_neumf --quick
  python ensemble.py --bpr results/quickstart/movielens/bpr-seed-42/validation_predictions.parquet --neural results/quickstart/movielens/weighted_neumf-seed-42/validation_predictions.parquet --output results/quickstart/movielens/ensemble

Quick checks are examples, not the full-population results in the report.
Full outputs go to results/<dataset>/<model>-seed-<seed>/. Each run saves
metrics and validation predictions, plus test predictions for full runs.
Trainable models also save model.pt; selection uses validation NDCG@10.

FIXED ENSEMBLE
Anime uses bpr_ensemble, which has different parameters from standalone bpr:
  python train.py --config configs/anime.yaml --model bpr_ensemble --seed 42
  python train.py --config configs/anime.yaml --model weighted_neumf --seed 42
  python ensemble.py --bpr results/anime/bpr_ensemble-seed-42/test_predictions.parquet --neural results/anime/weighted_neumf-seed-42/test_predictions.parquet --output results/anime/ensemble-seed-42

For MovieLens, use bpr and weighted_neumf with configs/movielens.yaml,
then blend their matching-seed test predictions in the same way.
Fusion uses per-user percentile ranks and fixed weights 0.7 BPR + 0.3
Weighted NeuMF. Repeat with seeds 43 and 44 for the report's three-seed runs.

Validation weight search is available through python ensemble.py --help.
For the alpha ablation, change confidence_alpha in weighted_neumf, set
evaluate_test to false, and choose a separate output_dir in the dataset YAML.
Tune on validation only; keep test settings fixed. Results may vary slightly
with hardware and library versions.

CODE GUIDE
  prepare.py    Command to prepare the fixed splits
  data.py       Dataset loading, filtering, split and candidate construction
  models.py     All model definitions in one file
  train.py      Training, sampling, validation selection and test evaluation
  evaluate.py   NDCG@10 and HR@10
  ensemble.py   Fixed-weight fusion; optional validation weight comparison
  configs/      One file per dataset; choose model and seed on the command line

Validation weight comparison example (never use test predictions to tune):
  python ensemble.py --bpr results/quickstart/movielens/bpr-seed-42/validation_predictions.parquet --neural results/quickstart/movielens/weighted_neumf-seed-42/validation_predictions.parquet --weights 0.0 0.3 0.5 0.7 1.0 --output results/quickstart/movielens/weights.json

No datasets, checkpoints, predictions, caches, or historical scripts included.
