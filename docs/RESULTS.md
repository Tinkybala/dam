# Recorded results and comparison scope

These are the held-out test results used in the group report.
Configurations were selected on validation data before test evaluation.

HR@10 counts whether the held-out positive appears in the top ten.
NDCG@10 also rewards placing that positive higher in the list.

## Selected configurations

| Dataset | Method | NDCG@10 | HR@10 | Evaluated users |
| --- | --- | ---: | ---: | --- |
| Anime | Content C4: genre + title | 0.259747 | 0.469114 | 60,384 |
| Anime | User-user, K = 50 | 0.6849 | 0.8695 | 10,000 development users |
| Anime | Item-item, K = 50 | 0.714 | 0.927 | Full eligible population |
| Anime | Fixed BPR–Weighted NeuMF ensemble | 0.799658 ± 0.000259 | 0.956335 ± 0.000843 | 60,384 |
| MovieLens 1M | Content M2: genre + decade | 0.149743 | 0.274279 | 6,034 |
| MovieLens 1M | User-user, K = 50 | 0.5531 | 0.7842 | All eligible MovieLens users |
| MovieLens 1M | Item-item, K = 50 | 0.538 | 0.783 | Full eligible population |
| MovieLens 1M | Fixed BPR–Weighted NeuMF ensemble | 0.632039 ± 0.003126 | 0.838692 ± 0.003120 | 6,034 |

Model-based means and sample SDs use training seeds 42, 43 and 44 on one fixed split.
Other rows are single reported measurements. SD across training seeds does not establish statistical significance.
User-user and item-item values retain their originally reported precision.

## How to read the comparison

- Validation NDCG@10 selected C4, M2 and the user-user K; the table above reports separate test scores.
- The content-based and model-based experiments use matching archived splits and candidate sets.
- The Anime user-user experiment evaluates fewer users, while its neighbour pool includes all training users.
- Split and candidate agreement across all four methods has not been checked. Treat neighbourhood comparisons as descriptive.
- All scores concern sampled candidates. They do not measure full-catalogue, temporal or cold-start accuracy.
- Different positive thresholds and populations also prevent interpreting raw Anime versus MovieLens scores as dataset difficulty.

Within the model-based experiments, the fixed ensemble improves both metrics over each component in every matched seed run.
That finding concerns the tested configurations; it does not isolate architecture, weighting or compute budget.
The Anime BPR ensemble component differs from the standalone BPR baseline.
The blend weight is 0.7 BPR / 0.3 Weighted NeuMF, transferred unchanged to MovieLens.

## Source records

- **Content-based:** [Anime results](../content_collab/anime_dataset/readme.md), [MovieLens validation and test results](../content_collab/movielens_dataset/README.md).
- **User-user:** [validation script](../src/run_user_user_knn.py), [Anime test script](../src/run_user_user_knn_test.py), [MovieLens test script](../src/run_user_user_knn_test_movielens.py).
  These scores were transcribed from the group report; full user-user result artifacts are not stored in Git.
- **Item-item:** [model](../src/models/item_item.py), [Anime configuration](../configs/final/anime_item_item_final.yaml), [MovieLens configuration](../configs/final/movielens_item_item_final.yaml).
  These scores were transcribed from the group report; full item-item result artifacts are not stored in Git.
- **Model-based:** [Anime report](../evidence/final_result_report_2026-09-05.md), [per-seed figure data](../evidence/anime_report_figure_data_20260926.json),
  [MovieLens report](../evidence/movielens_transfer_results_20260928.md), [MovieLens machine-readable summary](../evidence/movielens_transfer_summary_20260928.json).

Earlier rating-vector examples and their CSV under `examples/` are not sources for the final user-user table.
