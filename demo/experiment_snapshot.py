"""Curated, report-only summary for the demonstration dashboard.

This is a static presentation snapshot, not a runtime reader for predictions
or test labels.  Values come from the finalized report and retain the stated
sampled-candidate limitations.
"""

EXPERIMENT_ROWS = [
    {"system": "Ensemble (w=0.7)", "ndcg_mean": 0.799658, "ndcg_sd": 0.000259, "hr_mean": 0.956335, "hr_sd": 0.000843},
    {"system": "Weighted NeuMF", "ndcg_mean": 0.783825, "ndcg_sd": 0.000137, "hr_mean": 0.946862, "hr_sd": 0.003018},
    {"system": "BPR", "ndcg_mean": 0.770767, "ndcg_sd": 0.000330, "hr_mean": 0.952703, "hr_sd": 0.000526},
    {"system": "NeuMF", "ndcg_mean": 0.765100, "ndcg_sd": 0.000851, "hr_mean": 0.947094, "hr_sd": 0.000734},
    {"system": "GMF", "ndcg_mean": 0.716820, "ndcg_sd": 0.002421, "hr_mean": 0.935043, "hr_sd": 0.001161},
    {"system": "MLP", "ndcg_mean": 0.716791, "ndcg_sd": 0.000522, "hr_mean": 0.936037, "hr_sd": 0.000538},
    {"system": "Popular", "ndcg_mean": 0.507031, "ndcg_sd": 0.0, "hr_mean": 0.771761, "hr_sd": 0.0},
]

EXPERIMENT_FACTS = {
    "validation_test_ndcg_gap": -0.000475,
    "validation_test_hr_gap": -0.000861,
    "formal_run_count": 19,
    "ensemble_result_count": 6,
    "eligible_user_count": 60384,
    "warm_item_count": 7223,
    "train_positive_count": 5087394,
}

