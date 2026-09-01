import math

import pandas as pd
import pytest

from src.evaluate import aggregate_metrics, per_user_metrics, rank_candidates


def test_ties_are_broken_by_ascending_item_id():
    predictions = pd.DataFrame(
        [(1, 20, 1, 0.5), (1, 10, 0, 0.5), (1, 30, 0, 0.4)],
        columns=["user_id", "item_id", "label", "score"],
    )

    ranked = rank_candidates(predictions)

    assert list(ranked["item_id"]) == [10, 20, 30]
    assert list(ranked["rank"]) == [1, 2, 3]


def test_one_positive_metrics():
    predictions = pd.DataFrame(
        [
            (1, 10, 0, 0.9),
            (1, 11, 1, 0.8),
            (2, 20, 0, 0.9),
            (2, 21, 1, 0.1),
        ],
        columns=["user_id", "item_id", "label", "score"],
    )

    metrics = aggregate_metrics(predictions, k=1)
    user_metrics = per_user_metrics(predictions, k=2)

    assert metrics == {"hit_rate@1": 0.0, "ndcg@1": 0.0, "user_count": 2}
    assert user_metrics["hit_rate@2"].tolist() == [1.0, 1.0]
    assert user_metrics["ndcg@2"].tolist() == pytest.approx(
        [1 / math.log2(3), 1 / math.log2(3)]
    )


def test_metrics_require_exactly_one_positive():
    predictions = pd.DataFrame(
        [(1, 10, 0, 0.9), (1, 11, 0, 0.8)],
        columns=["user_id", "item_id", "label", "score"],
    )

    with pytest.raises(ValueError, match="exactly one positive"):
        per_user_metrics(predictions)
