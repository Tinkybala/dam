import pandas as pd
import pytest

from src.ensemble import blend_predictions, percentile_ranks, validate_prediction_consistency


def _predictions(scores):
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "item_id": [30, 10, 20, 10, 20],
            "label": [0, 1, 0, 0, 1],
            "score": scores,
        }
    )


def test_percentile_rank_uses_item_id_for_exact_ties():
    ranks = percentile_ranks(_predictions([0.5, 0.5, 0.2, 0.7, 0.1]))
    user_one = ranks.loc[ranks["user_id"].eq(1)].sort_values("item_id")
    assert list(user_one["item_id"]) == [10, 20, 30]
    assert list(user_one["percentile"]) == [1.0, 0.0, 0.5]


def test_blend_requires_identical_keys_and_labels():
    left = _predictions([0.1, 0.2, 0.3, 0.4, 0.5])
    right = left.copy()
    right.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="consistency"):
        validate_prediction_consistency(left, right)


def test_blend_score_and_boundary_weights():
    bpr = _predictions([0.9, 0.8, 0.1, 0.7, 0.2])
    neural = _predictions([0.1, 0.8, 0.7, 0.2, 0.9])
    blended = blend_predictions(bpr, neural, 0.25)
    row = blended.loc[(blended.user_id == 1) & (blended.item_id == 30)].iloc[0]
    assert row.score == pytest.approx(0.25 * 1.0 + 0.75 * 0.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        blend_predictions(bpr, neural, 1.1)
