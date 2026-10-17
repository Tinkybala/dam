import math

import numpy as np
import pandas as pd
import pytest

from src.models import ItemItemKNN


def _train_positives() -> pd.DataFrame:
    """Four users over four items; every cosine below is hand-computable.

    item 10 is liked by {1, 2, 3}, item 20 by {1, 2}, item 30 by {1, 4}, item 40 by {4}.
    """
    return pd.DataFrame(
        [
            (1, 10, 10),
            (1, 20, 8),
            (1, 30, 7),
            (2, 10, 9),
            (2, 20, 7),
            (3, 10, 8),
            (4, 30, 10),
            (4, 40, 9),
        ],
        columns=["user_id", "anime_id", "rating"],
    )


def _movielens_positives() -> pd.DataFrame:
    """A 1-to-5 rating frame, as MovieLens artifacts supply."""
    return pd.DataFrame(
        [
            (1, 10, 5),
            (1, 20, 4),
            (2, 10, 4),
            (2, 30, 5),
            (3, 20, 5),
            (3, 30, 4),
        ],
        columns=["user_id", "anime_id", "rating"],
    )


def test_similarity_matches_hand_computed_cosine():
    model = ItemItemKNN(top_k=None).fit(_train_positives())

    expected = np.array(
        [
            [0.0, 2 / math.sqrt(6), 1 / math.sqrt(6), 0.0],
            [2 / math.sqrt(6), 0.0, 0.5, 0.0],
            [1 / math.sqrt(6), 0.5, 0.0, 1 / math.sqrt(2)],
            [0.0, 0.0, 1 / math.sqrt(2), 0.0],
        ]
    )

    assert model.item_ids.tolist() == [10, 20, 30, 40]
    assert model.user_ids.tolist() == [1, 2, 3, 4]
    assert model.similarity == pytest.approx(expected, abs=1e-6)


def test_score_modes_separate_support_from_rating_weighting():
    train = _train_positives()
    users, items = pd.Series([2]), pd.Series([30])
    support = 1 / math.sqrt(6) + 0.5
    weighted = 9 / math.sqrt(6) + 7 * 0.5

    scores = {
        mode: float(
            ItemItemKNN(top_k=None, score_mode=mode)
            .fit(train)
            .predict(users, items)
            .iloc[0]
        )
        for mode in ("similarity_sum", "rating_mean")
    }

    assert scores["similarity_sum"] == pytest.approx(support, abs=1e-5)
    assert scores["rating_mean"] == pytest.approx(weighted / support, abs=1e-5)
    assert 7.0 < scores["rating_mean"] < 10.0


def test_rating_mean_degenerates_for_a_single_item_profile():
    model = ItemItemKNN(top_k=None, score_mode="rating_mean").fit(_train_positives())

    scores = model.predict(pd.Series([3, 3]), pd.Series([20, 30]))

    assert scores.tolist() == pytest.approx([8.0, 8.0])


def test_top_k_truncation_keeps_only_the_strongest_neighbours():
    train = _train_positives()
    users, items = pd.Series([2]), pd.Series([30])

    # Item 30's neighbours are 40 (0.707), 20 (0.5) and 10 (0.408). Dropping item 10
    # leaves only item 20 in user 2's profile; dropping item 20 as well leaves none.
    two = ItemItemKNN(top_k=2, score_mode="rating_mean").fit(train)
    one = ItemItemKNN(top_k=1, score_mode="rating_mean").fit(train)

    assert float(two.predict(users, items).iloc[0]) == pytest.approx(7.0, abs=1e-5)
    assert float(one.predict(users, items).iloc[0]) == 0.0


def test_candidates_without_a_neighbour_score_zero():
    model = ItemItemKNN(top_k=None, score_mode="rating_mean").fit(_train_positives())

    scores = model.predict(pd.Series([2, 1]), pd.Series([40, 40]))

    assert scores.tolist() == pytest.approx([0.0, 7.0])


def test_unknown_users_and_items_score_zero():
    model = ItemItemKNN(top_k=None).fit(_train_positives())

    scores = model.predict(pd.Series([99, 1, 99]), pd.Series([20, 999, 999]))

    assert scores.tolist() == [0.0, 0.0, 0.0]
    assert np.isfinite(scores.to_numpy()).all()


def test_similarity_is_independent_of_the_similarity_block():
    train = _train_positives()

    matrices = [
        ItemItemKNN(top_k=None, similarity_block=block).fit(train).similarity
        for block in (1, 3, 8)
    ]

    assert np.array_equal(matrices[0], matrices[1])
    assert np.array_equal(matrices[0], matrices[2])


def test_predict_preserves_index_and_ignores_row_order():
    model = ItemItemKNN(top_k=None, score_mode="rating_mean").fit(_train_positives())
    users = pd.Series([2, 3, 2], index=["c", "a", "b"])
    items = pd.Series([30, 20, 40], index=["c", "a", "b"])

    scores = model.predict(users, items)
    shuffled = model.predict(users.iloc[::-1], items.iloc[::-1])

    assert scores.index.tolist() == ["c", "a", "b"]
    assert float(scores.loc["a"]) == pytest.approx(8.0)
    assert float(scores.loc["b"]) == 0.0
    assert shuffled.sort_index().tolist() == pytest.approx(scores.sort_index().tolist())


def test_recommend_excludes_rated_items_and_ranks_by_score():
    model = ItemItemKNN(top_k=None, score_mode="similarity_sum").fit(_train_positives())

    recommendations = model.recommend(3, top_k=2)

    assert recommendations["anime_id"].tolist() == [20, 30]
    assert recommendations["score"].tolist() == pytest.approx(
        [2 / math.sqrt(6), 1 / math.sqrt(6)], abs=1e-6
    )


def test_recommend_for_ratings_scores_a_user_outside_the_training_data():
    model = ItemItemKNN(top_k=None, score_mode="similarity_sum").fit(_train_positives())

    recommendations = model.recommend_for_ratings(
        pd.Series({10: 9, 20: 7}, name="rating"), top_k=2
    )

    assert recommendations["anime_id"].tolist() == [30, 40]
    assert float(recommendations["score"].iloc[0]) == pytest.approx(
        1 / math.sqrt(6) + 0.5, abs=1e-6
    )


def test_get_neighbours_never_returns_the_query_item():
    model = ItemItemKNN(top_k=None).fit(_train_positives())

    neighbours = model.get_neighbours(30)

    assert neighbours["anime_id"].tolist() == [40, 20, 10]
    assert neighbours["similarity"].tolist() == pytest.approx(
        [1 / math.sqrt(2), 0.5, 1 / math.sqrt(6)], abs=1e-6
    )


def test_fits_a_one_to_five_rating_frame():
    model = ItemItemKNN(top_k=None, score_mode="rating_mean").fit(_movielens_positives())

    score = float(model.predict(pd.Series([1]), pd.Series([30])).iloc[0])

    assert model.item_ids.tolist() == [10, 20, 30]
    assert score == pytest.approx(4.5, abs=1e-5)


def test_fit_summary_reports_fitted_dimensions():
    model = ItemItemKNN(top_k=2, score_mode="similarity_sum").fit(_train_positives())

    summary = model.fit_summary()

    assert summary["score_mode"] == "similarity_sum"
    assert summary["neighbourhood_size"] == 2
    assert summary["similarity_item_count"] == 4
    assert summary["similarity_user_count"] == 4
    assert summary["mean_profile_size"] == pytest.approx(2.0)


def test_fit_requires_a_rating_column_only_for_rating_modes():
    binary = _train_positives().drop(columns=["rating"])

    ItemItemKNN(score_mode="similarity_sum").fit(binary)

    with pytest.raises(ValueError, match=r"missing required columns: \['rating'\]"):
        ItemItemKNN(score_mode="rating_mean").fit(binary)


def test_fit_rejects_duplicate_pairs():
    duplicated = pd.concat([_train_positives(), _train_positives().head(1)])

    with pytest.raises(ValueError, match="unique user-item pairs"):
        ItemItemKNN().fit(duplicated)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit must be called before predict"):
        ItemItemKNN().predict(pd.Series([1]), pd.Series([10]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 0}, "top_k must be positive when supplied"),
        ({"score_mode": "cosine"}, "score_mode must be one of"),
        ({"similarity_block": 0}, "similarity_block must be positive"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ItemItemKNN(**kwargs)


def test_recommend_for_ratings_requires_a_known_item():
    model = ItemItemKNN(top_k=None).fit(_train_positives())

    with pytest.raises(ValueError, match="share no item with the fitted catalogue"):
        model.recommend_for_ratings(pd.Series({999: 8}))
