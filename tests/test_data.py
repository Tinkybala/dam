import pandas as pd
import pytest

from src.data import (
    assert_split_integrity,
    build_observed_sets,
    iterative_positive_k_core,
    leave_two_out_split,
    remove_ambiguous_pairs,
    sample_evaluation_candidates,
)
from src.prepare import _fixed_user_sample, _id_mapping


def test_remove_ambiguous_pairs_removes_every_duplicate_version():
    ratings = pd.DataFrame(
        [(1, 10, 7), (1, 10, 8), (1, 11, -1), (2, 10, 9)],
        columns=["user_id", "anime_id", "rating"],
    )

    clean, count = remove_ambiguous_pairs(ratings)

    assert count == 1
    assert list(clean.itertuples(index=False, name=None)) == [(1, 11, -1), (2, 10, 9)]


def test_iterative_core_repeats_after_item_removal():
    positives = pd.DataFrame(
        [(1, 10), (1, 11), (2, 11), (2, 12)],
        columns=["user_id", "anime_id"],
    )

    core, stats = iterative_positive_k_core(positives, min_degree=2)

    assert core.empty
    assert stats.iterations == 2


def _dense_positives() -> pd.DataFrame:
    return pd.DataFrame(
        [(user, item, 8) for user in range(1, 6) for item in range(10, 15)],
        columns=["user_id", "anime_id", "rating"],
    )


def test_split_is_reproducible_and_warm():
    positives = _dense_positives()

    first = leave_two_out_split(positives, seed=42)
    second = leave_two_out_split(positives, seed=42)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)
    assert_split_integrity(first)


def test_candidates_exclude_every_observed_item():
    split = leave_two_out_split(_dense_positives(), seed=42)
    extra_items = list(range(15, 30))
    ratings = pd.concat(
        [
            _dense_positives(),
            pd.DataFrame([(1, 15, -1), (1, 16, 3)], columns=_dense_positives().columns),
        ],
        ignore_index=True,
    )
    observed = build_observed_sets(ratings, split.test["user_id"])
    warm_items = set(split.train["anime_id"]).union(extra_items)

    candidates = sample_evaluation_candidates(
        split.test, observed, warm_items, negative_count=5, seed=42
    )

    negatives_for_user_1 = set(
        candidates.loc[
            candidates["user_id"].eq(1) & candidates["label"].eq(0), "item_id"
        ]
    )
    assert negatives_for_user_1.isdisjoint({10, 11, 12, 13, 14, 15, 16})
    assert candidates.groupby("user_id")["label"].sum().eq(1).all()
    assert_split_integrity(split, test_candidates=candidates, observed_by_user=observed)


def test_candidate_generation_fails_when_unseen_pool_is_too_small():
    split = leave_two_out_split(_dense_positives(), seed=42)
    observed = {int(user): set(range(10, 15)) for user in split.test["user_id"]}

    with pytest.raises(ValueError, match="cannot sample"):
        sample_evaluation_candidates(
            split.test, observed, range(10, 17), negative_count=3, seed=42
        )


def test_id_mapping_is_sorted_and_zero_based():
    mapping = _id_mapping([10, 20, 30], "anime_id", "item_index")

    assert mapping.to_dict("list") == {
        "anime_id": [10, 20, 30],
        "item_index": [0, 1, 2],
    }


def test_fixed_development_sample_is_reproducible_and_sorted():
    first = _fixed_user_sample(list(range(100)), maximum=10, seed=42)
    second = _fixed_user_sample(list(range(100)), maximum=10, seed=42)

    assert first == second
    assert first == sorted(first)
    assert len(first) == 10
    assert len(set(first)) == 10
