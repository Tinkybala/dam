"""Data preparation for leakage-free implicit-feedback experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("user_id", "anime_id", "rating")


@dataclass(frozen=True)
class CoreStats:
    input_positive_count: int
    output_positive_count: int
    input_user_count: int
    output_user_count: int
    input_item_count: int
    output_item_count: int
    iterations: int


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_anime_ratings(path: str | Path) -> pd.DataFrame:
    """Load only columns used by the collaborative-filtering pipeline."""
    frame = pd.read_csv(
        path,
        usecols=list(REQUIRED_COLUMNS),
        dtype={"user_id": "int64", "anime_id": "int64", "rating": "int16"},
    )
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("ratings contain missing user_id, anime_id, or rating")
    return frame


def remove_ambiguous_pairs(ratings: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove all rows belonging to a duplicated user-item pair.

    Keeping neither version avoids inventing a preference when duplicate rows
    disagree. The returned count is the number of distinct ambiguous pairs.
    """
    _require_columns(ratings, REQUIRED_COLUMNS)
    duplicate_rows = ratings.duplicated(["user_id", "anime_id"], keep=False)
    ambiguous_pairs = int(
        ratings.loc[duplicate_rows, ["user_id", "anime_id"]]
        .drop_duplicates()
        .shape[0]
    )
    return ratings.loc[~duplicate_rows].reset_index(drop=True), ambiguous_pairs


def positive_interactions(
    ratings: pd.DataFrame, positive_threshold: int
) -> pd.DataFrame:
    """Return declared positive preferences, retaining ratings for confidence."""
    _require_columns(ratings, REQUIRED_COLUMNS)
    return ratings.loc[
        ratings["rating"] >= positive_threshold, list(REQUIRED_COLUMNS)
    ].reset_index(drop=True)


def iterative_positive_k_core(
    positives: pd.DataFrame, min_degree: int = 5
) -> tuple[pd.DataFrame, CoreStats]:
    """Repeatedly remove users and items below ``min_degree``."""
    if min_degree < 1:
        raise ValueError("min_degree must be positive")
    _require_columns(positives, ("user_id", "anime_id"))

    work = positives.copy()
    initial = _counts(work)
    iterations = 0
    while not work.empty:
        user_degree = work.groupby("user_id", sort=False)["anime_id"].size()
        item_degree = work.groupby("anime_id", sort=False)["user_id"].size()
        keep = work["user_id"].map(user_degree).ge(min_degree) & work[
            "anime_id"
        ].map(item_degree).ge(min_degree)
        if bool(keep.all()):
            break
        work = work.loc[keep].reset_index(drop=True)
        iterations += 1

    final = _counts(work)
    stats = CoreStats(
        input_positive_count=len(positives),
        output_positive_count=len(work),
        input_user_count=initial[0],
        output_user_count=final[0],
        input_item_count=initial[1],
        output_item_count=final[1],
        iterations=iterations,
    )
    return work, stats


def leave_two_out_split(positives: pd.DataFrame, seed: int = 42) -> SplitResult:
    """Create one validation and one test positive per user.

    Held-out items are deterministically repaired when necessary so every
    validation/test item has another positive interaction in training.
    """
    _require_columns(positives, ("user_id", "anime_id"))
    counts = positives.groupby("user_id")["anime_id"].size()
    if counts.empty or int(counts.min()) < 3:
        raise ValueError("every split user must have at least three positives")
    if positives.duplicated(["user_id", "anime_id"]).any():
        raise ValueError("positive interactions must contain unique user-item pairs")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    for _, group in positives.groupby("user_id", sort=True):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        validation_indices.append(int(indices[0]))
        test_indices.append(int(indices[1]))
        train_indices.extend(int(value) for value in indices[2:])

    split = SplitResult(
        train=positives.loc[train_indices].copy().reset_index(drop=True),
        validation=positives.loc[validation_indices].copy().reset_index(drop=True),
        test=positives.loc[test_indices].copy().reset_index(drop=True),
    )
    return _repair_warm_items(split)


def build_observed_sets(
    ratings: pd.DataFrame, eligible_users: Iterable[int]
) -> dict[int, set[int]]:
    """Map eligible users to every item they have observed at any rating."""
    eligible = set(int(user) for user in eligible_users)
    relevant = ratings.loc[ratings["user_id"].isin(eligible), ["user_id", "anime_id"]]
    return {
        int(user): set(group["anime_id"].astype(int))
        for user, group in relevant.groupby("user_id", sort=False)
    }


def sample_evaluation_candidates(
    held_out: pd.DataFrame,
    observed_by_user: dict[int, set[int]],
    warm_items: Iterable[int],
    negative_count: int = 99,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a fixed candidate table with one positive and unseen negatives."""
    if negative_count < 1:
        raise ValueError("negative_count must be positive")
    _require_columns(held_out, ("user_id", "anime_id"))
    if held_out["user_id"].duplicated().any():
        raise ValueError("held_out must contain exactly one row per user")

    universe = np.array(sorted(set(int(item) for item in warm_items)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, int, int]] = []
    for record in held_out.sort_values("user_id").itertuples(index=False):
        user = int(record.user_id)
        positive_item = int(record.anime_id)
        if positive_item not in universe:
            raise ValueError(f"held-out item {positive_item} is not a warm item")
        observed = observed_by_user.get(user, set())
        available = universe[~np.isin(universe, np.fromiter(observed, dtype=np.int64))]
        if len(available) < negative_count:
            raise ValueError(
                f"user {user} has only {len(available)} unseen warm items; "
                f"cannot sample {negative_count} negatives"
            )
        negatives = rng.choice(available, size=negative_count, replace=False)
        rows.append((user, positive_item, 1))
        rows.extend((user, int(item), 0) for item in negatives)

    return pd.DataFrame(rows, columns=["user_id", "item_id", "label"])


def assert_split_integrity(
    split: SplitResult,
    validation_candidates: pd.DataFrame | None = None,
    test_candidates: pd.DataFrame | None = None,
    observed_by_user: dict[int, set[int]] | None = None,
) -> None:
    """Raise ``AssertionError`` when a persisted split violates the protocol."""
    pair_sets = [
        set(map(tuple, part[["user_id", "anime_id"]].to_numpy()))
        for part in (split.train, split.validation, split.test)
    ]
    assert pair_sets[0].isdisjoint(pair_sets[1])
    assert pair_sets[0].isdisjoint(pair_sets[2])
    assert pair_sets[1].isdisjoint(pair_sets[2])
    assert split.validation["user_id"].value_counts().eq(1).all()
    assert split.test["user_id"].value_counts().eq(1).all()
    train_items = set(split.train["anime_id"].astype(int))
    assert set(split.validation["anime_id"].astype(int)).issubset(train_items)
    assert set(split.test["anime_id"].astype(int)).issubset(train_items)

    for candidates in (validation_candidates, test_candidates):
        if candidates is None:
            continue
        assert candidates.groupby("user_id")["label"].sum().eq(1).all()
        assert not candidates.duplicated(["user_id", "item_id"]).any()
        if observed_by_user is not None:
            negatives = candidates.loc[candidates["label"].eq(0)]
            assert all(
                int(row.item_id) not in observed_by_user[int(row.user_id)]
                for row in negatives.itertuples(index=False)
            )


def _repair_warm_items(split: SplitResult) -> SplitResult:
    train = split.train.copy()
    held_parts = [split.validation.copy(), split.test.copy()]

    while True:
        train_counts = train["anime_id"].value_counts()
        repair: tuple[int, int] | None = None
        for part_index, held in enumerate(held_parts):
            for row_index, row in held.sort_values(["user_id", "anime_id"]).iterrows():
                if int(train_counts.get(row["anime_id"], 0)) == 0:
                    repair = (part_index, int(row_index))
                    break
            if repair is not None:
                break
        if repair is None:
            break

        part_index, held_index = repair
        held = held_parts[part_index]
        user = int(held.at[held_index, "user_id"])
        candidates = train.loc[train["user_id"].eq(user)].copy()
        candidates["train_item_count"] = candidates["anime_id"].map(train_counts)
        candidates = candidates.loc[candidates["train_item_count"].gt(1)].sort_values(
            ["anime_id"]
        )
        if candidates.empty:
            raise ValueError(
                f"cannot keep held-out item warm for user {user}; no safe swap exists"
            )
        train_index = int(candidates.index[0])
        train_row = train.loc[train_index].copy()
        held_row = held.loc[held_index].copy()
        train.loc[train_index, held.columns] = held_row[held.columns]
        held.loc[held_index, held.columns] = train_row[held.columns]
        held_parts[part_index] = held

    result = SplitResult(
        train=train.reset_index(drop=True),
        validation=held_parts[0].reset_index(drop=True),
        test=held_parts[1].reset_index(drop=True),
    )
    return result


def _counts(frame: pd.DataFrame) -> tuple[int, int]:
    return frame["user_id"].nunique(), frame["anime_id"].nunique()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

