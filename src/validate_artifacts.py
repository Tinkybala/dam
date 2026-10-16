"""Check saved splits for leakage and incorrect evaluation candidates.

Checks split overlap and candidate labels in prepared data. It does not
train models, score predictions or verify file hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import (
    SplitResult,
    assert_split_integrity,
    build_observed_sets,
    load_anime_ratings,
    load_movielens_ratings,
    remove_ambiguous_pairs,
)


def validate_artifacts(artifacts: Path, ratings: Path) -> dict[str, object]:
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    loader = (load_movielens_ratings if manifest["dataset"] == "movielens_1m"
              else load_anime_ratings)
    clean, _ = remove_ambiguous_pairs(loader(ratings))
    train, validation, test = [
        pd.read_parquet(artifacts / f"{name}_positives.parquet")
        for name in ("train", "validation", "test")
    ]
    validation_candidates, test_candidates = [
        pd.read_parquet(artifacts / f"{name}_candidates.parquet")
        for name in ("validation", "test")
    ]
    users = set(train.user_id)
    items = set(train.anime_id)
    observed = build_observed_sets(clean, users)

    # Reuse the preparation checks for overlapping splits and observed negatives.
    assert_split_integrity(
        SplitResult(train, validation, test),
        validation_candidates, test_candidates, observed,
    )
    raw_positives = set(map(tuple, clean.loc[
        clean.rating.ge(manifest["positive_threshold"]),
        ["user_id", "anime_id", "rating"],
    ].to_numpy()))
    for part in (train, validation, test):
        pairs = set(map(tuple, part[["user_id", "anime_id", "rating"]].to_numpy()))
        if set(part.user_id) != users or not pairs <= raw_positives:
            raise ValueError("splits must use the same users and original positive ratings")

    for name, held_out, candidates in (
        ("validation", validation, validation_candidates),
        ("test", test, test_candidates),
    ):
        if set(candidates.user_id) != users or not set(candidates.item_id) <= items:
            raise ValueError(f"{name} candidates must use training users and items")
        if not candidates.label.isin([0, 1]).all():
            raise ValueError(f"{name} candidate labels must be 0 or 1")
        if not candidates.groupby("user_id").size().eq(manifest["negative_count"] + 1).all():
            raise ValueError(f"{name} candidate count mismatch")
        actual = candidates.loc[candidates.label.eq(1), ["user_id", "item_id"]]
        expected = held_out[["user_id", "anime_id"]]
        if set(map(tuple, actual.to_numpy())) != set(map(tuple, expected.to_numpy())):
            raise ValueError(f"{name} positive labels mismatch")

    return {"status": "PASS", "dataset": manifest["dataset"], "users": len(users),
            "items": len(items), "train_positives": len(train),
            "candidates_per_user": manifest["negative_count"] + 1,
            "test_metrics_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_artifacts(args.artifacts, args.ratings), indent=2))


if __name__ == "__main__":
    main()
