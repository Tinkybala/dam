"""Read-only artifact gate; checks split structure without computing test metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    SplitResult, assert_split_integrity, build_observed_sets,
    load_anime_ratings, load_movielens_ratings, remove_ambiguous_pairs,
)
from .prepare import _sha256


def validate_artifacts(artifacts: Path, ratings: Path) -> dict[str, object]:
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    require(_sha256(ratings) == manifest["ratings_sha256"], "raw ratings checksum mismatch")
    loader = {"movielens_1m": load_movielens_ratings,
              "anime_recommendations_database": load_anime_ratings}[manifest["dataset"]]
    raw = loader(ratings)
    clean, ambiguous = remove_ambiguous_pairs(raw)
    require(len(raw) == manifest["raw_row_count"], "raw row count mismatch")
    require(ambiguous == manifest["ambiguous_pair_count"], "ambiguous pair count mismatch")
    names = ["train_positives", "validation_positives", "test_positives",
             "validation_candidates", "test_candidates", "observed_interactions",
             "user_mapping", "item_mapping", "development_users"]
    frames = {}
    for name in names:
        filename = f"{name}.parquet"
        record = manifest["files"][filename]
        require(_sha256(artifacts / filename) == record["sha256"], f"checksum mismatch: {filename}")
        frame = pd.read_parquet(artifacts / filename)
        require(len(frame) == record["rows"], f"row count mismatch: {filename}")
        require(not frame.isna().any().any(), f"missing values: {filename}")
        require(np.isfinite(frame.to_numpy()).all(), f"non-finite values: {filename}")
        require(all(pd.api.types.is_integer_dtype(dtype) for dtype in frame.dtypes),
                f"non-integer values: {filename}")
        frames[name] = frame
    train = frames["train_positives"]
    users, items = set(train.user_id), set(train.anime_id)
    require(len(users) == manifest["eligible_user_count"], "user count mismatch")
    require(len(items) == manifest["warm_item_count"], "item count mismatch")
    observed = build_observed_sets(clean, users)
    split = SplitResult(train, frames["validation_positives"], frames["test_positives"])
    assert_split_integrity(split, frames["validation_candidates"], frames["test_candidates"], observed)
    raw_positive = set(map(tuple, clean.loc[clean.rating.ge(manifest["positive_threshold"]),
                                          ["user_id", "anime_id", "rating"]].to_numpy()))
    for name in ("train", "validation", "test"):
        part = frames[f"{name}_positives"]
        require(set(part.user_id) == users, f"{name} user set mismatch")
        require(not part.duplicated(["user_id", "anime_id"]).any(), f"duplicate {name} pairs")
        require(len(part) == manifest[f"{name}_positive_count"], f"{name} count mismatch")
        require(set(map(tuple, part[["user_id", "anime_id", "rating"]].to_numpy())) <= raw_positive,
                f"{name} contains non-positive or altered ratings")
        if name == "train":
            continue
        candidates = frames[f"{name}_candidates"]
        require(set(candidates.user_id) == users, f"{name} candidate users mismatch")
        require(set(candidates.item_id) <= items, f"{name} contains cold candidates")
        require(candidates.label.isin([0, 1]).all(), f"{name} invalid labels")
        require(candidates.groupby("user_id").size().eq(manifest["negative_count"] + 1).all(),
                f"{name} candidate count mismatch")
        actual = candidates.loc[candidates.label.eq(1), ["user_id", "item_id"]]
        require(set(map(tuple, actual.to_numpy())) == set(map(tuple, part[["user_id", "anime_id"]].to_numpy())),
                f"{name} positive labels mismatch")
    for name, id_column, index_column, expected in [
        ("user_mapping", "user_id", "user_index", users),
        ("item_mapping", "anime_id", "item_index", items),
    ]:
        mapping = frames[name]
        require(len(mapping) == len(expected) and set(mapping[id_column]) == expected, f"{name} IDs mismatch")
        require(set(mapping[index_column]) == set(range(len(expected))), f"{name} indices mismatch")
    saved = frames["observed_interactions"]
    expected = clean.loc[clean.user_id.isin(users) & clean.anime_id.isin(items), ["user_id", "anime_id"]]
    require(not saved.duplicated().any(), "duplicate observed pairs")
    require(set(map(tuple, saved.to_numpy())) == set(map(tuple, expected.to_numpy())), "observed pairs mismatch")
    development = frames["development_users"]
    require(len(development) == manifest["development_user_count"] and
            not development.user_id.duplicated().any() and set(development.user_id) <= users,
            "development users mismatch")
    return {"status": "PASS", "dataset": manifest["dataset"], "users": len(users),
            "items": len(items), "train_positives": len(train),
            "candidates_per_user": manifest["negative_count"] + 1,
            "artifact_files_verified": len(frames), "test_metrics_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_artifacts(args.artifacts, args.ratings), indent=2))


if __name__ == "__main__":
    main()
