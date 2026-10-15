from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data import (
    assert_split_integrity,
    build_observed_sets,
    iterative_positive_k_core,
    leave_two_out_split,
    positive_interactions,
    remove_ambiguous_pairs,
    sample_evaluation_candidates,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def id_mapping(values: list[int], source_name: str, index_name: str):
    return pd.DataFrame({
        source_name: values,
        index_name: range(len(values)),
    })


def prepare(args: argparse.Namespace) -> dict[str, object]:
    ratings_path = args.ratings.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # MovieLens format: UserID::MovieID::Rating::Timestamp.
    # Rename MovieID to anime_id internally to reuse the existing pipeline.
    raw = pd.read_csv(
        ratings_path,
        sep=r"::",
        engine="python",
        header=None,
        names=["user_id", "anime_id", "rating", "timestamp"],
        usecols=[0, 1, 2],
        dtype={
            "user_id": "int64",
            "anime_id": "int64",
            "rating": "int16",
        },
    )

    if raw[["user_id", "anime_id", "rating"]].isna().any().any():
        raise ValueError("MovieLens ratings contain missing required values")

    deduplicated, ambiguous_pair_count = remove_ambiguous_pairs(raw)
    positives = positive_interactions(
        deduplicated, args.positive_threshold
    )
    core, core_stats = iterative_positive_k_core(
        positives, args.core_size
    )

    if core.empty:
        raise ValueError("Positive k-core is empty")

    split = leave_two_out_split(core, args.seed)
    eligible_users = sorted(split.train["user_id"].unique().tolist())
    warm_items = sorted(split.train["anime_id"].unique().tolist())

    # Use every eligible user; MovieLens has fewer than 10,000 users.
    development_users = eligible_users

    # Exclude all movies previously rated by each user, including
    # movies with ratings below the positive threshold.
    observed = build_observed_sets(deduplicated, eligible_users)
    observed_interactions = deduplicated.loc[
        deduplicated["user_id"].isin(eligible_users)
        & deduplicated["anime_id"].isin(warm_items),
        ["user_id", "anime_id"],
    ].drop_duplicates()

    validation_candidates = sample_evaluation_candidates(
        split.validation,
        observed,
        warm_items,
        args.negative_count,
        args.seed,
    )
    test_candidates = sample_evaluation_candidates(
        split.test,
        observed,
        warm_items,
        args.negative_count,
        args.seed + 1,
    )

    assert_split_integrity(
        split, validation_candidates, test_candidates, observed
    )

    frames = {
        "train_positives.parquet": split.train,
        "validation_positives.parquet": split.validation,
        "test_positives.parquet": split.test,
        "validation_candidates.parquet": validation_candidates,
        "test_candidates.parquet": test_candidates,
        "observed_interactions.parquet": observed_interactions,
        "user_mapping.parquet": id_mapping(
            eligible_users, "user_id", "user_index"
        ),
        "item_mapping.parquet": id_mapping(
            warm_items, "anime_id", "item_index"
        ),
        "development_users.parquet": pd.DataFrame(
            {"user_id": development_users}
        ),
    }

    for filename, frame in frames.items():
        frame.to_parquet(output / filename, index=False)

    manifest = {
        "dataset": "movielens_1m",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ratings_file": ratings_path.name,
        "ratings_sha256": sha256_file(ratings_path),
        "raw_row_count": len(raw),
        "ambiguous_pair_count": ambiguous_pair_count,
        "removed_duplicate_row_count": len(raw) - len(deduplicated),
        "positive_threshold": args.positive_threshold,
        "positive_core_size": args.core_size,
        "split_seed": args.seed,
        "candidate_seed_validation": args.seed,
        "candidate_seed_test": args.seed + 1,
        "negative_count": args.negative_count,
        "eligible_user_count": len(eligible_users),
        "development_user_count": len(development_users),
        "warm_item_count": len(warm_items),
        "train_positive_count": len(split.train),
        "validation_positive_count": len(split.validation),
        "test_positive_count": len(split.test),
        "observed_interaction_count": len(observed_interactions),
        "core": asdict(core_stats),
        "observed_items_excluded_from_negatives": True,
        "item_id_column_internal": "anime_id",
    }

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MovieLens 1M for user-user KNN."
    )
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-threshold", type=int, default=4)
    parser.add_argument("--core-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-count", type=int, default=99)
    args = parser.parse_args()

    manifest = prepare(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
