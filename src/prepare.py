"""Command-line entry point for freezing Anime experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from .data import (
    assert_split_integrity,
    build_observed_sets,
    iterative_positive_k_core,
    leave_two_out_split,
    load_anime_ratings,
    positive_interactions,
    remove_ambiguous_pairs,
    sample_evaluation_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a reproducible warm-start Anime split and candidates."
    )
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-threshold", type=int, default=7)
    parser.add_argument("--core-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-count", type=int, default=99)
    parser.add_argument("--development-user-count", type=int, default=10_000)
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> dict[str, object]:
    ratings_path = args.ratings.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw = load_anime_ratings(ratings_path)
    deduplicated, ambiguous_pair_count = remove_ambiguous_pairs(raw)
    positives = positive_interactions(deduplicated, args.positive_threshold)
    core, core_stats = iterative_positive_k_core(positives, args.core_size)
    if core.empty:
        raise ValueError("positive k-core is empty")

    split = leave_two_out_split(core, args.seed)
    eligible_users = sorted(split.train["user_id"].unique())
    warm_items = sorted(split.train["anime_id"].unique())
    development_users = _fixed_user_sample(
        eligible_users, args.development_user_count, args.seed
    )
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
        "user_mapping.parquet": _id_mapping(eligible_users, "user_id", "user_index"),
        "item_mapping.parquet": _id_mapping(warm_items, "anime_id", "item_index"),
        "development_users.parquet": pd.DataFrame(
            {"user_id": development_users}
        ),
    }
    for filename, frame in frames.items():
        frame.to_parquet(output / filename, index=False)

    manifest: dict[str, object] = {
        "dataset": "anime_recommendations_database",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ratings_file": ratings_path.name,
        "ratings_sha256": _sha256(ratings_path),
        "raw_row_count": len(raw),
        "ambiguous_pair_count": ambiguous_pair_count,
        "removed_duplicate_row_count": len(raw) - len(deduplicated),
        "positive_threshold": args.positive_threshold,
        "unrated_value": -1,
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
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _id_mapping(
    values: list[int], source_column: str, index_column: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {source_column: values, index_column: range(len(values))}
    )


def _fixed_user_sample(values: list[int], maximum: int, seed: int) -> list[int]:
    if maximum < 1:
        raise ValueError("development_user_count must be positive")
    if maximum >= len(values):
        return sorted(int(value) for value in values)
    rng = np.random.default_rng(seed)
    return sorted(int(value) for value in rng.choice(values, size=maximum, replace=False))


def main() -> None:
    manifest = prepare(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
