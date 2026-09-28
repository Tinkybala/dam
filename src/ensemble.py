"""Validation-only percentile-rank blending for two recommender predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import aggregate_metrics


REQUIRED_COLUMNS = {"user_id", "item_id", "label", "score"}


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
    if frame.duplicated(["user_id", "item_id"]).any():
        raise ValueError(f"{path}: duplicate user-item pairs")
    if not np.isfinite(frame["score"].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: score contains non-finite values")
    return frame[["user_id", "item_id", "label", "score"]].copy()


def _canonical_keys(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["user_id", "item_id", "label"]].sort_values(
        ["user_id", "item_id"], kind="mergesort"
    ).reset_index(drop=True)


def validate_prediction_consistency(
    bpr: pd.DataFrame, neural: pd.DataFrame
) -> None:
    """Require identical validation ``(user_id, item_id, label)`` tuples."""

    left = _canonical_keys(bpr)
    right = _canonical_keys(neural)
    if len(left) != len(right) or not np.array_equal(
        left.to_numpy(), right.to_numpy()
    ):
        raise ValueError(
            "prediction consistency check failed: "
            "(user_id, item_id, label) tuples differ"
        )


def percentile_ranks(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic per-user percentile ranks in ``[0, 1]``.

    Scores are sorted descending and exact ties by ascending item ID. The top
    item receives 1.0 and the bottom item receives 0.0 (or 1.0 for a
    single-item user).
    """

    missing = REQUIRED_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    ranked = predictions.sort_values(
        ["user_id", "score", "item_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    counts = ranked.groupby("user_id", sort=False)["item_id"].transform("size").astype(float)
    ranks = ranked.groupby("user_id", sort=False).cumcount().to_numpy(dtype=float) + 1.0
    denominators = counts.to_numpy() - 1.0
    ranked["percentile"] = np.where(
        denominators == 0.0, 1.0, (counts.to_numpy() - ranks) / denominators
    )
    return ranked[["user_id", "item_id", "percentile"]]


def blend_predictions(
    bpr: pd.DataFrame, neural: pd.DataFrame, weight: float
) -> pd.DataFrame:
    """Blend two consistent prediction frames using percentile ranks."""

    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0 and 1")
    validate_prediction_consistency(bpr, neural)
    bpr_ranks = percentile_ranks(bpr).rename(columns={"percentile": "bpr_percentile"})
    neural_ranks = percentile_ranks(neural).rename(
        columns={"percentile": "neural_percentile"}
    )
    blended = bpr_ranks.merge(
        neural_ranks, on=["user_id", "item_id"], how="inner", validate="one_to_one"
    ).merge(
        bpr[["user_id", "item_id", "label"]],
        on=["user_id", "item_id"],
        how="left",
        validate="one_to_one",
    )
    blended["score"] = weight * blended["bpr_percentile"] + (1.0 - weight) * blended[
        "neural_percentile"
    ]
    return blended[["user_id", "item_id", "label", "score"]]


def evaluate_weights(
    bpr_path: Path, neural_path: Path, weights: list[float]
) -> list[dict[str, float | int]]:
    """Evaluate validation metrics for each declared blend weight."""

    bpr = _read_predictions(bpr_path)
    neural = _read_predictions(neural_path)
    validate_prediction_consistency(bpr, neural)
    results: list[dict[str, float | int]] = []
    for weight in weights:
        blended = blend_predictions(bpr, neural, weight)
        metrics = aggregate_metrics(blended, k=10)
        results.append({"weight": float(weight), **metrics})
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpr", type=Path, required=True)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[i / 10 for i in range(11)],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_weights(args.bpr, args.neural, args.weights)
    payload = {
        "bpr_predictions": str(args.bpr),
        "neural_predictions": str(args.neural),
        "weights": results,
        "user_count": int(results[0]["user_count"]) if results else 0,
        "candidate_count": int(len(_read_predictions(args.bpr))),
        "validation_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
