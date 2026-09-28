"""Ranking metrics for the one-held-out-positive evaluation protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_candidates(predictions: pd.DataFrame) -> pd.DataFrame:
    """Rank by descending score and ascending item ID for exact ties."""
    required = {"user_id", "item_id", "label", "score"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if predictions.duplicated(["user_id", "item_id"]).any():
        raise ValueError("predictions contain duplicate user-item pairs")

    ranked = predictions.sort_values(
        ["user_id", "score", "item_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ranked["rank"] = ranked.groupby("user_id", sort=False).cumcount() + 1
    return ranked


def per_user_metrics(predictions: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Compute NDCG and hit rate with exactly one positive per user."""
    if k < 1:
        raise ValueError("k must be positive")
    ranked = rank_candidates(predictions)
    positive_counts = ranked.groupby("user_id")["label"].sum()
    if not positive_counts.eq(1).all():
        raise ValueError("each user must have exactly one positive candidate")

    positives = ranked.loc[ranked["label"].eq(1), ["user_id", "rank"]].copy()
    positives[f"hit_rate@{k}"] = positives["rank"].le(k).astype(float)
    positives[f"ndcg@{k}"] = np.where(
        positives["rank"].le(k), 1.0 / np.log2(positives["rank"] + 1), 0.0
    )
    return positives.reset_index(drop=True)


def aggregate_metrics(predictions: pd.DataFrame, k: int = 10) -> dict[str, float]:
    per_user = per_user_metrics(predictions, k=k)
    return {
        f"hit_rate@{k}": float(per_user[f"hit_rate@{k}"].mean()),
        f"ndcg@{k}": float(per_user[f"ndcg@{k}"].mean()),
        "user_count": int(len(per_user)),
    }

