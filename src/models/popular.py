"""Non-personalized popularity baseline."""

from __future__ import annotations

import pandas as pd


class MostPopular:
    """Score items by their number of distinct training-positive users."""

    def __init__(self) -> None:
        self.item_scores: pd.Series | None = None

    def fit(self, train_positives: pd.DataFrame) -> "MostPopular":
        required = {"user_id", "anime_id"}
        missing = required.difference(train_positives.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        self.item_scores = train_positives.groupby("anime_id")["user_id"].nunique()
        return self

    def predict(self, item_ids: pd.Series) -> pd.Series:
        if self.item_scores is None:
            raise RuntimeError("fit must be called before predict")
        return item_ids.map(self.item_scores).fillna(0).astype(float)

