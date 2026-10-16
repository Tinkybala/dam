"""Item-item neighbourhood collaborative filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Binary profiles use a sum: normalizing by neighbour support would give
# a score of 1 for every supported candidate.
_SCORE_MODES: dict[str, tuple[str, bool]] = {
    "similarity_sum": ("binary", False),
    "rating_mean": ("rating", True),
}


class ItemItemKNN:
    """Rank items by cosine similarity to the user's training positives.

    Each item has a binary vector of users who rated it positively in training.
    Similarities use all training users in the warm population.
    """

    def __init__(
        self,
        top_k: int | None = 100,
        score_mode: str = "rating_mean",
        similarity_block: int = 8192,
    ) -> None:
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive when supplied")
        if score_mode not in _SCORE_MODES:
            raise ValueError(f"score_mode must be one of {sorted(_SCORE_MODES)}")
        if similarity_block < 1:
            raise ValueError("similarity_block must be positive")
        self.top_k = top_k
        self.score_mode = score_mode
        self.weighting, self.normalize = _SCORE_MODES[score_mode]
        self.similarity_block = similarity_block
        self.similarity: np.ndarray | None = None
        self.item_ids: np.ndarray | None = None
        self.user_ids: np.ndarray | None = None
        self.user_offsets: np.ndarray | None = None
        self.profile_items: np.ndarray | None = None
        self.profile_ratings: np.ndarray | None = None
        self.profile_weights: np.ndarray | None = None

    def fit(self, train_positives: pd.DataFrame) -> "ItemItemKNN":
        required = {"user_id", "anime_id"}
        if self.weighting == "rating":
            required = required | {"rating"}
        missing = required.difference(train_positives.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        if train_positives.empty:
            raise ValueError("train_positives must be non-empty")
        if train_positives.duplicated(["user_id", "anime_id"]).any():
            raise ValueError("train_positives must contain unique user-item pairs")

        frame = train_positives.sort_values(["user_id", "anime_id"], kind="mergesort")
        users = frame["user_id"].to_numpy(np.int64)
        items = frame["anime_id"].to_numpy(np.int64)
        self.user_ids = np.unique(users)
        self.item_ids = np.unique(items)
        rows = np.searchsorted(self.user_ids, users)
        self.profile_items = np.searchsorted(self.item_ids, items)
        self.user_offsets = np.concatenate(
            ([0], np.cumsum(np.bincount(rows, minlength=len(self.user_ids))))
        ).astype(np.int64)
        self.profile_ratings = (
            frame["rating"].to_numpy(np.float32)
            if "rating" in frame.columns
            else np.ones(len(frame), dtype=np.float32)
        )
        self.profile_weights = (
            self.profile_ratings
            if self.weighting == "rating"
            else np.ones(len(frame), dtype=np.float32)
        )
        self.similarity = self._cosine(rows)
        return self

    def predict(self, user_ids: pd.Series, item_ids: pd.Series) -> pd.Series:
        """Score candidate pairs, returning 0.0 where no neighbour supports a pair."""
        self._check_fitted()
        if len(user_ids) != len(item_ids):
            raise ValueError("user_ids and item_ids must have the same length")

        rows, known_users = _positions(self.user_ids, user_ids.to_numpy(np.int64))
        columns, known_items = _positions(self.item_ids, item_ids.to_numpy(np.int64))
        numerator = np.zeros(len(rows), dtype=np.float32)
        support = np.zeros(len(rows), dtype=np.float32)

        scorable = np.flatnonzero(known_users & known_items)
        order = scorable[np.argsort(rows[scorable], kind="stable")]
        for selection in _group_by_run(order, rows):
            user = rows[selection[0]]
            span = slice(self.user_offsets[user], self.user_offsets[user + 1])
            block = self.similarity[
                np.ix_(columns[selection], self.profile_items[span])
            ]
            numerator[selection] = block @ self.profile_weights[span]
            if self.normalize:
                support[selection] = block.sum(axis=1)

        if self.normalize:
            scores = np.where(
                support > 0.0, numerator / np.maximum(support, 1e-12), 0.0
            )
        else:
            scores = numerator
        return pd.Series(scores.astype(float), index=user_ids.index, name="score")

    def get_neighbours(self, anime_id: int, top_k: int = 10) -> pd.DataFrame:
        """Return the strongest neighbours of one item, never the item itself."""
        self._check_fitted()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        position = self._require_item(anime_id)
        row = self.similarity[position]
        order = np.argsort(-row, kind="stable")[:top_k]
        order = order[row[order] > 0.0]
        return pd.DataFrame(
            {
                "anime_id": self.item_ids[order],
                "similarity": row[order].astype(float),
            }
        )

    def user_profile(self, user_id: int) -> pd.Series:
        """Return one fitted user's training positives as ``{anime_id: rating}``."""
        self._check_fitted()
        position = self._require_user(user_id)
        span = slice(self.user_offsets[position], self.user_offsets[position + 1])
        return pd.Series(
            self.profile_ratings[span].astype(float),
            index=self.item_ids[self.profile_items[span]],
            name="rating",
        )

    def recommend(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """Return full-catalogue Top-K recommendations for a fitted user."""
        return self.recommend_for_ratings(self.user_profile(user_id), top_k=top_k)

    def recommend_for_ratings(
        self, ratings: pd.Series, top_k: int = 10
    ) -> pd.DataFrame:
        """Score the whole catalogue from an explicit ``{anime_id: rating}`` profile.

        The caller supplies the profile, so this also serves a user who is absent
        from the training data.
        """
        self._check_fitted()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        profile = pd.Series(ratings)
        columns, known = _positions(
            self.item_ids, profile.index.to_numpy(np.int64)
        )
        columns = columns[known]
        if len(columns) == 0:
            raise ValueError("ratings share no item with the fitted catalogue")
        weights = (
            profile.to_numpy(np.float32)[known]
            if self.weighting == "rating"
            else np.ones(len(columns), dtype=np.float32)
        )

        block = self.similarity[:, columns]
        numerator = block @ weights
        if self.normalize:
            support = block.sum(axis=1)
            scores = np.where(
                support > 0.0, numerator / np.maximum(support, 1e-12), 0.0
            )
        else:
            scores = numerator
        scores = scores.astype(float)
        scores[columns] = -np.inf
        order = np.argsort(-scores, kind="stable")[:top_k]
        order = order[np.isfinite(scores[order])]
        return pd.DataFrame(
            {"anime_id": self.item_ids[order], "score": scores[order]}
        )

    def fit_summary(self) -> dict[str, object]:
        """Return fitted dimensions and settings for a run's metrics record."""
        self._check_fitted()
        return {
            "score_mode": self.score_mode,
            "neighbourhood_size": self.top_k,
            "similarity_item_count": int(len(self.item_ids)),
            "similarity_user_count": int(len(self.user_ids)),
            "similarity_density": float(
                np.count_nonzero(self.similarity) / self.similarity.size
            ),
            "mean_profile_size": float(
                len(self.profile_items) / len(self.user_ids)
            ),
        }

    def _cosine(self, rows: np.ndarray) -> np.ndarray:
        item_count = len(self.item_ids)
        user_count = len(self.user_ids)
        height = min(self.similarity_block, user_count)
        gram = np.zeros((item_count, item_count), dtype=np.float32)
        product = np.empty((item_count, item_count), dtype=np.float32)
        liked = np.zeros((height, item_count), dtype=np.float32)
        for start in range(0, user_count, height):
            stop = min(start + height, user_count)
            span = slice(self.user_offsets[start], self.user_offsets[stop])
            block = liked[: stop - start]
            block.fill(0.0)
            block[rows[span] - start, self.profile_items[span]] = 1.0
            gram += np.matmul(block.T, block, out=product)

        # The diagonal holds each item's liked-user count, so read it before zeroing.
        norms = np.sqrt(np.diag(gram).copy())
        for start in range(0, item_count, 1024):
            chunk = gram[start : start + 1024]
            chunk /= np.outer(norms[start : start + 1024], norms)
        # Exclude self-similarity when building the neighbour matrix.
        np.fill_diagonal(gram, 0.0)
        if self.top_k is not None and self.top_k < item_count:
            self._truncate(gram)
        return gram

    def _truncate(self, similarity: np.ndarray) -> None:
        for start in range(0, len(similarity), 1024):
            chunk = similarity[start : start + 1024]
            threshold = np.partition(chunk, -self.top_k, axis=1)[:, -self.top_k, None]
            chunk[chunk < threshold] = 0.0

    def _check_fitted(self) -> None:
        if self.similarity is None:
            raise RuntimeError("fit must be called before predict")

    def _require_user(self, user_id: int) -> int:
        position, known = _positions(
            self.user_ids, np.array([int(user_id)], dtype=np.int64)
        )
        if not known[0]:
            raise ValueError(f"unknown user_id: {user_id}")
        return int(position[0])

    def _require_item(self, anime_id: int) -> int:
        position, known = _positions(
            self.item_ids, np.array([int(anime_id)], dtype=np.int64)
        )
        if not known[0]:
            raise ValueError(f"unknown anime_id: {anime_id}")
        return int(position[0])


def _positions(
    sorted_ids: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map IDs to positions in ``sorted_ids``, flagging those absent from it."""
    positions = np.minimum(
        np.searchsorted(sorted_ids, values), len(sorted_ids) - 1
    )
    return positions, sorted_ids[positions] == values


def _group_by_run(order: np.ndarray, rows: np.ndarray) -> list[np.ndarray]:
    """Split ``order`` into runs sharing the same ``rows`` value."""
    if len(order) == 0:
        return []
    starts = np.flatnonzero(np.concatenate(([True], np.diff(rows[order]) != 0)))
    bounds = np.append(starts, len(order))
    return [order[begin:end] for begin, end in zip(bounds[:-1], bounds[1:])]
