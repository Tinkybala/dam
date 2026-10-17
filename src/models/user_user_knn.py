"""Neighbourhood-based User-User Collaborative Filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class UserUserKNN:
    """User-User KNN using users' anime ratings as their representation.

    Users are represented by sparse rating vectors.
    Cosine distance is used to find similar users.
    """

    def __init__(self, k: int = 20):
        if k < 1:
            raise ValueError("k must be at least 1")

        self.k = k

        self.user_to_index: dict[int, int] = {}
        self.index_to_user: dict[int, int] = {}

        self.item_to_index: dict[int, int] = {}
        self.index_to_item: dict[int, int] = {}

        self.rating_matrix: csr_matrix | None = None
        self.knn_model: NearestNeighbors | None = None

        # Watched anime, including anime with rating == -1.
        self.watched_by_user: dict[int, set[int]] = {}

        self.is_fitted = False

    def fit(self, ratings: pd.DataFrame) -> "UserUserKNN":
        """Fit KNN using actual ratings.

        Parameters
        ----------
        ratings:
            DataFrame containing:
                user_id
                anime_id
                rating

            rating == -1 means watched but not rated.
            Such rows are NOT used for similarity.
        """

        required_columns = {"user_id", "anime_id", "rating"}
        missing = required_columns.difference(ratings.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        if ratings.empty:
            raise ValueError("ratings cannot be empty")

        data = ratings[
            ["user_id", "anime_id", "rating"]
        ].copy()

        data["user_id"] = data["user_id"].astype(int)
        data["anime_id"] = data["anime_id"].astype(int)
        data["rating"] = data["rating"].astype(float)

        # ---------------------------------------------------------
        # 1. Store everything the user has watched.
        # ---------------------------------------------------------
        self.watched_by_user = (
            data.groupby("user_id")["anime_id"]
            .apply(set)
            .to_dict()
        )

        # ---------------------------------------------------------
        # 2. Keep only actual ratings for the KNN representation.
        # ---------------------------------------------------------
        rated = data[data["rating"] >= 1].copy()

        if rated.empty:
            raise ValueError("No actual ratings found.")

        # ---------------------------------------------------------
        # 3. Check for duplicate user-anime ratings.
        # ---------------------------------------------------------
        if rated.duplicated(["user_id", "anime_id"]).any():
            raise ValueError(
                "Duplicate user-anime pairs found in rated data."
            )

        # ---------------------------------------------------------
        # 4. Create mappings.
        # ---------------------------------------------------------
        user_ids = sorted(rated["user_id"].unique())
        item_ids = sorted(rated["anime_id"].unique())

        self.user_to_index = {
            user_id: index
            for index, user_id in enumerate(user_ids)
        }

        self.index_to_user = {
            index: user_id
            for user_id, index in self.user_to_index.items()
        }

        self.item_to_index = {
            item_id: index
            for index, item_id in enumerate(item_ids)
        }

        self.index_to_item = {
            index: item_id
            for item_id, index in self.item_to_index.items()
        }

        # ---------------------------------------------------------
        # 5. Construct sparse User × Anime rating matrix.
        # ---------------------------------------------------------
        row_indices = rated["user_id"].map(self.user_to_index)
        col_indices = rated["anime_id"].map(self.item_to_index)

        self.rating_matrix = csr_matrix(
            (
                rated["rating"].to_numpy(dtype=np.float64),
                (
                    row_indices.to_numpy(),
                    col_indices.to_numpy(),
                ),
            ),
            shape=(len(user_ids), len(item_ids)),
        )

        # ---------------------------------------------------------
        # 6. Build KNN model using cosine distance.
        # ---------------------------------------------------------
        # We ask for k + 1 because the closest user to a user is
        # normally themselves.
        number_of_neighbours = min(
            self.k + 1,
            len(user_ids),
        )

        self.knn_model = NearestNeighbors(
            n_neighbors=number_of_neighbours,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )

        self.knn_model.fit(self.rating_matrix)

        self.is_fitted = True

        return self

    def _check_fitted(self) -> None:
        if (
            not self.is_fitted
            or self.rating_matrix is None
            or self.knn_model is None
        ):
            raise RuntimeError(
                "UserUserKNN has not been fitted. "
                "Call fit() first."
            )

    def _get_neighbours(
        self,
        user_id: int,
    ) -> list[tuple[int, float]]:
        """Return (user_id, cosine similarity) for nearest users."""

        self._check_fitted()

        if user_id not in self.user_to_index:
            raise ValueError(
                f"Unknown user_id: {user_id}"
            )

        user_index = self.user_to_index[user_id]

        distances, indices = self.knn_model.kneighbors(
            self.rating_matrix[user_index],
            n_neighbors=min(
                self.k + 1,
                len(self.user_to_index),
            ),
        )

        neighbours = []

        for distance, index in zip(
            distances[0],
            indices[0],
        ):
            neighbour_id = self.index_to_user[int(index)]

            # Remove the target user themselves.
            if neighbour_id == user_id:
                continue

            # Cosine similarity = 1 - cosine distance.
            similarity = 1.0 - float(distance)

            neighbours.append(
                (neighbour_id, similarity)
            )

            if len(neighbours) >= self.k:
                break

        return neighbours

    def get_neighbours(
        self,
        user_id: int,
    ) -> pd.DataFrame:
        """Return the K nearest users."""

        neighbours = self._get_neighbours(int(user_id))

        return pd.DataFrame(
            neighbours,
            columns=["user_id", "similarity"],
        )

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        like_threshold: float = 7,
    ) -> pd.DataFrame:
        """Recommend unseen anime using similarity-weighted votes.

        A neighbour contributes their similarity score to an anime
        if they rated that anime at or above like_threshold.
        """

        self._check_fitted()

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if user_id not in self.user_to_index:
            raise ValueError(
                f"Unknown user_id: {user_id}"
            )

        neighbours = self._get_neighbours(user_id)

        if not neighbours:
            return pd.DataFrame(
                columns=["anime_id", "score"]
            )

        # ---------------------------------------------------------
        # Score each anime according to how strongly similar
        # neighbours liked it.
        #
        # Example:
        #
        # neighbour A similarity = 0.8, liked anime X
        # neighbour B similarity = 0.6, liked anime X
        #
        # score(X) = 0.8 + 0.6 = 1.4
        # ---------------------------------------------------------
        scores = np.zeros(
            self.rating_matrix.shape[1],
            dtype=np.float64,
        )

        for neighbour_id, similarity in neighbours:

            neighbour_index = self.user_to_index[
                neighbour_id
            ]

            neighbour_ratings = (
                self.rating_matrix[neighbour_index]
                .toarray()
                .ravel()
            )

            liked_items = (
                neighbour_ratings >= like_threshold
            )

            scores += similarity * liked_items

        # ---------------------------------------------------------
        # Do not recommend anime already watched by the user.
        # This includes both:
        #   - rating == -1
        #   - actual ratings
        # ---------------------------------------------------------
        watched_items = self.watched_by_user.get(
            user_id,
            set(),
        )

        for anime_id in watched_items:
            item_index = self.item_to_index.get(anime_id)

            if item_index is not None:
                scores[item_index] = -np.inf

        # ---------------------------------------------------------
        # Remove anime that nobody in the neighbourhood liked.
        # ---------------------------------------------------------
        scores[scores == 0] = -np.inf

        # ---------------------------------------------------------
        # Rank by score.
        # ---------------------------------------------------------
        ranked_indices = np.argsort(-scores)

        recommendations = []

        for item_index in ranked_indices:

            score = scores[item_index]

            if not np.isfinite(score):
                continue

            anime_id = self.index_to_item[
                int(item_index)
            ]

            recommendations.append(
                (
                    anime_id,
                    float(score),
                )
            )

            if len(recommendations) >= top_k:
                break

        return pd.DataFrame(
            recommendations,
            columns=["anime_id", "score"],
        )