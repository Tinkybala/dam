from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from src.evaluate import aggregate_metrics


ARTIFACTS = Path("artifacts/user_user_knn_movielens")
OUTPUT = ARTIFACTS / "user_user_knn_results"
BEST_K = 50
EVAL_K = 10


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Loading MovieLens training data and test candidates...")
    train = pd.read_parquet(ARTIFACTS / "train_positives.parquet")
    dev_users = pd.read_parquet(
        ARTIFACTS / "development_users.parquet"
    )["user_id"].to_numpy()
    candidates = pd.read_parquet(
        ARTIFACTS / "test_candidates.parquet"
    )
    candidates = candidates[
        candidates["user_id"].isin(dev_users)
    ].copy()

    users = np.sort(train["user_id"].unique())
    items = np.sort(train["anime_id"].unique())
    user_to_row = {u: i for i, u in enumerate(users)}
    item_to_col = {item: i for i, item in enumerate(items)}

    rows = train["user_id"].map(user_to_row).to_numpy()
    cols = train["anime_id"].map(item_to_col).to_numpy()

    X = csr_matrix(
        (np.ones(len(train), dtype=np.float32), (rows, cols)),
        shape=(len(users), len(items)),
        dtype=np.float32,
    )
    X.data[:] = 1.0
    X.eliminate_zeros()

    print(f"Training users: {len(users):,}")
    print(f"Training movies: {len(items):,}")
    print(f"Test candidate rows: {len(candidates):,}")

    model = NearestNeighbors(
        n_neighbors=min(BEST_K + 1, len(users)),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    print("Fitting nearest-neighbour index...")
    model.fit(X)

    dev_rows = np.array([user_to_row[u] for u in dev_users])
    print("Finding neighbours...")
    distances, neighbour_rows = model.kneighbors(
        X[dev_rows],
        n_neighbors=min(BEST_K + 1, len(users)),
    )

    candidate_groups = {
        user_id: group
        for user_id, group in candidates.groupby("user_id", sort=False)
    }
    prediction_parts = []

    print(f"Scoring test candidates with K={BEST_K}...")
    for i, user_id in enumerate(dev_users):
        group = candidate_groups.get(user_id)
        if group is None:
            continue

        pairs = [
            (neighbour, 1.0 - distance)
            for neighbour, distance in zip(
                neighbour_rows[i], distances[i]
            )
            if users[neighbour] != user_id
        ][:BEST_K]

        if pairs:
            neighbours = np.array([p[0] for p in pairs])
            similarities = np.maximum(
                np.array([p[1] for p in pairs], dtype=np.float32),
                0.0,
            )
            scores = np.asarray(
                X[neighbours].T @ similarities
            ).ravel()
        else:
            scores = np.zeros(len(items), dtype=np.float32)

        item_cols = group["item_id"].map(item_to_col)
        if item_cols.isna().any():
            raise ValueError(
                f"Test candidate item missing from training mapping "
                f"for user {user_id}"
            )

        prediction = group.copy()
        prediction["score"] = scores[item_cols.to_numpy(dtype=int)]
        prediction_parts.append(prediction)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = aggregate_metrics(predictions, k=EVAL_K)

    result = {
        "dataset": "movielens_1m",
        "neighbours_k": BEST_K,
        "evaluation_k": EVAL_K,
        **metrics,
    }

    predictions.to_parquet(
        OUTPUT / "test_predictions_k50.parquet",
        index=False,
    )
    pd.DataFrame([result]).to_csv(
        OUTPUT / "test_results_k50.csv",
        index=False,
    )

    print("\nHeld-out test results:")
    print(pd.Series(result).to_string())
    print(f"\nSaved results in: {OUTPUT}")


if __name__ == "__main__":
    main()
