from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from src.evaluate import aggregate_metrics


ARTIFACTS = Path("artifacts/user_user_knn_anime")
OUTPUT = ARTIFACTS / "user_user_knn_results"
K = 50
EVAL_K = 10


def main():
    print("Loading training data and fixed development users...")
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
        (
            np.ones(len(train), dtype=np.float32),
            (rows, cols),
        ),
        shape=(len(users), len(items)),
        dtype=np.float32,
    )
    X.data[:] = 1.0
    X.eliminate_zeros()

    print("Fitting cosine nearest-neighbour model...")
    model = NearestNeighbors(
        n_neighbors=min(K + 1, len(users)),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    model.fit(X)

    dev_rows = np.array([user_to_row[u] for u in dev_users])
    distances, neighbour_rows = model.kneighbors(X[dev_rows])

    print("Scoring test candidates...")
    candidate_groups = {
        user_id: group
        for user_id, group in candidates.groupby("user_id", sort=False)
    }

    predictions = []

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
        ][:K]

        neighbour_indices = np.array([p[0] for p in pairs])
        similarities = np.maximum(
            np.array([p[1] for p in pairs], dtype=np.float32),
            0.0,
        )

        scores = np.asarray(
            X[neighbour_indices].T @ similarities
        ).ravel()

        item_cols = group["item_id"].map(item_to_col)
        if item_cols.isna().any():
            raise ValueError(
                f"Test candidate missing from training item mapping "
                f"for user {user_id}"
            )

        scored = group.copy()
        scored["score"] = scores[item_cols.to_numpy(dtype=int)]
        predictions.append(scored)

    predictions = pd.concat(predictions, ignore_index=True)
    metrics = aggregate_metrics(predictions, k=EVAL_K)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        OUTPUT / "test_predictions_k50.parquet",
        index=False,
    )

    result = {
        "neighbours_k": K,
        "evaluation_k": EVAL_K,
        "split": "test",
        "user_count": len(predictions["user_id"].unique()),
        **metrics,
    }
    pd.DataFrame([result]).to_csv(
        OUTPUT / "test_results_k50.csv",
        index=False,
    )

    print("\nFinal test result:")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()
