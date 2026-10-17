import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from src.evaluate import aggregate_metrics


ARTIFACTS = Path("artifacts/user_user_knn_movielens")
OUTPUT = ARTIFACTS / "user_user_knn_results"
K_VALUES = [5, 10, 20, 50]
EVAL_K = 10


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Loading prepared MovieLens Parquet files...")
    train = pd.read_parquet(ARTIFACTS / "train_positives.parquet")
    dev_users = pd.read_parquet(
        ARTIFACTS / "development_users.parquet"
    )["user_id"].to_numpy()

    candidates = pd.read_parquet(
        ARTIFACTS / "validation_candidates.parquet"
    )
    candidates = candidates[
        candidates["user_id"].isin(dev_users)
    ].copy()

    # Build a binary user-movie interaction matrix.
    # The existing pipeline uses "anime_id" as its generic item column.
    users = np.sort(train["user_id"].unique())
    items = np.sort(train["anime_id"].unique())

    user_to_row = {user_id: i for i, user_id in enumerate(users)}
    item_to_col = {item_id: i for i, item_id in enumerate(items)}

    row_indices = train["user_id"].map(user_to_row).to_numpy()
    col_indices = train["anime_id"].map(item_to_col).to_numpy()

    X = csr_matrix(
        (
            np.ones(len(train), dtype=np.float32),
            (row_indices, col_indices),
        ),
        shape=(len(users), len(items)),
        dtype=np.float32,
    )
    X.data[:] = 1.0
    X.eliminate_zeros()

    print(f"Training users: {len(users):,}")
    print(f"Training movies: {len(items):,}")
    print(f"Training positive interactions: {X.nnz:,}")
    print(f"Development users: {len(dev_users):,}")
    print(f"Validation candidate rows: {len(candidates):,}")

    # Find the nearest neighbours for each development user.
    max_k = min(max(K_VALUES) + 1, len(users))
    model = NearestNeighbors(
        n_neighbors=max_k,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    print("Fitting nearest-neighbour index...")
    model.fit(X)

    dev_rows = np.array([user_to_row[u] for u in dev_users])
    print("Finding neighbours for development users...")
    distances, neighbour_rows = model.kneighbors(
        X[dev_rows], n_neighbors=max_k
    )

    candidate_groups = {
        user_id: group
        for user_id, group in candidates.groupby("user_id", sort=False)
    }

    result_rows = []

    for k in K_VALUES:
        print(f"Evaluating K={k}...")
        prediction_parts = []

        for i, user_id in enumerate(dev_users):
            candidate_group = candidate_groups.get(user_id)
            if candidate_group is None:
                continue

            # Exclude the target user from their own neighbours.
            pairs = [
                (neighbour, 1.0 - distance)
                for neighbour, distance in zip(
                    neighbour_rows[i], distances[i]
                )
                if users[neighbour] != user_id
            ][:k]

            if pairs:
                neighbours = np.array([p[0] for p in pairs])
                similarities = np.maximum(
                    np.array([p[1] for p in pairs], dtype=np.float32),
                    0.0,
                )

                # Score a candidate by the summed similarities of
                # neighbours who positively rated that movie.
                scores = np.asarray(
                    X[neighbours].T @ similarities
                ).ravel()
            else:
                scores = np.zeros(len(items), dtype=np.float32)

            item_cols = candidate_group["item_id"].map(item_to_col)

            if item_cols.isna().any():
                raise ValueError(
                    f"Candidate item missing from training item mapping "
                    f"for user {user_id}"
                )

            prediction = candidate_group.copy()
            prediction["score"] = scores[item_cols.to_numpy(dtype=int)]
            prediction_parts.append(prediction)

        predictions = pd.concat(prediction_parts, ignore_index=True)
        metrics = aggregate_metrics(predictions, k=EVAL_K)

        result = {
            "neighbours_k": k,
            "evaluation_k": EVAL_K,
            **metrics,
        }
        result_rows.append(result)

        predictions.to_parquet(
            OUTPUT / f"validation_predictions_k{k}.parquet",
            index=False,
        )
        print(json.dumps(result, indent=2))

    results = pd.DataFrame(result_rows)
    results.to_csv(OUTPUT / "validation_results.csv", index=False)

    metric_columns = ["ndcg@10", "hit_rate@10"]
    missing = [column for column in metric_columns if column not in results]
    if missing:
        raise ValueError(
            f"Expected evaluation metrics are missing: {missing}. "
            f"Available columns: {results.columns.tolist()}"
        )

    best = results.sort_values(
        ["ndcg@10", "hit_rate@10"],
        ascending=False,
    ).iloc[0].to_dict()

    print("\nValidation results:")
    print(results.to_string(index=False))
    print(f"\nResults saved in: {OUTPUT}")
    print("\nBest configuration (validation):")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
