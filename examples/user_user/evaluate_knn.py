
"""Evaluate User-User KNN with Hit Rate@10 and NDCG@10."""

import numpy as np
import pandas as pd

from src.models.user_user_knn import UserUserKNN


DATA_PATH = "data/rating.csv"
USER_LIMIT = 500          # Initial pilot; increase after verifying runtime
K_VALUES = [5, 10, 20, 50]
TOP_K = 10
NUM_NEGATIVES = 99
LIKE_THRESHOLD = 7
SEED = 42


def evaluate():
    rng = np.random.default_rng(SEED)

    # 1. Load a manageable subset of users for the initial experiment.
    ratings = pd.read_csv(DATA_PATH)
    ratings = ratings[ratings["user_id"] <= USER_LIMIT].copy()

    # Keep -1 rows for watch-history filtering, but never treat -1 as a rating.
    all_seen = (
        ratings.groupby("user_id")["anime_id"].apply(set).to_dict()
    )

    actual_ratings = ratings[ratings["rating"] >= 1].copy()

    # A held-out positive must be rated >= 7 and have another actual
    # rating for the same anime, so that anime remains in the training catalog.
    item_rating_counts = actual_ratings.groupby("anime_id").size()
    eligible_items = set(item_rating_counts[item_rating_counts >= 2].index)

    positives = actual_ratings[
        (actual_ratings["rating"] >= LIKE_THRESHOLD)
        & (actual_ratings["anime_id"].isin(eligible_items))
    ]

    # Select one held-out positive per eligible user.
    heldout_rows = []
    for user_id, group in positives.groupby("user_id"):
        chosen_index = rng.choice(group.index.to_numpy())
        heldout_rows.append(ratings.loc[chosen_index])

    if not heldout_rows:
        raise RuntimeError("No eligible held-out positive ratings were found.")

    heldout = pd.DataFrame(heldout_rows).reset_index(drop=True)

    # 2. Remove held-out rows before training.
    train = ratings.drop(index=heldout.index, errors="ignore")

    # The heldout DataFrame above has reset indices, so remove by user-item pair
    # instead of relying on the reset index.
    heldout_pairs = set(
        zip(heldout["user_id"].astype(int), heldout["anime_id"].astype(int))
    )
    train = ratings[
        ~ratings.apply(
            lambda row: (
                int(row["user_id"]), int(row["anime_id"])
            ) in heldout_pairs,
            axis=1,
        )
    ].copy()

    # 3. Train a separate model for each K and calculate ranking metrics.
    results = []

    # Prepare the same candidate set for every K.
    # This makes the parameter comparison fair.
    train_items = set(
        train.loc[train["rating"] >= 1, "anime_id"].astype(int)
    )

    candidate_sets = {}
    for _, test_row in heldout.iterrows():
        user_id = int(test_row["user_id"])
        positive_item = int(test_row["anime_id"])

        # The user and positive item must exist in training.
        user_train = train[
            (train["user_id"] == user_id) & (train["rating"] >= 1)
        ]
        if user_train.empty or positive_item not in train_items:
            continue

        seen_items = all_seen.get(user_id, set())
        available_negatives = list(
            train_items - seen_items - {positive_item}
        )

        if len(available_negatives) < NUM_NEGATIVES:
            continue

        negatives = rng.choice(
            available_negatives,
            size=NUM_NEGATIVES,
            replace=False,
        ).tolist()

        candidates = [positive_item] + [int(x) for x in negatives]
        rng.shuffle(candidates)
        candidate_sets[user_id] = (positive_item, candidates)

    results = []

    for k in K_VALUES:
        model = UserUserKNN(k=k)
        model.fit(train)

        hits = []
        ndcgs = []

        for user_id, (positive_item, candidates) in candidate_sets.items():
            if user_id not in model.user_to_index:
                continue

            if positive_item not in model.item_to_index:
                continue

            neighbours = model._get_neighbours(user_id)
            candidate_scores = {item_id: 0.0 for item_id in candidates}

            for neighbour_id, similarity in neighbours:
                neighbour_index = model.user_to_index[neighbour_id]
                neighbour_row = model.rating_matrix.getrow(neighbour_index)

                for item_index, rating in zip(
                    neighbour_row.indices, neighbour_row.data
                ):
                    if rating >= LIKE_THRESHOLD:
                        item_id = model.index_to_item[int(item_index)]
                        if item_id in candidate_scores:
                            candidate_scores[item_id] += similarity

            # Candidates are shuffled once and reused for every K.
            ranked = sorted(
                candidates,
                key=lambda item_id: candidate_scores[item_id],
                reverse=True,
            )

            rank = ranked.index(positive_item) + 1
            hits.append(1.0 if rank <= TOP_K else 0.0)
            ndcgs.append(
                1.0 / np.log2(rank + 1) if rank <= TOP_K else 0.0
            )

        evaluated = len(hits)


        results.append(
            {
                "K": k,
                "Evaluated users": evaluated,
                "Hit Rate@10": np.mean(hits) if hits else np.nan,
                "NDCG@10": np.mean(ndcgs) if ndcgs else np.nan,
            }
        )

    results_df = pd.DataFrame(results)
    print("\nUser-User KNN evaluation")
    print(f"Users in initial subset: {ratings['user_id'].nunique()}")
    print(f"Held-out positive users: {len(heldout)}")
    print(f"Negative candidates per user: {NUM_NEGATIVES}")
    print(f"Evaluated candidate set: 1 positive + {NUM_NEGATIVES} negatives")
    print(results_df.to_string(index=False))

    results_df.to_csv("knn_evaluation_results.csv", index=False)
    print("\nSaved results to knn_evaluation_results.csv")


if __name__ == "__main__":
    evaluate()
