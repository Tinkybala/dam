import pandas as pd

from src.models.user_user_knn import UserUserKNN


print("Loading ratings...")

ratings = pd.read_csv("data/rating.csv")

# Same small dataset for now.
test_ratings = ratings[
    ratings["user_id"] <= 500
].copy()

print("Users:", test_ratings["user_id"].nunique())
print("Rows:", len(test_ratings))


for k in [5, 10, 20, 50]:

    print("\n" + "=" * 50)
    print(f"K = {k}")
    print("=" * 50)

    model = UserUserKNN(k=k)

    model.fit(test_ratings)

    neighbours = model.get_neighbours(1)

    print("\nNearest neighbours:")
    print(neighbours)

    recommendations = model.recommend(
        1,
        top_k=10,
    )

    print("\nRecommendations:")
    print(recommendations)