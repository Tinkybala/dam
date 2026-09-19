import pandas as pd

from src.models.user_user_knn import UserUserKNN


print("Loading ratings...")

ratings = pd.read_csv(
    "data/rating.csv"
)

print("Total rows:", len(ratings))

# Use a small subset for the first test.
#
# We take users 1-500 so the test runs quickly.
test_ratings = ratings[
    ratings["user_id"] <= 500
].copy()

print("Test rows:", len(test_ratings))
print(
    "Test users:",
    test_ratings["user_id"].nunique()
)

print("\nFitting KNN...")

model = UserUserKNN(k=5)

model.fit(test_ratings)

print("KNN fitted!")

# Find neighbours for user 1.
print("\nNearest neighbours of user 1:")

neighbours = model.get_neighbours(1)

print(neighbours)

# Generate recommendations.
print("\nRecommendations for user 1:")

recommendations = model.recommend(
    1,
    top_k=10,
)

print(recommendations)


print("\nChecking whether recommendations were already watched...")

watched = set(
    test_ratings[
        test_ratings["user_id"] == 1
    ]["anime_id"]
)

recommended = set(
    recommendations["anime_id"]
)

overlap = watched & recommended

print("Watched anime:", len(watched))
print("Recommended anime:", len(recommended))
print("Overlap:", overlap)

if overlap:
    print("WARNING: Some recommendations were already watched!")
else:
    print("PASS: No recommended anime were already watched.")