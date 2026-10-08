# 🎥 MovieLens Content Experiment Setup

## Objective

Rank one movie a user rated positively among 99 fixed unobserved movies.

This is a top-10 retrieval task, not exact rating prediction or chronological next-watch prediction.

## Shared protocol

The content experiment reuses Lijie's frozen MovieLens 1M protocol.

| Setting | Value |
| --- | ---: |
| Positive rating | 4 or 5 |
| Positive-core users | 6,034 |
| Warm movies | 3,125 |
| Training positives | 562,308 |
| Validation positives | 6,034 |
| Test positives | 6,034 |
| Candidates per user | 1 positive + 99 unobserved |
| Split seed | 42 |
| Validation candidate seed | 42 |
| Test candidate seed | 43 |
| Primary metric | NDCG@10 |
| Secondary metric | Hit Rate@10 |

Ratings 1–3 remain observed. They are excluded from distractor sampling but do not build positive profiles.

## Data preparation

~~~text
ratings.dat + movies.dat
          ↓
Validate IDs, ratings, titles, and genres
          ↓
Ratings 4–5 become positives
          ↓
Iterative positive 5-core
          ↓
One validation and one test positive per user
          ↓
Fixed 100-movie candidate sets
          ↓
Join all 3,125 warm movies to metadata
~~~

The timestamp is discarded because the shared protocol uses a seeded random split.

## Movie metadata

A MovieLens movie row looks like:

~~~text
1::Toy Story (1995)::Animation|Children's|Comedy
~~~

It becomes:

| Feature | Value |
| --- | --- |
| Clean title | Toy Story |
| Release year | 1995 |
| Release decade | 1990s |
| Genres | Animation, Children's, Comedy |

The release year is removed from title text before title TF-IDF.

## User split

For each retained user:

~~~text
positive movies
    ├── all except two → training
    ├── one            → validation
    └── one            → test
~~~

Training builds the profile. Validation selects M1–M4. Test evaluates the selected configuration once.

## Candidate meaning

The 99 distractors are unobserved, not confirmed dislikes. They may be liked, disliked, or unknown to the user.

Their role is to create a ranking problem.

There is no cosine threshold. Success depends on the hidden positive's rank.

## Test boundary

After validation selects one configuration:

1. Freeze the features and preprocessing.
2. Keep the training interactions unchanged.
3. Score the fixed test candidates once.
4. Report NDCG@10 and Hit Rate@10.
5. Do not change the configuration from test results.

## Interpretation

The result measures whether content retrieves a known positive above random unobserved movies under the shared 100-candidate protocol.

It does not measure full-catalog recommendation, dislike detection, or future viewing behaviour.
