# MovieLens content-based filtering

## Pipeline

~~~text
Movie metadata
      ↓
M1–M4 movie vectors
      ↓
Rating-weighted user profiles
      ↓
Cosine similarity with fixed candidates
      ↓
Top-10 ranking
~~~

## Movie vectors

### M1: genres

~~~text
movie vector = genre TF-IDF
~~~

Pipe-separated genres remain complete labels. MovieLens 1M contains 18 genre dimensions.

### M2: genres and decade

~~~text
movie vector = [genre TF-IDF | decade one-hot]
~~~

Example:

~~~text
Toy Story (1995) → decade 1990s
~~~

### M3: genres and title

~~~text
movie vector = [genre TF-IDF | title TF-IDF]
~~~

Title processing removes the final release year, lowercases text, normalizes punctuation and whitespace, and uses word unigrams and bigrams appearing in at least two warm movies.

### M4: genres, title, and decade

~~~text
movie vector =
[genre TF-IDF | title TF-IDF | decade one-hot]
~~~

## Balanced feature blocks

Each block is L2-normalized before concatenation. The complete vector is normalized again.

For M4, a movie with all three blocks has approximately equal squared-magnitude contribution from genre, title, and decade.

This controls the title block's larger contribution observed in the Anime experiment. No block weights are tuned manually.

## User profiles

Only positive training ratings contribute.

~~~text
weight = rating - 3

rating 4 → weight 1
rating 5 → weight 2
~~~

For user u:

~~~text
profile(u) = normalize(
    sum of [weight(u, movie) × movie_vector(movie)]
)
~~~

Example:

~~~text
Toy Story = 5 → 2 × Toy Story vector
The Matrix = 5 → 2 × The Matrix vector
Alien      = 4 → 1 × Alien vector
~~~

Ratings 1–3 remain observed but do not move the profile.

## Candidate scoring

~~~text
score(user, movie)
    = cosine(user profile, movie vector)
~~~

Candidates are sorted by descending score. Exact ties use ascending movie ID.

The absolute score is not thresholded. Only rank matters.

## Selection and test

1. Score the same validation candidates with M1–M4.
2. Select the highest validation NDCG@10.
3. Score the selected configuration on test once.
4. Compare it with MovieLens Popularity, GMF, NeuMF, BPR, Weighted NeuMF, and the fixed ensemble.

## Limitations

- Genres are broad.
- Titles often represent franchise identity rather than plot.
- A single averaged profile can mix unrelated user interests.
- Ratings 1–3 are not used as negative preference directions.
- Sampled-candidate evaluation is easier than full-catalog ranking.
