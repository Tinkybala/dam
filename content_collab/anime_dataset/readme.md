# 🎬 Content-Based Filtering Algorithm

## 🎯 Objective

For each user, rank candidate anime by how closely their content matches the user's known preferences.

The model does not predict an exact rating.

~~~text
Anime metadata
      ↓
Anime feature vectors
      ↓
Weighted user profile
      ↓
Cosine similarity with candidates
      ↓
Top-10 recommendations
~~~

---

## 📥 Inputs

The algorithm receives the prepared outputs described in **experiment_setup.md**:

- 5,087,394 positive training interactions
- 60,384 users
- 7,223 warm anime
- fixed validation and test candidates
- anime metadata

Only training ratings from 7 to 10 build user profiles.

Community rating and member count are not content features.

---

## 🧱 1. Convert each anime into a vector

The implementation uses sparse, interpretable feature vectors rather than learned neural embeddings.

### Genre: TF-IDF

A genre string is split at commas:

~~~text
Action, Adventure, Sci-Fi
→
Action
Adventure
Sci-Fi
~~~

Complete labels such as **Sci-Fi** and **Slice of Life** remain single features.

The dataset has 43 genre features.

TF-IDF gives common genres less influence and distinctive genres more influence.

### Type: one-hot encoding

The six type features are:

~~~text
Movie, Music, ONA, OVA, Special, TV
~~~

For example:

~~~text
TV    → [0, 0, 0, 0, 0, 1]
Movie → [1, 0, 0, 0, 0, 0]
~~~

### Episodes: bucketed one-hot encoding

Raw episode counts are converted into seven groups:

~~~text
1
2–6
7–13
14–26
27–52
53+
Unknown
~~~

For example:

~~~text
Steins;Gate: 24 episodes → 14–26 bucket
~~~

This prevents a large raw value such as 500 episodes from dominating the vector.

### Title: TF-IDF

Titles are processed by:

1. decoding HTML entities
2. converting to lowercase
3. normalizing whitespace
4. extracting words and two-word phrases
5. removing terms appearing in only one anime

For example:

~~~text
Fullmetal Alchemist: Brotherhood
→
fullmetal
alchemist
brotherhood
fullmetal alchemist
alchemist brotherhood
~~~

Title features can identify related franchise entries, but may also produce repetitive recommendations.

---

## 🧪 2. Feature configurations

Four configurations are evaluated.

| Configuration | Anime-vector composition | Dimensions |
| --- | --- | ---: |
| C1 | Genre TF-IDF | 43 |
| C2 | Genre TF-IDF + type one-hot | 49 |
| C3 | Genre TF-IDF + type one-hot + episode bucket | 56 |
| C4 | Genre TF-IDF + title TF-IDF | 6,721 in the current data |

Feature blocks are concatenated:

~~~text
C3 vector =
[43 genre values | 6 type values | 7 episode values]
~~~

Different blocks do not need equal dimensions.

No manual weights are applied to the blocks.

After concatenation, the complete anime vector is L2-normalized:

~~~text
anime_vector = anime_vector / length(anime_vector)
~~~

The final vector has length 1.

---

## 🎞️ Concrete anime-vector example

Consider Steins;Gate:

| Attribute | Value |
| --- | --- |
| Genre | Sci-Fi, Thriller |
| Type | TV |
| Episodes | 24 |
| Title | Steins;Gate |

Its C3 vector contains:

~~~text
Genre section:
  Sci-Fi    = TF-IDF weight
  Thriller  = TF-IDF weight
  others    = 0

Type section:
  TV        = 1
  others    = 0

Episode section:
  14–26     = 1
  others    = 0
~~~

The sections are concatenated into 56 values and normalized.

---

## 👤 3. Build a user profile

A user profile is a weighted combination of anime vectors from the user's positive training history.

The weight is:

~~~text
weight = rating - 6
~~~

| Rating | Weight |
| ---: | ---: |
| 7 | 1 |
| 8 | 2 |
| 9 | 3 |
| 10 | 4 |

For user u:

~~~text
profile(u) = normalize(
    sum of [weight(u, i) × anime_vector(i)]
)
~~~

Ratings below 7 and unrated watches do not enter the profile.

### Concrete user example

Suppose Alice's training positives are:

| Anime | Rating | Weight |
| --- | ---: | ---: |
| Steins;Gate | 10 | 4 |
| Death Note | 9 | 3 |
| Naruto | 7 | 1 |

Her profile is:

~~~text
Alice_profile = normalize(
    4 × Steins;Gate_vector
  + 3 × Death_Note_vector
  + 1 × Naruto_vector
)
~~~

Steins;Gate influences the profile four times as strongly as Naruto.

The resulting vector represents Alice's overall content direction.

---

## 🔎 4. Score candidate anime

Every candidate anime is compared with the user profile using cosine similarity.

~~~text
score(user, anime)
    = cosine(user_profile, anime_vector)
~~~

Because both vectors have length 1, this is equivalent to a dot product.

A larger score means stronger content alignment.

There is no fixed rule such as:

~~~text
score > 0.5 → recommend
~~~

Only the relative order matters.

### Concrete ranking example

| Candidate | Cosine score | Rank |
| --- | ---: | ---: |
| Psycho-Pass | 0.61 | 1 |
| Monster | 0.54 | 2 |
| Cowboy Bebop | 0.48 | 3 |
| One Piece | 0.31 | 4 |

The candidates are sorted by descending score.

Exact score ties are broken by ascending anime ID.

---

## 📏 5. Select the feature configuration

C1–C4 score the same fixed validation candidates.

~~~text
C1 ┐
C2 ├── validation NDCG@10
C3 ┤
C4 ┘
        ↓
Highest value is selected
~~~

The selection rule is:

1. choose the highest validation NDCG@10
2. use Hit Rate@10 as a secondary diagnostic
3. do not use test results to change the choice

This is an ablation study. It shows whether type, episode length, or title improves genre-only ranking.

---

## 🔒 6. Final test evaluation

After selecting one configuration:

1. freeze its feature definition
2. build profiles from the unchanged training positives
3. score the fixed test candidates
4. calculate NDCG@10 and Hit Rate@10 once

The final test result estimates how well the selected content representation generalizes to different hidden positives and distractors.

---

## 📊 Experiment results

The full experiment evaluated **60,384 users**. Each user had one hidden positive anime and 99 fixed unobserved candidates in both validation and test.

### Validation ablation

| Configuration | Features | NDCG@10 | Hit Rate@10 |
| --- | --- | ---: | ---: |
| C1 | Genre | 0.218531 | 0.393349 |
| C2 | Genre + type | 0.235430 | 0.417809 |
| C3 | Genre + type + episode bucket | 0.242140 | 0.427232 |
| **C4** | **Genre + title** | **0.261212** | **0.470356** |

Adding type and episode length improved the genre-only model. Title TF-IDF produced the largest improvement, so **C4 was selected using validation NDCG@10**.

### Final test result

| Selected configuration | NDCG@10 | Hit Rate@10 | Users |
| --- | ---: | ---: | ---: |
| **C4: Genre + title** | **0.259747** | **0.469114** | 60,384 |

The hidden positive appeared in the top 10 for approximately **46.9% of users**. NDCG@10 was lower because it also rewards placing the positive near the very top of the list.

### Popularity reference

The shared popularity baseline achieved validation NDCG@10 of **0.505679** and Hit Rate@10 of **0.770684**.

This does not invalidate the content model. It shows that, for warm anime with substantial interaction history, popularity is a strong signal that genre, type, episode length, and title alone do not capture. Content representations may still help explain recommendations and support items with limited interaction data, but this experiment does not directly test cold-start performance.

### Main conclusion

~~~text
Genre only
    ↓ add type and episode information
Moderate improvement
    ↓ add title words and phrases
Best content configuration: C4
    ↓ compare with popularity
Content alone remains insufficient for the strongest warm-item ranking
~~~

The next practical step would be a hybrid recommender that combines content similarity with popularity or collaborative scores.

---

## ⚠️ Special metadata case

Anime 33242, **IS: Infinite Stratos 2 - Infinite Wedding**, has no genre but remains in the shared anime universe.

| Configuration | Representation |
| --- | --- |
| C1 | All-zero genre vector |
| C2 | Type remains available |
| C3 | Type and episode bucket remain available |
| C4 | Title remains available |

The item is never removed from validation or test candidates.

---

## ✅ Algorithm summary

~~~text
For each configuration C1–C4:

1. Encode all 7,223 anime
2. Concatenate the selected feature blocks
3. L2-normalize each anime vector
4. Build each user profile from weighted training positives
5. L2-normalize each user profile
6. Score the fixed validation candidates with cosine similarity
7. Calculate validation NDCG@10 and Hit Rate@10

Select the highest-validation-NDCG configuration

8. Score its fixed test candidates once
9. Report test NDCG@10 and Hit Rate@10
~~~

The algorithm recommends anime whose engineered content vectors point in a similar direction to the user's rating-weighted preference vector.

