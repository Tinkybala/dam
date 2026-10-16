# 🧪 Content-Based Recommender: Experiment Setup

## 🎯 Objective

The experiment asks:

> Can a content-based model rank an anime that a user is known to like above randomly sampled anime whose preference is unknown?

This is a **top-10 ranking task**.

It does not predict an exact rating or the user's literal next watch.

---

## 📦 Raw data

### Rating table

Each row contains:

~~~text
user_id, anime_id, rating
~~~

Ratings are interpreted as:

| Rating | Meaning in this experiment |
| ---: | --- |
| 7–10 | Positive preference |
| 1–6 | Observed, but not positive |
| -1 | Watched, but not rated |
| No row | Unobserved, preference unknown |

An unobserved anime is not automatically a dislike.

### Anime table

The content system may use:

- genre
- type
- episode count
- title

Community rating and members are not used as content because they describe popularity or global quality rather than the anime itself.

---

## 🧹 Data-cleaning process

### 1. Remove ambiguous duplicate pairs

If the same user-anime pair appears more than once, all rows for that pair are removed.

~~~text
7 ambiguous pairs
14 rows removed
~~~

This avoids arbitrarily deciding which duplicated rating is correct.

### 2. Define positive interactions

~~~text
rating >= 7 → positive
rating <= 6 → observed, but not positive
rating == -1 → observed and unrated
~~~

Only positives can build a user profile.

Low and unrated interactions are still retained as observed history. They prevent an already observed anime from being sampled as unseen.

### 3. Apply the iterative positive 5-core

Every retained user must have at least five positive anime.

Every retained anime must have at least five positive users.

The filter repeats until both conditions hold:

~~~text
Remove sparse users or anime
           ↓
Interaction counts change
           ↓
Repeat until stable
~~~

Consequences:

- A user with many low ratings but fewer than five ratings of 7–10 is removed.
- An anime with many low ratings but fewer than five ratings of 7–10 is removed.
- Low ratings are not used as negative training signals.

### Final shared population

| Quantity | Count |
| --- | ---: |
| Users | 60,384 |
| Warm anime | 7,223 |
| Positive interactions before splitting | 5,208,162 |

A warm anime is an anime that remains represented in training.

---

## ✂️ Training, validation, and test split

The split is performed within each user using seed 42.

~~~text
One user's positive anime
        │
        ├── all except two → training
        ├── one positive   → validation
        └── one positive   → test
~~~

| Partition | Rows | Purpose |
| --- | ---: | --- |
| Training positives | 5,087,394 | Build user profiles |
| Validation positives | 60,384 | Select the content configuration |
| Test positives | 60,384 | Final evaluation only |

The preparation code ensures every validation and test anime still appears in training.

### Concrete example

~~~text
Alice's training positives:
  Steins;Gate = 10
  Death Note = 9
  Naruto = 7

Alice's validation positive:
  Psycho-Pass = 9

Alice's test positive:
  Ghost in the Shell = 8
~~~

Only the training anime may build Alice's profile.

---

## 👤 How ratings are used during training

Only positive training interactions enter the user profile.

The rating controls confidence:

~~~text
weight = rating - 6
~~~

| Rating | Profile weight |
| ---: | ---: |
| 7 | 1 |
| 8 | 2 |
| 9 | 3 |
| 10 | 4 |

Therefore, ratings 7 and 9 are both positive, but they are not equally influential during training.

~~~text
Alice's profile = normalize(
    4 × Steins;Gate vector
  + 3 × Death Note vector
  + 1 × Naruto vector
)
~~~

Ratings 1–6 and -1 do not move the profile toward or away from an anime.

---

## 🔎 Evaluation candidates

Each validation or test case has 100 candidates:

~~~text
1 hidden positive
+
99 unobserved warm anime
=
100 candidates
~~~

The 99 unobserved anime are better described as **distractors**, not confirmed negatives.

They could be anime that the user would like, dislike, or feel neutral about. The dataset does not tell us.

Their role is to create a ranking problem. Without competing candidates, the hidden positive would always rank first.

### Candidate rules

A distractor must:

- be one of the 7,223 warm anime
- have no interaction row for that user
- differ from the hidden positive

Validation candidates use seed 42.

Test candidates use seed 43.

Every compared model receives the same fixed candidates.

---

## 📏 What validation measures

The model assigns a cosine-similarity score to all 100 candidates and sorts them.

There is no rule such as:

~~~text
cosine similarity > 0.5 → positive
~~~

Only relative rank matters.

### Low score can still succeed

~~~text
Hidden positive score = 0.32
Hidden positive rank  = 3
~~~

This is a successful top-10 result even though the score is below 0.5.

### High score can still fail

~~~text
Hidden positive score = 0.80
Hidden positive rank  = 11
~~~

This fails because ten candidates received higher scores.

Validation therefore asks:

> Which content configuration places the known hidden positive highest among the same 100 candidates?

The configuration with the highest average validation NDCG@10 is selected.

---

## 📊 Metrics

### NDCG@10: primary metric

NDCG rewards higher placement.

| Hidden-positive rank | NDCG contribution |
| ---: | ---: |
| 1 | 1.000 |
| 2 | 0.631 |
| 5 | 0.387 |
| 10 | 0.289 |
| 11 or lower | 0.000 |

### Hit Rate@10: secondary metric

~~~text
rank 1–10  → hit
rank 11+   → miss
~~~

Metrics are calculated per user and averaged, so every user has equal influence.

---

## 🔒 Validation versus test

### Validation

Validation selects the content representation.

For example:

- genre only
- genre and type
- genre, type, and episode length
- genre and title

### Test

After one configuration is selected:

1. freeze its feature definition
2. keep the training data unchanged
3. score the fixed test candidates
4. report test NDCG@10 and Hit Rate@10 once

The test result must not be used to choose another configuration.

---

## ⚠️ What the experiment does and does not measure

### It measures

> Whether a known positive is ranked above random unobserved items.

### It does not measure

- whether the model can predict exact ratings
- whether the model can detect explicit dislikes
- whether every unobserved distractor is genuinely negative
- whether the user will watch the anime next
- full-catalog ranking performance

An unobserved anime ranked above the hidden positive may actually be a good recommendation. The evaluator cannot verify it because no rating exists.

This uncertainty is a limitation of offline recommendation datasets.

---

## ✅ Data handed to the content system

After preparation, the content-based system receives:

| Data | Purpose |
| --- | --- |
| Training positives | Build weighted user profiles |
| Validation candidates | Select a content configuration |
| Test candidates | Final evaluation |
| Observed histories | Prevent observed anime from becoming distractors |
| 60,384 eligible users | Shared evaluation population |
| 7,223 warm anime | Shared candidate universe |
| Anime metadata | Construct content vectors |

At this boundary, no recommendation scores have been calculated yet.
