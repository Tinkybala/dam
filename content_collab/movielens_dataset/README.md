# MovieLens content recommender

This folder transfers the Anime content-based experiment to MovieLens 1M while keeping the repository's existing split, candidate sets, and evaluation rules.

## Run order

| Step | File | Purpose |
| --- | --- | --- |
| 1 | `1_data_preparation.ipynb` | Prepare the shared split and candidates |
| 2 | `2_content_based_movielens.ipynb` | Build M1 to M4, select on validation, and evaluate once on test |
| 3 | `3_embedding_visualization.ipynb` | Inspect movie vectors, user profiles, neighbours, and score separation |

Supporting files:

- `movielens_content.py`: reusable preparation, feature, profile, scoring, and evaluation functions
- `test_movielens_content.py`: focused tests for metadata parsing, vector composition, and profile weighting
- `experiment_setup.md`: conceptual explanation of the shared evaluation protocol
- `content_based_filtering.md`: conceptual explanation of the content model

## Experiment flow

~~~text
MovieLens ratings + movie metadata
                 ↓
      shared positive 5-core
                 ↓
 train / validation / test positives
                 ↓
  1 positive + 99 fixed distractors
                 ↓
    M1–M4 content configurations
                 ↓
 select best validation NDCG@10
                 ↓
       evaluate test once
~~~

For each user, the model ranks one hidden positive movie among 99 unobserved movies.

## Content configurations

| Configuration | Movie vector |
| --- | --- |
| M1 | genre TF-IDF |
| M2 | genre TF-IDF + release-decade one-hot |
| M3 | genre TF-IDF + title TF-IDF |
| M4 | genre TF-IDF + title TF-IDF + release-decade one-hot |

Each feature block is normalized before concatenation. The complete movie vector is normalized again. User profiles are weighted averages of positive training movies, with rating 4 weighted as 1 and rating 5 weighted as 2.

## Full-data results

The following result was measured on the official MovieLens 1M archive with MD5 `c4d9eecfca2ab87c1945afe126590906`.

### Validation selection

| Configuration | NDCG@10 | Hit Rate@10 |
| --- | ---: | ---: |
| M1: genre | 0.145564 | 0.283063 |
| M2: genre + decade | 0.148506 | 0.274776 |
| M3: genre + title | 0.136788 | 0.272622 |
| M4: genre + title + decade | 0.137411 | 0.267484 |

M2 is selected because NDCG@10 is the primary metric.

### Test results

| Model | NDCG@10 | Hit Rate@10 |
| --- | ---: | ---: |
| Content M2 | 0.149743 | 0.274279 |
| Popularity baseline | 0.355596 | 0.580709 |
| BPR + Weighted NeuMF | 0.632039 | 0.838692 |

M2 scores below popularity and the ensemble on this warm-item task. The metadata features used here capture less user-specific preference than the interaction-based models.

## Reproduce

Set `MOVIELENS_1M_DIR` to the extracted folder containing `ratings.dat` and `movies.dat`, then run the notebooks in order.

~~~bash
export MOVIELENS_1M_DIR=/path/to/ml-1m
jupyter lab
~~~

If the variable is absent, the notebooks search common repository and temporary-data locations, then attempt to download the official archive.

Recorded validation environment:

- Python 3.10.12
- NumPy 2.2.6
- pandas 2.3.3
- SciPy 1.15.3
- scikit-learn 1.7.2
- Matplotlib 3.10.9

UMAP is optional. The visualization notebook uses Truncated SVD when UMAP is unavailable.

## Limitations

- The protocol evaluates warm movies only. It does not test true cold-start movies.
- The 99 distractors are unobserved, not verified dislikes.
- Ratings 1 to 3 prevent sampled leakage but do not directly shape the positive user profile.
- Sampled ranking is easier than ranking the full movie catalogue.
- Metadata comes from the official MovieLens 1M movie table.
