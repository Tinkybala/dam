from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
for path in (REPO_ROOT, MODULE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from movielens_content import (  # noqa: E402
    build_movie_features,
    build_user_profiles,
    load_movies,
)


def test_load_movies_extracts_year_title_and_decade(tmp_path):
    path = tmp_path / "movies.dat"
    path.write_text(
        "1::Toy Story (1995)::Animation|Children's|Comedy\n"
        "2::Movie Without Year::Drama\n",
        encoding="latin-1",
    )
    movies = load_movies(path)
    assert movies.loc[0, "title"] == "Toy Story"
    assert movies.loc[0, "release_year"] == 1995
    assert movies.loc[0, "release_decade"] == "1990s"
    assert movies.loc[1, "title"] == "Movie Without Year"
    assert pd.isna(movies.loc[1, "release_year"])
    assert movies.loc[1, "release_decade"] == "Unknown"


def test_feature_configurations_are_normalized_and_compositional():
    metadata = pd.DataFrame(
        {
            "movie_id": [1, 2, 3],
            "title": ["Toy Story", "Toy Story Two", "Heat Story"],
            "genres": ["Animation|Comedy", "Animation|Comedy", "Crime|Drama"],
            "release_decade": ["1990s", "1990s", "1980s"],
        }
    )
    bundle = build_movie_features(metadata)
    assert set(bundle.configurations) == {"M1", "M2", "M3", "M4"}
    assert bundle.configurations["M1"].shape[1] == 4
    assert bundle.configurations["M2"].shape[1] == 6
    assert bundle.configurations["M4"].shape[1] == (
        bundle.genre_block.shape[1]
        + bundle.title_block.shape[1]
        + bundle.decade_block.shape[1]
    )
    for matrix in bundle.configurations.values():
        norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
        assert np.allclose(norms, 1.0)


def test_weighted_profile_is_normalized():
    weights = sparse.csr_matrix([[2.0, 1.0]])
    features = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    profile = build_user_profiles(weights, features).toarray()
    expected = np.array([[2.0, 1.0]]) / np.sqrt(5.0)
    assert np.allclose(profile, expected)
