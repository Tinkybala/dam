"""Shared MovieLens 1M content-recommendation utilities.

Uses the shared preparation and evaluation functions to match the
model-based MovieLens split, candidates and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import shutil
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, normalize

from src.data import (
    SplitResult,
    assert_split_integrity,
    build_observed_sets,
    iterative_positive_k_core,
    leave_two_out_split,
    load_movielens_ratings,
    positive_interactions,
    remove_ambiguous_pairs,
    sample_evaluation_candidates,
)
from src.evaluate import per_user_metrics


SOURCE_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SOURCE_MD5 = "c4d9eecfca2ab87c1945afe126590906"
SEED = 42
POSITIVE_THRESHOLD = 4
CORE_SIZE = 5
NEGATIVE_COUNT = 99


@dataclass
class SharedMovieLensData:
    dataset_dir: Path
    movies_raw: pd.DataFrame
    metadata: pd.DataFrame
    ratings_raw: pd.DataFrame
    deduplicated_ratings: pd.DataFrame
    split: SplitResult
    eligible_users: list[int]
    warm_items: list[int]
    observed_by_user: dict[int, set[int]]
    validation_candidates: pd.DataFrame
    test_candidates: pd.DataFrame
    ambiguous_pair_count: int
    core_stats: object


@dataclass
class MovieFeatureBundle:
    configurations: dict[str, sparse.csr_matrix]
    genre_block: sparse.csr_matrix
    title_block: sparse.csr_matrix
    decade_block: sparse.csr_matrix
    genre_names: np.ndarray
    title_names: np.ndarray
    decade_names: np.ndarray


def resolve_dataset_dir(
    configured: str | Path | None,
    repo_root: Path,
    auto_download: bool = True,
) -> Path:
    """Find a MovieLens 1M directory, optionally downloading the official ZIP."""
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            repo_root / "data" / "movielens-1m" / "ml-1m",
            repo_root / "data" / "movielens-1m",
            Path("/tmp/dam-movielens-1m/ml-1m"),
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if _has_dataset_files(candidate):
            return candidate
        if _has_dataset_files(candidate / "ml-1m"):
            return (candidate / "ml-1m").resolve()

    if not auto_download:
        raise FileNotFoundError(
            "MovieLens 1M was not found. Set MOVIELENS_1M_DIR to the folder "
            "containing ratings.dat and movies.dat."
        )

    target = repo_root / "data" / "movielens-1m"
    return download_movielens_1m(target)


def download_movielens_1m(target: Path) -> Path:
    """Download and verify the official MovieLens 1M archive."""
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "ml-1m.zip"
    if not archive.exists() or _md5(archive) != SOURCE_MD5:
        temporary = target / "ml-1m.zip.partial"
        with urlopen(SOURCE_URL, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        if _md5(temporary) != SOURCE_MD5:
            raise ValueError("MovieLens archive MD5 does not match the official value")
        temporary.replace(archive)
    with ZipFile(archive) as zipped:
        zipped.extractall(target)
    dataset_dir = target / "ml-1m"
    if not _has_dataset_files(dataset_dir):
        raise FileNotFoundError("Downloaded archive does not contain ratings.dat and movies.dat")
    return dataset_dir.resolve()


def load_movies(path: str | Path) -> pd.DataFrame:
    """Load movies.dat and extract clean title, release year, and decade."""
    movies = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title_with_year", "genres"],
        encoding="latin-1",
        dtype={"movie_id": "int64", "title_with_year": str, "genres": str},
    )
    if movies.isna().any().any():
        raise ValueError("movies.dat contains missing movie ID, title, or genre values")
    if movies["movie_id"].duplicated().any():
        raise ValueError("movies.dat contains duplicate movie IDs")
    extracted = movies["title_with_year"].str.extract(r"\((\d{4})\)\s*$")[0]
    movies["release_year"] = pd.to_numeric(extracted, errors="coerce").astype("Int64")
    movies["title"] = (
        movies["title_with_year"]
        .str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
        .str.strip()
    )
    movies["release_decade"] = movies["release_year"].map(
        lambda year: "Unknown" if pd.isna(year) else f"{int(year) // 10 * 10}s"
    )
    return movies


def prepare_shared_data(dataset_dir: str | Path) -> SharedMovieLensData:
    """Recreate the frozen MovieLens split and candidate protocol in memory."""
    dataset_dir = Path(dataset_dir).resolve()
    ratings_raw = load_movielens_ratings(dataset_dir / "ratings.dat")
    movies_raw = load_movies(dataset_dir / "movies.dat")
    deduplicated, ambiguous_pair_count = remove_ambiguous_pairs(ratings_raw)
    positives = positive_interactions(deduplicated, POSITIVE_THRESHOLD)
    core, core_stats = iterative_positive_k_core(positives, CORE_SIZE)
    split = leave_two_out_split(core, SEED)

    eligible_users = sorted(int(value) for value in split.train["user_id"].unique())
    warm_items = sorted(int(value) for value in split.train["anime_id"].unique())
    observed_by_user = build_observed_sets(deduplicated, eligible_users)
    validation_candidates = sample_evaluation_candidates(
        split.validation,
        observed_by_user,
        warm_items,
        negative_count=NEGATIVE_COUNT,
        seed=SEED,
    )
    test_candidates = sample_evaluation_candidates(
        split.test,
        observed_by_user,
        warm_items,
        negative_count=NEGATIVE_COUNT,
        seed=SEED + 1,
    )
    assert_split_integrity(
        split,
        validation_candidates,
        test_candidates,
        observed_by_user,
    )

    metadata = (
        movies_raw.set_index("movie_id")
        .reindex(warm_items)
        .rename_axis("movie_id")
        .reset_index()
    )
    if metadata[["movie_id", "title", "genres"]].isna().any().any():
        missing = metadata.loc[metadata["title"].isna(), "movie_id"].tolist()
        raise ValueError(f"warm movies are missing metadata: {missing[:10]}")

    return SharedMovieLensData(
        dataset_dir=dataset_dir,
        movies_raw=movies_raw,
        metadata=metadata,
        ratings_raw=ratings_raw,
        deduplicated_ratings=deduplicated,
        split=split,
        eligible_users=eligible_users,
        warm_items=warm_items,
        observed_by_user=observed_by_user,
        validation_candidates=validation_candidates,
        test_candidates=test_candidates,
        ambiguous_pair_count=ambiguous_pair_count,
        core_stats=core_stats,
    )


def build_movie_features(metadata: pd.DataFrame) -> MovieFeatureBundle:
    """Build block-normalized M1-M4 sparse movie vectors."""
    genre_vectorizer = TfidfVectorizer(
        analyzer=_split_genres,
        lowercase=False,
        norm="l2",
        dtype=np.float32,
    )
    genre_block = genre_vectorizer.fit_transform(metadata["genres"]).tocsr()

    title_vectorizer = TfidfVectorizer(
        preprocessor=_clean_title,
        token_pattern=r"(?u)\b\w[\w']+\b",
        ngram_range=(1, 2),
        min_df=2,
        norm="l2",
        dtype=np.float32,
    )
    title_block = title_vectorizer.fit_transform(metadata["title"]).tocsr()

    decade_categories = sorted(metadata["release_decade"].astype(str).unique())
    decade_encoder = OneHotEncoder(
        categories=[decade_categories],
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )
    decade_block = decade_encoder.fit_transform(
        metadata[["release_decade"]].astype(str)
    ).tocsr()

    raw = {
        "M1": genre_block,
        "M2": sparse.hstack([genre_block, decade_block], format="csr"),
        "M3": sparse.hstack([genre_block, title_block], format="csr"),
        "M4": sparse.hstack(
            [genre_block, title_block, decade_block], format="csr"
        ),
    }
    configurations = {
        name: normalize(matrix, axis=1).tocsr().astype(np.float32)
        for name, matrix in raw.items()
    }
    return MovieFeatureBundle(
        configurations=configurations,
        genre_block=genre_block,
        title_block=title_block,
        decade_block=decade_block,
        genre_names=genre_vectorizer.get_feature_names_out(),
        title_names=title_vectorizer.get_feature_names_out(),
        decade_names=np.asarray(decade_categories),
    )


def build_user_item_weights(
    shared: SharedMovieLensData,
) -> tuple[sparse.csr_matrix, dict[int, int], dict[int, int]]:
    """Build the rating-confidence matrix: rating 4 -> 1, rating 5 -> 2."""
    user_to_index = {
        user_id: index for index, user_id in enumerate(shared.eligible_users)
    }
    item_to_index = {
        item_id: index for index, item_id in enumerate(shared.warm_items)
    }
    train = shared.split.train
    matrix = sparse.csr_matrix(
        (
            train["rating"].to_numpy(dtype=np.float32) - 3.0,
            (
                train["user_id"].map(user_to_index).to_numpy(),
                train["anime_id"].map(item_to_index).to_numpy(),
            ),
        ),
        shape=(len(shared.eligible_users), len(shared.warm_items)),
        dtype=np.float32,
    )
    return matrix, user_to_index, item_to_index


def build_user_profiles(
    user_item_weights: sparse.csr_matrix,
    item_features: sparse.csr_matrix,
) -> sparse.csr_matrix:
    return normalize(user_item_weights @ item_features, axis=1).tocsr()


def score_candidates(
    candidates: pd.DataFrame,
    profiles: sparse.csr_matrix,
    item_features: sparse.csr_matrix,
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    batch_size: int = 250_000,
) -> pd.DataFrame:
    """Score fixed candidates with cosine similarity in bounded batches."""
    users = candidates["user_id"].map(user_to_index)
    items = candidates["item_id"].map(item_to_index)
    if users.isna().any() or items.isna().any():
        raise ValueError("candidate contains a user or movie outside the warm mapping")
    user_indices = users.to_numpy(dtype=np.int32)
    item_indices = items.to_numpy(dtype=np.int32)
    scores = np.empty(len(candidates), dtype=np.float32)
    for start in range(0, len(candidates), batch_size):
        stop = min(start + batch_size, len(candidates))
        scores[start:stop] = np.asarray(
            profiles[user_indices[start:stop]]
            .multiply(item_features[item_indices[start:stop]])
            .sum(axis=1)
        ).ravel()
    predictions = candidates.copy()
    predictions["score"] = scores
    return predictions


def evaluate_predictions(
    predictions: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    users = per_user_metrics(predictions, k=10)
    return (
        {
            "ndcg@10": float(users["ndcg@10"].mean()),
            "hit_rate@10": float(users["hit_rate@10"].mean()),
            "user_count": int(len(users)),
        },
        users,
    )


def assert_archived_counts(shared: SharedMovieLensData) -> None:
    expected = {
        "raw ratings": (len(shared.ratings_raw), 1_000_209),
        "ambiguous pairs": (shared.ambiguous_pair_count, 0),
        "positive core": (shared.core_stats.output_positive_count, 574_376),
        "users": (len(shared.eligible_users), 6_034),
        "movies": (len(shared.warm_items), 3_125),
        "train positives": (len(shared.split.train), 562_308),
        "validation positives": (len(shared.split.validation), 6_034),
        "test positives": (len(shared.split.test), 6_034),
        "validation candidates": (len(shared.validation_candidates), 603_400),
        "test candidates": (len(shared.test_candidates), 603_400),
    }
    failures = [
        f"{name}: expected {wanted:,}, got {actual:,}"
        for name, (actual, wanted) in expected.items()
        if int(actual) != wanted
    ]
    if failures:
        raise AssertionError("; ".join(failures))


def _split_genres(value: str) -> list[str]:
    if not value or value == "(no genres listed)":
        return []
    return [genre.strip() for genre in str(value).split("|") if genre.strip()]


def _clean_title(value: str) -> str:
    value = str(value).lower()
    value = re.sub(r"[^\w']+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _has_dataset_files(path: Path) -> bool:
    return (path / "ratings.dat").is_file() and (path / "movies.dat").is_file()


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
