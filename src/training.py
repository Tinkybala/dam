"""Shared loading, sampling, scoring, and reproducibility helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mapping(path: Path, id_column: str, index_column: str) -> pd.Series:
    frame = pd.read_parquet(path, columns=[id_column, index_column])
    return frame.set_index(id_column)[index_column]


def select_development_users(
    user_ids: np.ndarray, maximum: int | None, seed: int
) -> np.ndarray:
    unique = np.unique(user_ids)
    if maximum is None or maximum >= len(unique):
        return unique
    if maximum < 1:
        raise ValueError("max_users must be positive when supplied")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(unique, size=maximum, replace=False))


def map_interactions(
    frame: pd.DataFrame, user_mapping: pd.Series, item_mapping: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    users = frame["user_id"].map(user_mapping)
    items = frame["anime_id"].map(item_mapping)
    if users.isna().any() or items.isna().any():
        raise ValueError("interactions contain IDs outside the training-visible mappings")
    return users.to_numpy(np.int64), items.to_numpy(np.int64)


def observed_pair_codes(
    observed: pd.DataFrame,
    user_mapping: pd.Series,
    item_mapping: pd.Series,
    item_count: int,
) -> np.ndarray:
    users, items = map_interactions(observed, user_mapping, item_mapping)
    return np.unique(users * np.int64(item_count) + items)


def iter_bpr_batches(
    users: np.ndarray,
    positive_items: np.ndarray,
    observed_codes: np.ndarray,
    item_count: int,
    batch_size: int,
    negatives_per_positive: int,
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield shuffled triples while rejecting every observed user-item pair."""
    if len(users) != len(positive_items):
        raise ValueError("users and positive_items must have the same length")
    if min(item_count, batch_size, negatives_per_positive) < 1:
        raise ValueError("sampling dimensions must be positive")

    for _ in range(negatives_per_positive):
        order = rng.permutation(len(users))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_users = users[indices]
            batch_positives = positive_items[indices]
            negatives = rng.integers(0, item_count, size=len(indices), dtype=np.int64)
            invalid = _codes_present(
                observed_codes, batch_users * np.int64(item_count) + negatives
            )
            while invalid.any():
                negatives[invalid] = rng.integers(
                    0, item_count, size=int(invalid.sum()), dtype=np.int64
                )
                invalid = _codes_present(
                    observed_codes, batch_users * np.int64(item_count) + negatives
                )
            yield batch_users, batch_positives, negatives


def iter_bpr_batches_torch(
    users: np.ndarray,
    positive_items: np.ndarray,
    observed_codes: np.ndarray,
    item_count: int,
    batch_size: int,
    negatives_per_positive: int,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield BPR triples with shuffling and rejection sampling on ``device``.

    This keeps the same uniform-unseen-negative protocol as
    :func:`iter_bpr_batches`, while avoiding a NumPy round-trip for CUDA
    training. Torch's seeded RNG is used for the device-side draws.
    """
    if len(users) != len(positive_items):
        raise ValueError("users and positive_items must have the same length")
    if min(item_count, batch_size, negatives_per_positive) < 1:
        raise ValueError("sampling dimensions must be positive")

    user_tensor = torch.as_tensor(users, dtype=torch.long, device=device)
    positive_tensor = torch.as_tensor(positive_items, dtype=torch.long, device=device)
    observed_tensor = torch.as_tensor(observed_codes, dtype=torch.long, device=device)
    for _ in range(negatives_per_positive):
        order = torch.randperm(len(user_tensor), device=device)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_users = user_tensor.index_select(0, indices)
            batch_positives = positive_tensor.index_select(0, indices)
            negatives = torch.randint(
                item_count, (len(indices),), dtype=torch.long, device=device
            )
            invalid = _torch_codes_present(
                observed_tensor, batch_users * item_count + negatives
            )
            while bool(invalid.any()):
                negatives[invalid] = torch.randint(
                    item_count,
                    (int(invalid.sum().item()),),
                    dtype=torch.long,
                    device=device,
                )
                invalid = _torch_codes_present(
                    observed_tensor, batch_users * item_count + negatives
                )
            yield batch_users, batch_positives, negatives


def iter_pointwise_batches(
    users: np.ndarray,
    positive_items: np.ndarray,
    positive_weights: np.ndarray,
    observed_codes: np.ndarray,
    item_count: int,
    positive_batch_size: int,
    negatives_per_positive: int,
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield positives and rejected unseen negatives under one shared protocol."""
    if not (len(users) == len(positive_items) == len(positive_weights)):
        raise ValueError("pointwise input arrays must have the same length")
    if min(item_count, positive_batch_size, negatives_per_positive) < 1:
        raise ValueError("sampling dimensions must be positive")
    order = rng.permutation(len(users))
    for start in range(0, len(order), positive_batch_size):
        indices = order[start : start + positive_batch_size]
        positive_users = users[indices]
        batch_positive_items = positive_items[indices]
        negative_users = np.repeat(positive_users, negatives_per_positive)
        negatives = rng.integers(
            0, item_count, size=len(negative_users), dtype=np.int64
        )
        invalid = _codes_present(
            observed_codes, negative_users * np.int64(item_count) + negatives
        )
        while invalid.any():
            negatives[invalid] = rng.integers(
                0, item_count, size=int(invalid.sum()), dtype=np.int64
            )
            invalid = _codes_present(
                observed_codes, negative_users * np.int64(item_count) + negatives
            )

        batch_users = np.concatenate([positive_users, negative_users])
        batch_items = np.concatenate([batch_positive_items, negatives])
        labels = np.concatenate(
            [np.ones(len(positive_users)), np.zeros(len(negative_users))]
        ).astype(np.float32)
        weights = np.concatenate(
            [positive_weights[indices], np.ones(len(negative_users), dtype=np.float32)]
        ).astype(np.float32)
        shuffle = rng.permutation(len(batch_users))
        yield (
            batch_users[shuffle],
            batch_items[shuffle],
            labels[shuffle],
            weights[shuffle],
        )


def normalized_confidence_weights(
    ratings: np.ndarray,
    positive_threshold: int,
    maximum_rating: int,
    alpha: float,
) -> np.ndarray:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if maximum_rating <= positive_threshold:
        raise ValueError("maximum_rating must exceed positive_threshold")
    if len(ratings) == 0 or np.any(ratings < positive_threshold):
        raise ValueError("ratings must be non-empty declared positives")
    strength = (ratings.astype(np.float64) - positive_threshold) / (
        maximum_rating - positive_threshold
    )
    weights = 1.0 + alpha * strength
    weights /= weights.mean()
    return weights.astype(np.float32)


def _codes_present(sorted_codes: np.ndarray, queries: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_codes, queries)
    valid_positions = positions < len(sorted_codes)
    present = np.zeros(len(queries), dtype=bool)
    present[valid_positions] = (
        sorted_codes[positions[valid_positions]] == queries[valid_positions]
    )
    return present


def _torch_codes_present(sorted_codes: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    if sorted_codes.numel() == 0:
        return torch.zeros_like(queries, dtype=torch.bool)
    positions = torch.searchsorted(sorted_codes, queries)
    valid_positions = positions < len(sorted_codes)
    safe_positions = positions.clamp(max=max(len(sorted_codes) - 1, 0))
    present = valid_positions & sorted_codes.index_select(0, safe_positions).eq(queries)
    return present


def score_candidates(
    model: torch.nn.Module,
    candidates: pd.DataFrame,
    user_mapping: pd.Series,
    item_mapping: pd.Series,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    users = candidates["user_id"].map(user_mapping)
    items = candidates["item_id"].map(item_mapping)
    if users.isna().any() or items.isna().any():
        raise ValueError("candidates contain IDs outside the training-visible mappings")
    user_values = users.to_numpy(np.int64)
    item_values = items.to_numpy(np.int64)
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            stop = start + batch_size
            batch_users = torch.from_numpy(user_values[start:stop]).to(device)
            batch_items = torch.from_numpy(item_values[start:stop]).to(device)
            scores.append(model(batch_users, batch_items).cpu().numpy())
    output = candidates.copy()
    output["score"] = np.concatenate(scores) if scores else np.array([], dtype=float)
    return output


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
