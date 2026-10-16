"""Train a model, select on validation, then evaluate the selected checkpoint."""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd
import torch
import yaml
from evaluate import aggregate_metrics
from models import BPRMatrixFactorization, GMF, MLP, MostPopular, NeuMF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a recommender from a YAML config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True,
                        help="Model name listed in the selected dataset config.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Two CPU epochs on 100 users, validation only.")
    return parser.parse_args()


def load_config(config_path: Path, model_name: str, seed: int = 42,
                device: str | None = None, quick: bool = False) -> dict:
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Shared dataset settings plus the selected model's report parameters.
    config = {**settings["defaults"], **settings["models"][model_name]}
    config["seed"] = seed
    config["output_dir"] = config["output_dir"].format(model=model_name, seed=seed)
    if device is not None:
        config["device"] = device
    if quick:
        config.update(device="cpu", epochs=2, max_users=100, evaluate_test=False)
        config["output_dir"] = f"../results/quickstart/{config_path.stem}/{model_name}-seed-{seed}"
    return config


def train_from_config(config_path: Path, model_name: str, seed: int = 42,
                      device: str | None = None, quick: bool = False) -> dict[str, object]:
    config_path = config_path.resolve()
    config = load_config(config_path, model_name, seed, device, quick)
    supported_models = {"popular", "bpr", "gmf", "mlp", "neumf", "weighted_neumf"}
    if config["model"] not in supported_models:
        raise ValueError(f"model must be one of {sorted(supported_models)}")

    artifacts = _resolve(config_path, config["artifacts_dir"])
    output = _resolve(config_path, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    set_random_seed(seed)

    train = pd.read_parquet(artifacts / "train_positives.parquet")
    selected_users = select_development_users(train["user_id"].to_numpy(), config.get("max_users"), seed)
    train = train.loc[train["user_id"].isin(selected_users)].reset_index(drop=True)
    validation_candidates = _load_candidates(
        artifacts / "validation_candidates.parquet", selected_users
    )
    # Tuning uses validation only; test scoring is enabled explicitly for final runs.
    test_candidates = (
        _load_candidates(artifacts / "test_candidates.parquet", selected_users)
        if bool(config.get("evaluate_test", False))
        else None
    )

    started = time.perf_counter()
    if config["model"] == "popular":
        result = _run_popular(train, validation_candidates, test_candidates, output)
    else:
        result = _run_trained(config, artifacts, train, validation_candidates,
                              test_candidates, output, seed)
    result.update(
        {
            "model": config["model"],
            "seed": seed,
            "selected_user_count": int(len(selected_users)),
            "training_positive_count": int(len(train)),
            "total_runtime_seconds": time.perf_counter() - started,
            "config": config,
        }
    )
    write_json(output / "metrics.json", result)
    return result


def _run_popular(
    train: pd.DataFrame,
    validation_candidates: pd.DataFrame,
    test_candidates: pd.DataFrame | None,
    output: Path,
) -> dict[str, object]:
    model = MostPopular().fit(train)
    validation_predictions = validation_candidates.copy()
    validation_predictions["score"] = model.predict(validation_predictions["item_id"])
    validation_predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    result = {"validation": aggregate_metrics(validation_predictions, k=10)}
    if test_candidates is not None:
        test_predictions = test_candidates.copy()
        test_predictions["score"] = model.predict(test_predictions["item_id"])
        test_predictions.to_parquet(output / "test_predictions.parquet", index=False)
        result["test"] = aggregate_metrics(test_predictions, k=10)
    return result


def _load_candidates(path: Path, selected_users: np.ndarray) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return frame.loc[frame["user_id"].isin(selected_users)].reset_index(drop=True)


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()



# Shared training loop.
def _run_trained(config, artifacts, train, validation_candidates, test_candidates, output, seed):
    """Train BPR or a neural model with the same validation/early-stopping loop."""
    user_mapping = load_mapping(artifacts / "user_mapping.parquet", "user_id", "user_index")
    item_mapping = load_mapping(artifacts / "item_mapping.parquet", "anime_id", "item_index")
    observed = pd.read_parquet(artifacts / "observed_interactions.parquet")
    observed = observed.loc[observed["user_id"].isin(train["user_id"].unique())]
    train_users, train_items = map_interactions(train, user_mapping, item_mapping)
    item_count = len(item_mapping)
    codes = observed_pair_codes(observed, user_mapping, item_mapping, item_count)

    model_name = config["model"]
    pairwise = model_name == "bpr"
    requested_device = config.get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    common = dict(user_count=len(user_mapping), item_count=item_count,
                  embedding_dim=int(config.get("embedding_dim", 32)))
    if pairwise:
        model = BPRMatrixFactorization(**common)
    elif model_name == "gmf":
        model = GMF(**common)
    else:
        architecture = MLP if model_name == "mlp" else NeuMF
        model = architecture(**common,
                             hidden_layers=[int(w) for w in config.get("hidden_layers", [64, 32, 16, 8])],
                             dropout=float(config.get("dropout", 0.0)))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(config.get("learning_rate", 1e-3)),
                                 weight_decay=float(config.get("l2", 0.0)))
    rng = np.random.default_rng(seed)
    gpu_sampling = device.type == "cuda" and bool(config.get("gpu_sampling", True))
    negatives = int(config.get("negatives_per_positive", 1))
    eval_batch_size = int(config.get("eval_batch_size", 65536))
    alpha = float(config.get("confidence_alpha", 0.0))
    if not pairwise:
        if model_name != "weighted_neumf" and alpha != 0.0:
            raise ValueError("confidence_alpha is only valid for weighted_neumf")
        positive_weights = normalized_confidence_weights(
            train["rating"].to_numpy(), int(config.get("positive_threshold", 7)),
            int(config.get("maximum_rating", 10)), alpha)

    # Keep CPU and GPU sampling unchanged so the report's seeded runs remain reproducible.
    if pairwise:
        sampler = iter_bpr_batches_torch if gpu_sampling else iter_bpr_batches
        batch_args = (train_users, train_items, codes, item_count,
                      int(config.get("batch_size", 4096)), negatives)
    else:
        sampler = iter_pointwise_batches_torch if gpu_sampling else iter_pointwise_batches
        batch_args = (train_users, train_items, positive_weights, codes, item_count,
                      int(config.get("positive_batch_size", 2048)), negatives)

    best_ndcg, stale_epochs = -1.0, 0
    best_state = None
    history = []
    training_started = time.perf_counter()
    for epoch in range(1, int(config.get("epochs", 5)) + 1):
        model.train()
        loss_sum, example_count = 0.0, 0
        epoch_started = time.perf_counter()
        for batch in sampler(*batch_args, device if gpu_sampling else rng):
            tensors = batch if gpu_sampling else tuple(torch.from_numpy(x).to(device) for x in batch)
            optimizer.zero_grad(set_to_none=True)
            if pairwise:
                loss = model.pairwise_loss(*tensors)
            else:
                users, items, labels, weights = tensors
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(users, items), labels, reduction="none")
                # Weighted NeuMF changes the loss weights, not the architecture.
                loss = (losses * weights).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch[0])
            example_count += len(batch[0])

        predictions = score_candidates(model, validation_candidates, user_mapping,
                                       item_mapping, device, eval_batch_size)
        validation = aggregate_metrics(predictions, k=10)
        history.append({"epoch": epoch, "loss": loss_sum / example_count,
                        "validation_ndcg@10": validation["ndcg@10"],
                        "validation_hit_rate@10": validation["hit_rate@10"],
                        "seconds": time.perf_counter() - epoch_started})
        # Select the checkpoint on validation NDCG only; test is scored afterwards.
        if validation["ndcg@10"] > best_ndcg:
            best_ndcg = validation["ndcg@10"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config.get("early_stopping_patience", 3)):
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    predictions = score_candidates(model, validation_candidates, user_mapping,
                                   item_mapping, device, eval_batch_size)
    predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    torch.save(best_state, output / "model.pt")
    result = {"device": str(device), "epochs_completed": len(history),
              "training_runtime_seconds": time.perf_counter() - training_started,
              "gpu_sampling": gpu_sampling, "history": history,
              "validation": aggregate_metrics(predictions, k=10)}
    if not pairwise:
        result.update(confidence_alpha=alpha, mean_positive_weight=float(positive_weights.mean()))
    if test_candidates is not None:
        predictions = score_candidates(model, test_candidates, user_mapping,
                                       item_mapping, device, eval_batch_size)
        predictions.to_parquet(output / "test_predictions.parquet", index=False)
        result["test"] = aggregate_metrics(predictions, k=10)
    return result


# Sampling and scoring.
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
    # A pair has one integer code; sorted unique codes make membership checks cheap.
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
            # Redraw observed items until every negative is unseen by this user.
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

        # Positive examples use confidence weights; sampled negatives keep weight 1.
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


def iter_pointwise_batches_torch(
    users: np.ndarray,
    positive_items: np.ndarray,
    positive_weights: np.ndarray,
    observed_codes: np.ndarray,
    item_count: int,
    positive_batch_size: int,
    negatives_per_positive: int,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield pointwise batches with sampling and shuffling on ``device``."""
    if not (len(users) == len(positive_items) == len(positive_weights)):
        raise ValueError("pointwise input arrays must have the same length")
    if min(item_count, positive_batch_size, negatives_per_positive) < 1:
        raise ValueError("sampling dimensions must be positive")

    user_tensor = torch.as_tensor(users, dtype=torch.long, device=device)
    positive_tensor = torch.as_tensor(positive_items, dtype=torch.long, device=device)
    weight_tensor = torch.as_tensor(positive_weights, dtype=torch.float32, device=device)
    observed_tensor = torch.as_tensor(observed_codes, dtype=torch.long, device=device)
    order = torch.randperm(len(user_tensor), device=device)
    for start in range(0, len(order), positive_batch_size):
        indices = order[start : start + positive_batch_size]
        positive_users = user_tensor.index_select(0, indices)
        batch_positive_items = positive_tensor.index_select(0, indices)
        negative_users = positive_users.repeat_interleave(negatives_per_positive)
        negatives = torch.randint(
            item_count, (len(negative_users),), dtype=torch.long, device=device
        )
        invalid = _torch_codes_present(
            observed_tensor, negative_users * item_count + negatives
        )
        while bool(invalid.any()):
            negatives[invalid] = torch.randint(
                item_count,
                (int(invalid.sum().item()),),
                dtype=torch.long,
                device=device,
            )
            invalid = _torch_codes_present(
                observed_tensor, negative_users * item_count + negatives
            )

        batch_users = torch.cat([positive_users, negative_users])
        batch_items = torch.cat([batch_positive_items, negatives])
        labels = torch.cat(
            [
                torch.ones(len(positive_users), dtype=torch.float32, device=device),
                torch.zeros(len(negative_users), dtype=torch.float32, device=device),
            ]
        )
        weights = torch.cat(
            [
                weight_tensor.index_select(0, indices),
                torch.ones(len(negative_users), dtype=torch.float32, device=device),
            ]
        )
        shuffle = torch.randperm(len(batch_users), device=device)
        yield (
            batch_users.index_select(0, shuffle),
            batch_items.index_select(0, shuffle),
            labels.index_select(0, shuffle),
            weights.index_select(0, shuffle),
        )


def normalized_confidence_weights(
    ratings: np.ndarray,
    positive_threshold: int,
    maximum_rating: int,
    alpha: float,
) -> np.ndarray:
    """Give higher ratings more weight, keeping the mean positive weight at one.

    For Anime the rating strength is (rating - 7) / 3; MovieLens uses
    (rating - 4) / 1. With alpha=0 this is ordinary unweighted NeuMF.
    """
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
    # Normalization changes relative confidence without increasing mean loss scale.
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
    safe_positions = positions.clamp(max=len(sorted_codes) - 1)
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


def main() -> None:
    args = parse_args()
    result = train_from_config(args.config, args.model, args.seed, args.device, args.quick)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
