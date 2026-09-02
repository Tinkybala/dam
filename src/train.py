"""Train and evaluate the initial Most Popular and BPR-MF baselines."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .evaluate import aggregate_metrics
from .models import BPRMatrixFactorization, GMF, MLP, MostPopular, NeuMF
from .training import (
    iter_bpr_batches,
    iter_bpr_batches_torch,
    iter_pointwise_batches,
    load_mapping,
    map_interactions,
    observed_pair_codes,
    normalized_confidence_weights,
    score_candidates,
    select_development_users,
    set_random_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a recommender from a YAML config.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def train_from_config(config_path: Path) -> dict[str, object]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    supported_models = {"popular", "bpr", "gmf", "mlp", "neumf", "weighted_neumf"}
    if config["model"] not in supported_models:
        raise ValueError(f"model must be one of {sorted(supported_models)}")

    artifacts = _resolve(config_path, config["artifacts_dir"])
    output = _resolve(config_path, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    set_random_seed(seed)

    train = pd.read_parquet(artifacts / "train_positives.parquet")
    selected_users = _selected_users(config, artifacts, train, seed)
    train = train.loc[train["user_id"].isin(selected_users)].reset_index(drop=True)
    validation_candidates = _load_candidates(
        artifacts / "validation_candidates.parquet", selected_users
    )
    test_candidates = (
        _load_candidates(artifacts / "test_candidates.parquet", selected_users)
        if bool(config.get("evaluate_test", False))
        else None
    )

    started = time.perf_counter()
    if config["model"] == "popular":
        result = _run_popular(train, validation_candidates, test_candidates, output)
    elif config["model"] == "bpr":
        result = _run_bpr(
            config,
            artifacts,
            train,
            validation_candidates,
            test_candidates,
            output,
            seed,
        )
    else:
        result = _run_pointwise(
            config,
            artifacts,
            train,
            validation_candidates,
            test_candidates,
            output,
            seed,
        )
    result.update(
        {
            "model": config["model"],
            "seed": seed,
            "selected_user_count": int(len(selected_users)),
            "training_positive_count": int(len(train)),
            "total_runtime_seconds": time.perf_counter() - started,
            "commit": _git_commit(config_path.parent),
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


def _run_bpr(
    config: dict[str, object],
    artifacts: Path,
    train: pd.DataFrame,
    validation_candidates: pd.DataFrame,
    test_candidates: pd.DataFrame | None,
    output: Path,
    seed: int,
) -> dict[str, object]:
    user_mapping = load_mapping(artifacts / "user_mapping.parquet", "user_id", "user_index")
    item_mapping = load_mapping(artifacts / "item_mapping.parquet", "anime_id", "item_index")
    observed = pd.read_parquet(artifacts / "observed_interactions.parquet")
    observed = observed.loc[observed["user_id"].isin(train["user_id"].unique())]
    train_users, train_items = map_interactions(train, user_mapping, item_mapping)
    item_count = len(item_mapping)
    codes = observed_pair_codes(observed, user_mapping, item_mapping, item_count)

    requested_device = str(config.get("device", "auto"))
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    model = BPRMatrixFactorization(
        user_count=len(user_mapping),
        item_count=item_count,
        embedding_dim=int(config.get("embedding_dim", 32)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("l2", 0.0)),
    )
    rng = np.random.default_rng(seed)
    epochs = int(config.get("epochs", 5))
    patience = int(config.get("early_stopping_patience", 3))
    batch_size = int(config.get("batch_size", 4096))
    eval_batch_size = int(config.get("eval_batch_size", 65536))
    negatives = int(config.get("negatives_per_positive", 1))
    best_ndcg = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    stale_epochs = 0

    training_started = time.perf_counter()
    gpu_sampling = device.type == "cuda" and bool(config.get("gpu_sampling", True))
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        example_count = 0
        epoch_started = time.perf_counter()
        if gpu_sampling:
            batches = iter_bpr_batches_torch(
                train_users,
                train_items,
                codes,
                item_count,
                batch_size,
                negatives,
                device,
            )
        else:
            batches = iter_bpr_batches(
                train_users,
                train_items,
                codes,
                item_count,
                batch_size,
                negatives,
                rng,
            )
        for batch_users, batch_positives, batch_negatives in batches:
            if gpu_sampling:
                users_tensor = batch_users
                positives_tensor = batch_positives
                negatives_tensor = batch_negatives
            else:
                users_tensor = torch.from_numpy(batch_users).to(device)
                positives_tensor = torch.from_numpy(batch_positives).to(device)
                negatives_tensor = torch.from_numpy(batch_negatives).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.pairwise_loss(users_tensor, positives_tensor, negatives_tensor)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite BPR loss")
            loss.backward()
            optimizer.step()
            batch_count = len(batch_users)
            loss_sum += float(loss.detach()) * batch_count
            example_count += batch_count

        validation_predictions = score_candidates(
            model,
            validation_candidates,
            user_mapping,
            item_mapping,
            device,
            eval_batch_size,
        )
        validation = aggregate_metrics(validation_predictions, k=10)
        epoch_record = {
            "epoch": epoch,
            "loss": loss_sum / example_count,
            "validation_ndcg@10": validation["ndcg@10"],
            "validation_hit_rate@10": validation["hit_rate@10"],
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_record)
        if validation["ndcg@10"] > best_ndcg:
            best_ndcg = validation["ndcg@10"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("BPR training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation_predictions = score_candidates(
        model, validation_candidates, user_mapping, item_mapping, device, eval_batch_size
    )
    validation_predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    torch.save(best_state, output / "model.pt")
    result = {
        "device": str(device),
        "epochs_completed": len(history),
        "training_runtime_seconds": time.perf_counter() - training_started,
        "gpu_sampling": gpu_sampling,
        "history": history,
        "validation": aggregate_metrics(validation_predictions, k=10),
    }
    if test_candidates is not None:
        test_predictions = score_candidates(
            model, test_candidates, user_mapping, item_mapping, device, eval_batch_size
        )
        test_predictions.to_parquet(output / "test_predictions.parquet", index=False)
        result["test"] = aggregate_metrics(test_predictions, k=10)
    return result


def _run_pointwise(
    config: dict[str, object],
    artifacts: Path,
    train: pd.DataFrame,
    validation_candidates: pd.DataFrame,
    test_candidates: pd.DataFrame | None,
    output: Path,
    seed: int,
) -> dict[str, object]:
    user_mapping = load_mapping(artifacts / "user_mapping.parquet", "user_id", "user_index")
    item_mapping = load_mapping(artifacts / "item_mapping.parquet", "anime_id", "item_index")
    observed = pd.read_parquet(artifacts / "observed_interactions.parquet")
    observed = observed.loc[observed["user_id"].isin(train["user_id"].unique())]
    train_users, train_items = map_interactions(train, user_mapping, item_mapping)
    item_count = len(item_mapping)
    codes = observed_pair_codes(observed, user_mapping, item_mapping, item_count)
    model_name = str(config["model"])
    alpha = float(config.get("confidence_alpha", 0.0))
    if model_name != "weighted_neumf" and alpha != 0.0:
        raise ValueError("confidence_alpha is only valid for weighted_neumf")
    positive_weights = normalized_confidence_weights(
        train["rating"].to_numpy(),
        int(config.get("positive_threshold", 7)),
        int(config.get("maximum_rating", 10)),
        alpha,
    )

    requested_device = str(config.get("device", "auto"))
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    common = {
        "user_count": len(user_mapping),
        "item_count": item_count,
        "embedding_dim": int(config.get("embedding_dim", 32)),
    }
    hidden_layers = [int(width) for width in config.get("hidden_layers", [64, 32, 16, 8])]
    dropout = float(config.get("dropout", 0.0))
    if model_name == "gmf":
        model = GMF(**common)
    elif model_name == "mlp":
        model = MLP(**common, hidden_layers=hidden_layers, dropout=dropout)
    else:
        model = NeuMF(**common, hidden_layers=hidden_layers, dropout=dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("l2", 0.0)),
    )
    rng = np.random.default_rng(seed)
    epochs = int(config.get("epochs", 5))
    patience = int(config.get("early_stopping_patience", 3))
    positive_batch_size = int(config.get("positive_batch_size", 2048))
    eval_batch_size = int(config.get("eval_batch_size", 65536))
    negative_count = int(config.get("negatives_per_positive", 1))
    best_ndcg = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    stale_epochs = 0
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        example_count = 0
        epoch_started = time.perf_counter()
        for batch_users, batch_items, labels, weights in iter_pointwise_batches(
            train_users,
            train_items,
            positive_weights,
            codes,
            item_count,
            positive_batch_size,
            negative_count,
            rng,
        ):
            users_tensor = torch.from_numpy(batch_users).to(device)
            items_tensor = torch.from_numpy(batch_items).to(device)
            labels_tensor = torch.from_numpy(labels).to(device)
            weights_tensor = torch.from_numpy(weights).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(users_tensor, items_tensor)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels_tensor, reduction="none"
            )
            loss = (losses * weights_tensor).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite pointwise loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_users)
            example_count += len(batch_users)

        validation_predictions = score_candidates(
            model, validation_candidates, user_mapping, item_mapping, device, eval_batch_size
        )
        validation = aggregate_metrics(validation_predictions, k=10)
        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / example_count,
                "validation_ndcg@10": validation["ndcg@10"],
                "validation_hit_rate@10": validation["hit_rate@10"],
                "seconds": time.perf_counter() - epoch_started,
            }
        )
        if validation["ndcg@10"] > best_ndcg:
            best_ndcg = validation["ndcg@10"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("pointwise training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation_predictions = score_candidates(
        model, validation_candidates, user_mapping, item_mapping, device, eval_batch_size
    )
    validation_predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    torch.save(best_state, output / "model.pt")
    result = {
        "device": str(device),
        "confidence_alpha": alpha,
        "mean_positive_weight": float(positive_weights.mean()),
        "epochs_completed": len(history),
        "training_runtime_seconds": time.perf_counter() - training_started,
        "history": history,
        "validation": aggregate_metrics(validation_predictions, k=10),
    }
    if test_candidates is not None:
        test_predictions = score_candidates(
            model, test_candidates, user_mapping, item_mapping, device, eval_batch_size
        )
        test_predictions.to_parquet(output / "test_predictions.parquet", index=False)
        result["test"] = aggregate_metrics(test_predictions, k=10)
    return result


def _load_candidates(path: Path, selected_users: np.ndarray) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return frame.loc[frame["user_id"].isin(selected_users)].reset_index(drop=True)


def _selected_users(
    config: dict[str, object], artifacts: Path, train: pd.DataFrame, seed: int
) -> np.ndarray:
    development_file = config.get("development_users_file")
    if development_file is None:
        return select_development_users(
            train["user_id"].to_numpy(), config.get("max_users"), seed
        )
    if config.get("max_users") is not None:
        raise ValueError("use development_users_file or max_users, not both")
    path = Path(str(development_file))
    if not path.is_absolute():
        path = artifacts / path
    selected = pd.read_parquet(path, columns=["user_id"])["user_id"].to_numpy()
    if len(selected) == 0 or len(np.unique(selected)) != len(selected):
        raise ValueError("development user artifact must be non-empty and unique")
    known_users = np.unique(train["user_id"].to_numpy())
    if not np.isin(selected, known_users).all():
        raise ValueError("development user artifact contains an ineligible user")
    return np.sort(selected)


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _git_commit(workdir: Path) -> str | None:
    if "SOURCE_COMMIT" in os.environ:
        return os.environ["SOURCE_COMMIT"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    result = train_from_config(parse_args().config)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
