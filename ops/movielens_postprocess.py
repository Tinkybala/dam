"""Verify the locked MovieLens campaign and derive the fixed 0.7/0.3 ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from src.ensemble import blend_predictions, validate_prediction_consistency
from src.evaluate import aggregate_metrics


COMMIT = "61863193e11650cdd579a102f172a51582502552"
SEEDS = (42, 43, 44)
MODELS = ("bpr", "gmf", "neumf", "weighted_neumf")
WEIGHT = 0.7
USERS = 6034
POSITIVES = 562308
CANDIDATES = USERS * 100


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def validate_campaign(root: Path) -> dict[str, dict]:
    output = root / "results" / "movielens"
    require((output / "CAMPAIGN_COMPLETE").is_file(), "campaign has not completed")
    require(not (output / "CAMPAIGN_FAILED").exists(), "campaign failure marker exists")
    runs: dict[str, dict] = {}
    names = ["popular-seed-42"] + [f"{model}-seed-{seed}" for model in MODELS for seed in SEEDS]
    require(len(names) == 13, "run matrix is not 13")
    for name in names:
        path = output / name / "metrics.json"
        require(path.is_file(), f"missing metrics: {name}")
        record = load_json(path)
        config = record["config"]
        require(record["commit"] == COMMIT, f"commit mismatch: {name}")
        require(config["evaluate_test"] is True, f"test still sealed: {name}")
        require(record["seed"] == config["seed"], f"seed mismatch: {name}")
        require(record["model"] == config["model"], f"model mismatch: {name}")
        require(record["selected_user_count"] == USERS and record["training_positive_count"] == POSITIVES,
                f"population mismatch: {name}")
        for split in ("validation", "test"):
            metrics = record.get(split)
            require(isinstance(metrics, dict) and metrics.get("user_count") == USERS, f"missing {split}: {name}")
            require(all(math.isfinite(metrics[k]) and 0 <= metrics[k] <= 1 for k in ("ndcg@10", "hit_rate@10")),
                    f"invalid {split} metrics: {name}")
            predictions = output / name / f"{split}_predictions.parquet"
            require(predictions.is_file(), f"missing {split} predictions: {name}")
        if name != "popular-seed-42":
            require(record.get("device") == "cuda" and record.get("gpu_sampling") is True,
                    f"CUDA gate failed: {name}")
            require(config.get("device") == "cuda" and config.get("gpu_sampling") is True,
                    f"configuration GPU gate failed: {name}")
        log = output / "logs" / f"{name}.log"
        require(log.is_file(), f"missing log: {name}")
        require(not any(word in log.read_text(encoding="utf-8").lower() for word in
                        ("traceback", "out of memory", "non-finite", "timeout")), f"error in log: {name}")
        runs[name] = {"model": record["model"], "seed": record["seed"],
                      "validation": record["validation"], "test": record["test"],
                      "total_runtime_seconds": record["total_runtime_seconds"],
                      "metrics_sha256": sha256(path)}
    monitor = (output / "gpu-monitor.log").read_text(encoding="utf-8")
    require("GPU-" in monitor, "GPU monitor did not capture any process")
    return runs


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["user_id", "item_id", "label"]].sort_values(
        ["user_id", "item_id"], kind="mergesort"
    ).reset_index(drop=True)


def derive_ensemble(root: Path) -> dict[str, dict]:
    output = root / "results" / "movielens"
    artifact = root / "artifacts" / "movielens-1m-r4"
    result = {}
    for seed in SEEDS:
        destination = output / f"ensemble-seed-{seed}"
        require(not destination.exists(), f"ensemble output already exists: {seed}")
        per_split = {}
        for split in ("validation", "test"):
            expected = pd.read_parquet(artifact / f"{split}_candidates.parquet")
            bpr = pd.read_parquet(output / f"bpr-seed-{seed}" / f"{split}_predictions.parquet")
            neural = pd.read_parquet(output / f"weighted_neumf-seed-{seed}" / f"{split}_predictions.parquet")
            require(len(expected) == len(bpr) == len(neural) == CANDIDATES,
                    f"candidate count mismatch: seed {seed} {split}")
            validate_prediction_consistency(bpr, neural)
            require(canonical(bpr).equals(canonical(expected)),
                    f"predictions differ from fixed {split} candidates: seed {seed}")
            blended = blend_predictions(bpr, neural, WEIGHT)
            require(len(blended) == CANDIDATES and not blended.duplicated(["user_id", "item_id"]).any(),
                    f"blended candidate integrity failed: seed {seed} {split}")
            require(np.isfinite(blended["score"].to_numpy()).all(), f"non-finite ensemble score: {seed} {split}")
            per_split[split] = (blended, aggregate_metrics(blended, k=10))
        destination.mkdir()
        record = {"status": "complete", "commit": COMMIT, "model": "ensemble_w0p7",
                  "seed": seed, "weight_bpr": WEIGHT, "weight_weighted_neumf": 1 - WEIGHT,
                  "user_count": USERS, "candidates_per_user": 100}
        for split, (predictions, metrics) in per_split.items():
            predictions.to_parquet(destination / f"{split}_predictions.parquet", index=False)
            record[split] = metrics
        write_new(destination / "metrics.json", record)
        result[f"ensemble-seed-{seed}"] = {"model": "ensemble_w0p7", "seed": seed,
                                           "validation": record["validation"], "test": record["test"],
                                           "metrics_sha256": sha256(destination / "metrics.json")}
    return result


def summarize(runs: dict[str, dict]) -> dict:
    result = {"status": "complete", "source_commit": COMMIT, "dataset": "movielens_1m",
              "protocol": {"users": USERS, "warm_items": 3125, "candidates_per_user": 100,
                           "positive_threshold": 4, "seeds": list(SEEDS),
                           "seed_dispersion": "sample standard deviation over 3 training seeds"},
              "systems": {}, "runs": runs}
    for model in ("popular", *MODELS, "ensemble_w0p7"):
        selected = [record for record in runs.values() if record["model"] == model]
        expected = [42] if model == "popular" else list(SEEDS)
        require(sorted(record["seed"] for record in selected) == expected, f"missing seeds: {model}")
        entry = {"seeds": expected}
        for split in ("validation", "test"):
            for metric in ("ndcg@10", "hit_rate@10"):
                values = [record[split][metric] for record in selected]
                entry[f"{split}_{metric}"] = {"mean": statistics.mean(values),
                                                 "sample_sd": statistics.stdev(values) if len(values) > 1 else None}
        if all("total_runtime_seconds" in record for record in selected):
            values = [record["total_runtime_seconds"] for record in selected]
            entry["runtime_seconds"] = {"mean": statistics.mean(values),
                                        "sample_sd": statistics.stdev(values) if len(values) > 1 else None}
        result["systems"][model] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    runs = validate_campaign(root)
    runs.update(derive_ensemble(root))
    output = root / "results" / "movielens" / "final_summary.json"
    write_new(output, summarize(runs))
    print(json.dumps({"status": "complete", "run_count": 13, "ensembles": 3,
                      "summary_sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
