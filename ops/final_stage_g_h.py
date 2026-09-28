"""Derive the locked final ensembles and aggregate final metrics.

This is an external, hash-recorded reporting helper. It does not train models,
change final configurations, select a blend weight, or overwrite prior output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.ensemble import evaluate_weights


EXPECTED_COMMIT = "b2f4d6b8222f9f5a9afd0633f54a235f50e52c69"
EXPECTED_USERS = 60_384
EXPECTED_CANDIDATES = 6_038_400
LOCKED_WEIGHT = 0.7
SEEDS = (42, 43, 44)
SEEDED_GROUPS = (
    "bpr",
    "bpr_ensemble_component",
    "gmf",
    "mlp",
    "neumf",
    "weighted_neumf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--source-commit", default=EXPECTED_COMMIT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def classify_run(directory_name: str) -> str:
    if directory_name == "anime-popular":
        return "popular"
    if directory_name.startswith("anime-bpr-ensemble-component-seed-"):
        return "bpr_ensemble_component"
    for group in ("bpr", "gmf", "mlp", "neumf", "weighted-neumf"):
        if directory_name.startswith(f"anime-{group}-seed-"):
            return group.replace("-", "_")
    raise AssertionError(f"unexpected final run directory: {directory_name}")


def finite_metric(block: dict[str, Any], key: str) -> float:
    value = float(block[key])
    if not math.isfinite(value):
        raise AssertionError(f"non-finite metric: {key}={value}")
    return value


def validate_and_collect_metrics(
    results_dir: Path, source_commit: str
) -> dict[str, list[dict[str, Any]]]:
    paths = sorted(results_dir.glob("*/metrics.json"))
    if len(paths) != 19:
        raise AssertionError(f"expected 19 metrics files, found {len(paths)}")

    groups: dict[str, list[dict[str, Any]]] = {
        "popular": [],
        **{group: [] for group in SEEDED_GROUPS},
    }
    for path in paths:
        data = read_json(path)
        group = classify_run(path.parent.name)
        if data.get("commit") != source_commit:
            raise AssertionError(f"commit mismatch: {path}")
        if data.get("selected_user_count") != EXPECTED_USERS:
            raise AssertionError(f"user count mismatch: {path}")
        if data.get("config", {}).get("evaluate_test") is not True:
            raise AssertionError(f"test flag mismatch: {path}")
        for split in ("validation", "test"):
            block = data.get(split, {})
            if block.get("user_count") != EXPECTED_USERS:
                raise AssertionError(f"{split} user count mismatch: {path}")
            finite_metric(block, "ndcg@10")
            finite_metric(block, "hit_rate@10")
        if group != "popular":
            if data.get("device") != "cuda" or data.get("gpu_sampling") is not True:
                raise AssertionError(f"CUDA provenance mismatch: {path}")
        data["_metrics_path"] = str(path.relative_to(results_dir.parent.parent))
        data["_group"] = group
        groups[group].append(data)

    if len(groups["popular"]) != 1:
        raise AssertionError("expected one Popular result")
    for group in SEEDED_GROUPS:
        rows = groups[group]
        if len(rows) != 3 or {int(row["seed"]) for row in rows} != set(SEEDS):
            raise AssertionError(f"seed matrix mismatch: {group}")
        rows.sort(key=lambda row: int(row["seed"]))
    return groups


def collect_ensemble_inputs(release_dir: Path) -> list[tuple[int, str, Path, Path]]:
    inputs: list[tuple[int, str, Path, Path]] = []
    for seed in SEEDS:
        for split in ("validation", "test"):
            bpr = release_dir / (
                "results/final/"
                f"anime-bpr-ensemble-component-seed-{seed}/"
                f"{split}_predictions.parquet"
            )
            neural = release_dir / (
                "results/final/"
                f"anime-weighted-neumf-seed-{seed}/"
                f"{split}_predictions.parquet"
            )
            if not bpr.is_file() or not neural.is_file():
                raise FileNotFoundError(f"missing predictions: {bpr} or {neural}")
            if pq.ParquetFile(bpr).metadata.num_rows != EXPECTED_CANDIDATES:
                raise AssertionError(f"BPR candidate count mismatch: {bpr}")
            if pq.ParquetFile(neural).metadata.num_rows != EXPECTED_CANDIDATES:
                raise AssertionError(f"neural candidate count mismatch: {neural}")
            inputs.append((seed, split, bpr, neural))
    return inputs


def derive_ensembles(
    release_dir: Path,
    output_dir: Path,
    inputs: list[tuple[int, str, Path, Path]],
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True)
    ensemble_rows: list[dict[str, Any]] = []
    for seed, split, bpr, neural in inputs:
        rows = evaluate_weights(bpr, neural, [LOCKED_WEIGHT])
        if len(rows) != 1 or float(rows[0]["weight"]) != LOCKED_WEIGHT:
            raise AssertionError("locked ensemble weight changed")
        if int(rows[0]["user_count"]) != EXPECTED_USERS:
            raise AssertionError("ensemble user count mismatch")
        payload = {
            "seed": seed,
            "split": split,
            "source_commit": EXPECTED_COMMIT,
            "weight": LOCKED_WEIGHT,
            "bpr_predictions": str(bpr.relative_to(release_dir)),
            "neural_predictions": str(neural.relative_to(release_dir)),
            "metrics": rows[0],
            "user_count": EXPECTED_USERS,
            "candidate_count": EXPECTED_CANDIDATES,
            "validation_only": split == "validation",
        }
        out = output_dir / f"seed-{seed}-{split}.json"
        write_json_new(out, payload)
        ensemble_rows.append(payload)
    return ensemble_rows


def mean_sd(values: list[float]) -> dict[str, float]:
    if len(values) != 3:
        raise AssertionError(f"expected three seeded values, found {len(values)}")
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
    }


def aggregate_seeded_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_seed = []
    for row in rows:
        per_seed.append(
            {
                "seed": int(row["seed"]),
                "validation_ndcg@10": finite_metric(row["validation"], "ndcg@10"),
                "validation_hit_rate@10": finite_metric(
                    row["validation"], "hit_rate@10"
                ),
                "test_ndcg@10": finite_metric(row["test"], "ndcg@10"),
                "test_hit_rate@10": finite_metric(row["test"], "hit_rate@10"),
                "epochs_completed": int(row["epochs_completed"]),
                "training_runtime_seconds": float(row["training_runtime_seconds"]),
                "total_runtime_seconds": float(row["total_runtime_seconds"]),
                "metrics_path": row["_metrics_path"],
            }
        )
    summary: dict[str, Any] = {"per_seed": per_seed}
    for key in (
        "validation_ndcg@10",
        "validation_hit_rate@10",
        "test_ndcg@10",
        "test_hit_rate@10",
        "training_runtime_seconds",
        "total_runtime_seconds",
    ):
        summary[key] = mean_sd([float(row[key]) for row in per_seed])
    summary["test_minus_validation_ndcg@10"] = (
        summary["test_ndcg@10"]["mean"]
        - summary["validation_ndcg@10"]["mean"]
    )
    summary["test_minus_validation_hit_rate@10"] = (
        summary["test_hit_rate@10"]["mean"]
        - summary["validation_hit_rate@10"]["mean"]
    )
    return summary


def aggregate_ensemble_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_seed = []
    for seed in SEEDS:
        by_split = {
            row["split"]: row
            for row in rows
            if int(row["seed"]) == seed
        }
        if set(by_split) != {"validation", "test"}:
            raise AssertionError(f"ensemble split mismatch: seed {seed}")
        per_seed.append(
            {
                "seed": seed,
                "validation_ndcg@10": finite_metric(
                    by_split["validation"]["metrics"], "ndcg@10"
                ),
                "validation_hit_rate@10": finite_metric(
                    by_split["validation"]["metrics"], "hit_rate@10"
                ),
                "test_ndcg@10": finite_metric(
                    by_split["test"]["metrics"], "ndcg@10"
                ),
                "test_hit_rate@10": finite_metric(
                    by_split["test"]["metrics"], "hit_rate@10"
                ),
            }
        )
    summary: dict[str, Any] = {
        "weight_bpr": LOCKED_WEIGHT,
        "weight_weighted_neumf": 1.0 - LOCKED_WEIGHT,
        "per_seed": per_seed,
    }
    for key in (
        "validation_ndcg@10",
        "validation_hit_rate@10",
        "test_ndcg@10",
        "test_hit_rate@10",
    ):
        summary[key] = mean_sd([float(row[key]) for row in per_seed])
    summary["test_minus_validation_ndcg@10"] = (
        summary["test_ndcg@10"]["mean"]
        - summary["validation_ndcg@10"]["mean"]
    )
    summary["test_minus_validation_hit_rate@10"] = (
        summary["test_hit_rate@10"]["mean"]
        - summary["validation_hit_rate@10"]["mean"]
    )
    return summary


def build_summary(
    release_dir: Path,
    source_commit: str,
    groups: dict[str, list[dict[str, Any]]],
    ensemble_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    popular = groups["popular"][0]
    models: dict[str, Any] = {
        "popular": {
            "seed": int(popular["seed"]),
            "validation_ndcg@10": finite_metric(popular["validation"], "ndcg@10"),
            "validation_hit_rate@10": finite_metric(
                popular["validation"], "hit_rate@10"
            ),
            "test_ndcg@10": finite_metric(popular["test"], "ndcg@10"),
            "test_hit_rate@10": finite_metric(popular["test"], "hit_rate@10"),
            "total_runtime_seconds": float(popular["total_runtime_seconds"]),
            "metrics_path": popular["_metrics_path"],
        }
    }
    for group in SEEDED_GROUPS:
        models[group] = aggregate_seeded_rows(groups[group])
    models["ensemble_w0p7"] = aggregate_ensemble_rows(ensemble_rows)

    ranked_groups = [
        "bpr",
        "gmf",
        "mlp",
        "neumf",
        "weighted_neumf",
        "ensemble_w0p7",
    ]
    ranking = sorted(
        ranked_groups,
        key=lambda group: models[group]["test_ndcg@10"]["mean"],
        reverse=True,
    )
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "release_dir": str(release_dir),
        "reporting_script_sha256": script_sha,
        "protocol": {
            "user_count": EXPECTED_USERS,
            "candidate_count_per_split": EXPECTED_CANDIDATES,
            "candidates_per_user": 100,
            "primary_metric": "test_ndcg@10",
            "secondary_metric": "test_hit_rate@10",
            "seed_summary": "mean and sample standard deviation over seeds 42, 43, 44",
            "ensemble_weight_locked_before_test": LOCKED_WEIGHT,
        },
        "models": models,
        "ranking_by_mean_test_ndcg@10": ranking,
    }


def main() -> None:
    args = parse_args()
    release_dir = args.release_dir.resolve()
    if args.source_commit != EXPECTED_COMMIT:
        raise AssertionError("unexpected formal source commit")
    complete_marker = release_dir / "results/final/CAMPAIGN_COMPLETE"
    if not complete_marker.is_file():
        raise FileNotFoundError(f"missing campaign marker: {complete_marker}")

    results_dir = release_dir / "results/final"
    summary_path = results_dir / "final_summary.json"
    ensemble_dir = results_dir / "ensemble"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite: {summary_path}")
    if ensemble_dir.exists():
        raise FileExistsError(f"refusing to reuse ensemble directory: {ensemble_dir}")
    groups = validate_and_collect_metrics(results_dir, args.source_commit)
    inputs = collect_ensemble_inputs(release_dir)
    if args.preflight_only:
        print("STAGE_G_PREFLIGHT=PASS")
        print("metrics_files=19")
        print("ensemble_input_pairs=6")
        return
    ensemble_rows = derive_ensembles(release_dir, ensemble_dir, inputs)
    summary = build_summary(release_dir, args.source_commit, groups, ensemble_rows)
    write_json_new(summary_path, summary)
    print("STAGE_G_H=PASS")
    print(f"ensemble_files={len(ensemble_rows)}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
