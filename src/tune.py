"""Validation-only, resumable hyperparameter sweeps across available GPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import yaml

from .training import write_json


@dataclass(frozen=True)
class Trial:
    trial_id: str
    model: str
    config: dict[str, object]
    output_dir: Path


@dataclass(frozen=True)
class SweepPlan:
    name: str
    gpus: tuple[str, ...]
    trials: tuple[Trial, ...]
    output_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a validation-only GPU sweep.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_sweep_plan(path: Path) -> SweepPlan:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = str(raw["name"])
    gpus = tuple(str(gpu) for gpu in raw.get("gpus", [0]))
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("gpus must be a non-empty unique list")
    output_root = _resolve(path, str(raw["output_root"]))
    trials: list[Trial] = []
    per_model_counts: dict[str, int] = {}

    for specification in raw["base_configs"]:
        base_path = _resolve(path, str(specification["path"]))
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        if bool(base.get("evaluate_test", False)):
            raise ValueError(f"tuning config may not evaluate test: {base_path}")
        base["evaluate_test"] = False
        base["artifacts_dir"] = str(_resolve(base_path, str(base["artifacts_dir"])))
        model = str(base["model"])
        grid = specification.get("grid", {})
        keys = sorted(grid)
        values = [grid[key] for key in keys]
        combinations = product(*values) if keys else [()]
        count = 0
        for combination in combinations:
            overrides = dict(zip(keys, combination))
            config = {**base, **overrides}
            suffix = "__".join(
                f"{_slug(key)}-{_slug(value)}" for key, value in overrides.items()
            )
            trial_id = _slug(model) + (f"__{suffix}" if suffix else "")
            trial_output = output_root / trial_id
            config["output_dir"] = str(trial_output)
            trials.append(Trial(trial_id, model, config, trial_output))
            count += 1
        per_model_counts[model] = per_model_counts.get(model, 0) + count

    trial_ids = [trial.trial_id for trial in trials]
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("sweep expands to duplicate trial IDs")
    if raw.get("require_equal_trial_count", False):
        if len(set(per_model_counts.values())) != 1:
            raise ValueError(f"unequal tuning budgets: {per_model_counts}")
    return SweepPlan(name, gpus, tuple(trials), output_root)


def run_sweep(plan: SweepPlan) -> dict[str, object]:
    plan.output_root.mkdir(parents=True, exist_ok=True)
    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu in plan.gpus:
        gpu_queue.put(gpu)

    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(plan.gpus)) as executor:
        future_map = {
            executor.submit(_run_trial, trial, gpu_queue): trial
            for trial in plan.trials
        }
        for future in as_completed(future_map):
            records.append(future.result())

    records.sort(
        key=lambda record: (
            record["status"] != "complete",
            -float(record.get("validation_ndcg@10", -1.0)),
            str(record["trial_id"]),
        )
    )
    summary: dict[str, object] = {
        "name": plan.name,
        "trial_count": len(plan.trials),
        "gpus": list(plan.gpus),
        "records": records,
    }
    write_json(plan.output_root / "summary.json", summary)
    if any(record["status"] == "failed" for record in records):
        raise RuntimeError("one or more tuning trials failed; inspect summary.json")
    return summary


def _run_trial(
    trial: Trial, gpu_queue: queue.Queue[str]
) -> dict[str, object]:
    trial.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial.output_dir / "config.yaml"
    metrics_path = trial.output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("config") != trial.config:
            raise ValueError(
                f"refusing to overwrite stale trial with changed config: {trial.trial_id}"
            )
        return _record(trial, metrics, "complete", resumed=True)

    config_path.write_text(
        yaml.safe_dump(trial.config, sort_keys=False), encoding="utf-8"
    )
    gpu = gpu_queue.get()
    try:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        log_path = trial.output_dir / "run.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, "-m", "src.train", "--config", str(config_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        if completed.returncode != 0:
            return {
                "trial_id": trial.trial_id,
                "model": trial.model,
                "status": "failed",
                "return_code": completed.returncode,
                "gpu": gpu,
                "output_dir": str(trial.output_dir),
            }
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {**_record(trial, metrics, "complete", resumed=False), "gpu": gpu}
    finally:
        gpu_queue.put(gpu)


def _record(
    trial: Trial,
    metrics: dict[str, object],
    status: str,
    resumed: bool,
) -> dict[str, object]:
    validation = metrics["validation"]
    return {
        "trial_id": trial.trial_id,
        "model": trial.model,
        "status": status,
        "resumed": resumed,
        "validation_ndcg@10": validation["ndcg@10"],
        "validation_hit_rate@10": validation["hit_rate@10"],
        "epochs_completed": metrics.get("epochs_completed", 0),
        "total_runtime_seconds": metrics["total_runtime_seconds"],
        "output_dir": str(trial.output_dir),
    }


def _resolve(reference_file: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (reference_file.parent / candidate).resolve()


def _slug(value: object) -> str:
    text = str(value).lower().replace(".", "p")
    return re.sub(r"[^a-z0-9_-]+", "-", text).strip("-")


def main() -> None:
    args = parse_args()
    plan = load_sweep_plan(args.plan)
    if args.dry_run:
        print(
            yaml.safe_dump(
                {
                    "name": plan.name,
                    "gpus": list(plan.gpus),
                    "output_root": str(plan.output_root),
                    "trial_ids": [trial.trial_id for trial in plan.trials],
                },
                sort_keys=False,
            )
        )
        return
    summary = run_sweep(plan)
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()

