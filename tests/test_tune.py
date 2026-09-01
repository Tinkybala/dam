from pathlib import Path

import pytest
import yaml

from src.tune import load_sweep_plan


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def test_sweep_expansion_is_deterministic_and_validation_only(tmp_path):
    base = {
        "model": "gmf",
        "artifacts_dir": "artifacts",
        "output_dir": "unused",
        "evaluate_test": False,
    }
    _write_yaml(tmp_path / "base.yaml", base)
    _write_yaml(
        tmp_path / "plan.yaml",
        {
            "name": "test",
            "output_root": "results",
            "gpus": [0, 1],
            "base_configs": [
                {
                    "path": "base.yaml",
                    "grid": {"embedding_dim": [16, 32], "learning_rate": [0.001]},
                }
            ],
        },
    )

    plan = load_sweep_plan(tmp_path / "plan.yaml")

    assert plan.gpus == ("0", "1")
    assert [trial.trial_id for trial in plan.trials] == [
        "gmf__embedding_dim-16__learning_rate-0p001",
        "gmf__embedding_dim-32__learning_rate-0p001",
    ]
    assert all(trial.config["evaluate_test"] is False for trial in plan.trials)
    assert all(Path(trial.config["artifacts_dir"]).is_absolute() for trial in plan.trials)


def test_sweep_rejects_test_evaluation(tmp_path):
    _write_yaml(
        tmp_path / "base.yaml",
        {
            "model": "gmf",
            "artifacts_dir": "artifacts",
            "output_dir": "unused",
            "evaluate_test": True,
        },
    )
    _write_yaml(
        tmp_path / "plan.yaml",
        {
            "name": "test",
            "output_root": "results",
            "base_configs": [{"path": "base.yaml"}],
        },
    )

    with pytest.raises(ValueError, match="may not evaluate test"):
        load_sweep_plan(tmp_path / "plan.yaml")


def test_sweep_enforces_equal_model_budgets(tmp_path):
    for model in ("gmf", "mlp"):
        _write_yaml(
            tmp_path / f"{model}.yaml",
            {"model": model, "artifacts_dir": "artifacts", "output_dir": "unused"},
        )
    _write_yaml(
        tmp_path / "plan.yaml",
        {
            "name": "test",
            "output_root": "results",
            "require_equal_trial_count": True,
            "base_configs": [
                {"path": "gmf.yaml", "grid": {"embedding_dim": [16, 32]}},
                {"path": "mlp.yaml", "grid": {"embedding_dim": [16]}},
            ],
        },
    )

    with pytest.raises(ValueError, match="unequal tuning budgets"):
        load_sweep_plan(tmp_path / "plan.yaml")
