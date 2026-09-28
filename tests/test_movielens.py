from pathlib import Path
import json

import pandas as pd
import pytest
import yaml

from src.data import load_movielens_ratings
from src.prepare import _sha256, parse_args, prepare
from src.validate_artifacts import validate_artifacts


@pytest.fixture
def ratings_file(tmp_path):
    # Each of 150 items has ten positive users; every user has 139 unseen items.
    rows = []
    for user in range(1, 151):
        for offset in range(10):
            item = (user - 1 + offset) % 150 + 1
            rows.append(f"{user}::{item}::{4 + offset % 2}::978300760\n")
        item = (user - 1 + 20) % 150 + 1
        rows.append(f"{user}::{item}::3::978300760\n")
    path = tmp_path / "ratings.dat"
    path.write_text("".join(rows), encoding="utf-8")
    return path


def movie_args(path, output):
    return parse_args([
        "--dataset", "movielens-1m", "--ratings", str(path), "--output", str(output)
    ])


def test_loader_discards_timestamp_and_preserves_integer_ratings(ratings_file):
    frame = load_movielens_ratings(ratings_file)
    assert frame.columns.tolist() == ["user_id", "anime_id", "rating"]
    assert frame.iloc[0].tolist() == [1, 1, 4]
    assert all(pd.api.types.is_integer_dtype(dtype) for dtype in frame.dtypes)
    assert set(frame.rating) == {3, 4, 5}


@pytest.mark.parametrize("row", [
    "1::2::6::123", "1::2::0::123", "1::2::4.5::123", "0::2::4::123",
    "1::2::4", "1::2::4::123::extra", "1::2::::123", "1::2::4::oops",
])
def test_loader_rejects_invalid_rows(tmp_path, row):
    path = tmp_path / "ratings.dat"
    path.write_text(row + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_movielens_ratings(path)


def test_movie_artifacts_are_reproducible_and_exclude_low_ratings(ratings_file, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = prepare(movie_args(ratings_file, first))
    other = prepare(movie_args(ratings_file, second))
    assert manifest["dataset"] == "movielens_1m"
    assert (manifest["positive_threshold"], manifest["maximum_rating"]) == (4, 5)
    assert manifest["unrated_value"] is None
    assert manifest["ratings_sha256"] == _sha256(ratings_file)
    assert manifest["development_user_count"] == 150
    assert manifest["eligible_user_count"] == 150
    assert manifest["files"] == other["files"]
    raw = load_movielens_ratings(ratings_file)
    observed = set(map(tuple, raw[["user_id", "anime_id"]].to_numpy()))
    train = pd.read_parquet(first / "train_positives.parquet")
    assert train.rating.ge(4).all()
    for name, record in manifest["files"].items():
        a, b = pd.read_parquet(first / name), pd.read_parquet(second / name)
        pd.testing.assert_frame_equal(a, b)
        assert "timestamp" not in a
        assert record["sha256"] == _sha256(first / name)
    for split in ("validation", "test"):
        held = pd.read_parquet(first / f"{split}_positives.parquet")
        candidates = pd.read_parquet(first / f"{split}_candidates.parquet")
        assert held.rating.ge(4).all()
        assert held.user_id.value_counts().eq(1).all()
        assert set(held.user_id) == set(train.user_id)
        assert set(held.anime_id) <= set(train.anime_id)
        assert candidates.groupby("user_id").size().eq(100).all()
        assert candidates.groupby("user_id").label.sum().eq(1).all()
        negative = candidates.loc[candidates.label.eq(0), ["user_id", "item_id"]]
        assert observed.isdisjoint(map(tuple, negative.to_numpy()))
    saved_observed = pd.read_parquet(first / "observed_interactions.parquet")
    assert set(map(tuple, saved_observed.to_numpy())) == observed


def test_prepare_does_not_overwrite_artifacts(ratings_file, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "manifest.json"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare(movie_args(ratings_file, output))
    assert marker.read_text() == "keep"


def test_anime_default_is_backward_compatible(ratings_file, tmp_path):
    raw = load_movielens_ratings(ratings_file)
    raw["rating"] = raw.rating.map({3: -1, 4: 7, 5: 10})
    path = tmp_path / "rating.csv"
    raw.to_csv(path, index=False)
    manifest = prepare(parse_args(["--ratings", str(path), "--output", str(tmp_path / "anime")]))
    assert manifest["dataset"] == "anime_recommendations_database"
    assert manifest["positive_threshold"] == 7
    assert manifest["maximum_rating"] == 10
    assert manifest["unrated_value"] == -1


def test_movie_configs_preserve_locked_hyperparameters():
    root = Path(__file__).resolve().parents[1]
    configs = sorted((root / "configs" / "movielens").glob("*.yaml"))
    assert len(configs) == 13
    allowed = {"artifacts_dir", "output_dir", "positive_threshold", "maximum_rating", "evaluate_test"}
    outputs = set()
    test_states = set()
    for path in configs:
        config = yaml.safe_load(path.read_text())
        model, seed = config["model"], config["seed"]
        source_model = "bpr_ensemble_component" if model == "bpr" else model
        source_name = ("anime_popular_final.yaml" if model == "popular" else
                       f"anime_{source_model}_final_seed{seed}.yaml")
        source = yaml.safe_load((root / "configs" / "final" / source_name).read_text())
        assert {k: v for k, v in config.items() if k not in allowed} == {
            k: v for k, v in source.items() if k not in allowed
        }
        assert type(config["evaluate_test"]) is bool
        test_states.add(config["evaluate_test"])
        assert config["positive_threshold"] == 4 and config["maximum_rating"] == 5
        assert seed in {42, 43, 44}
        assert config["artifacts_dir"] == "../../artifacts/movielens-1m-r4"
        assert config["output_dir"].startswith("../../results/movielens/")
        assert config["output_dir"] not in outputs
        outputs.add(config["output_dir"])
        if model != "popular":
            assert config["device"] == "cuda" and config["gpu_sampling"] is True
    assert test_states == {True}, "the final-run config set must be fully unsealed"


def test_artifact_gate_detects_tampering(ratings_file, tmp_path):
    output = tmp_path / "artifacts"
    prepare(movie_args(ratings_file, output))
    assert validate_artifacts(output, ratings_file)["status"] == "PASS"
    with (output / "train_positives.parquet").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_artifacts(output, ratings_file)


def test_artifact_gate_detects_incorrect_positive_labels(ratings_file, tmp_path):
    output = tmp_path / "artifacts"
    manifest = prepare(movie_args(ratings_file, output))
    path = output / "test_candidates.parquet"
    candidates = pd.read_parquet(path)
    # Swap validation/test positives: still observed, warm and one positive/user,
    # but no longer the declared test holdout. Rehash to exercise semantic checks.
    validation = pd.read_parquet(output / "validation_positives.parquet").set_index("user_id")
    selected = candidates.label.eq(1)
    candidates.loc[selected, "item_id"] = candidates.loc[selected, "user_id"].map(validation.anime_id)
    candidates.to_parquet(path, index=False)
    manifest["files"][path.name]["sha256"] = _sha256(path)
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="positive labels mismatch"):
        validate_artifacts(output, ratings_file)


def test_validation_only_training_never_opens_test_candidates(ratings_file, tmp_path, monkeypatch):
    from src.train import train_from_config

    output = tmp_path / "artifacts"
    prepare(movie_args(ratings_file, output))
    config_path = tmp_path / "popular.yaml"
    config_path.write_text(yaml.safe_dump({
        "model": "popular", "artifacts_dir": "artifacts", "output_dir": "results",
        "seed": 42, "evaluate_test": False,
    }), encoding="utf-8")
    original = pd.read_parquet

    def read_without_test(path, *args, **kwargs):
        assert not Path(path).name.startswith("test_"), "sealed test was read by training"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read_without_test)
    metrics = train_from_config(config_path)
    assert "validation" in metrics and "test" not in metrics
    assert not (tmp_path / "results" / "test_predictions.parquet").exists()
