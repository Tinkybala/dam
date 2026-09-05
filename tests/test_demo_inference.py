"""Unit tests for the offline visual recommender demo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from demo.data_access import BundleError
from demo.inference import InferenceError, Recommender
from src.ensemble import percentile_ranks
from src.models.bpr import BPRMatrixFactorization
from src.models.neural import NeuMF


def _write_fixture(root: Path) -> Path:
    bundle = root / "demo_bundle"
    bundle.mkdir()
    (root / "demo" / "assets").mkdir(parents=True)
    users = pd.DataFrame({"user_id": [10, 20, 30], "user_index": [0, 1, 2]})
    items = pd.DataFrame({"anime_id": [101, 102, 103, 104, 105], "item_index": range(5)})
    observed = pd.DataFrame({"user_id": [10, 10, 20], "anime_id": [101, 102, 103]})
    demo_users = pd.DataFrame(
        {
            "user_id": [10, 20],
            "demo_label": ["Anonymous profile 01", "Anonymous profile 02"],
            "history_count": [2, 1],
        }
    )
    metadata = pd.DataFrame(
        {
            "anime_id": [101, 102, 103, 104, 105],
            "name": ["Alpha", "Beta", "Gamma", None, "Epsilon"],
            "genre": ["Action", "Drama", "Mystery", None, "Sci-Fi"],
            "type": ["TV", "Movie", "TV", "TV", "Movie"],
            "episodes": [12, 1, 24, None, 2],
            "rating": [8.0, 7.5, 8.2, None, 7.9],
            "members": [100, 200, 300, 400, 500],
        }
    )
    users.to_parquet(bundle / "user_mapping.parquet", index=False)
    items.to_parquet(bundle / "item_mapping.parquet", index=False)
    observed.to_parquet(bundle / "observed_by_user.parquet", index=False)
    demo_users.to_parquet(bundle / "demo_users.parquet", index=False)
    metadata.to_parquet(root / "demo" / "assets" / "anime_metadata.parquet", index=False)

    torch.manual_seed(7)
    bpr = BPRMatrixFactorization(3, 5, 4)
    torch.save(bpr.state_dict(), bundle / "bpr_seed42_model.pt")
    neural = NeuMF(3, 5, 3, [4, 2], 0.0)
    torch.save(neural.state_dict(), bundle / "weighted_neumf_seed42_model.pt")

    files = {
        "bpr_seed42_model.pt": bundle / "bpr_seed42_model.pt",
        "weighted_neumf_seed42_model.pt": bundle / "weighted_neumf_seed42_model.pt",
        "user_mapping.parquet": bundle / "user_mapping.parquet",
        "item_mapping.parquet": bundle / "item_mapping.parquet",
        "observed_by_user.parquet": bundle / "observed_by_user.parquet",
        "demo_users.parquet": bundle / "demo_users.parquet",
        "demo/assets/anime_metadata.parquet": root / "demo" / "assets" / "anime_metadata.parquet",
    }
    records = {
        name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in files.items()
    }
    manifest = {
        "schema_version": 1,
        "source_commit": "0" * 40,
        "dataset": "fixture",
        "catalog_type": "warm_items",
        "catalog_count": 5,
        "eligible_user_count": 3,
        "demo_user_count": 2,
        "demo_observed_row_count": 3,
        "ensemble": {"bpr_weight": 0.7, "weighted_neumf_weight": 0.3},
        "models": {
            "bpr_seed42": {"file": "bpr_seed42_model.pt", "embedding_dim": 4, "user_count": 3, "item_count": 5},
            "weighted_neumf_seed42": {
                "file": "weighted_neumf_seed42_model.pt",
                "embedding_dim": 3,
                "hidden_layers": [4, 2],
                "dropout": 0.0,
                "user_count": 3,
                "item_count": 5,
            },
        },
        "files": records,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return bundle


def test_recommendation_filters_seen_and_returns_top_k(tmp_path: Path) -> None:
    engine = Recommender(_write_fixture(tmp_path))
    result = engine.recommend(10, top_k=3)
    assert len(result) == 3
    assert not set(result["item_id"]).intersection({101, 102})
    assert result["ensemble_rank"].tolist() == [1, 2, 3]
    assert result["item_id"].is_unique
    assert result["name"].notna().all()


def test_history_and_stats_are_available_for_demo_user(tmp_path: Path) -> None:
    engine = Recommender(_write_fixture(tmp_path))
    assert engine.catalog_stats["catalog_count"] == 5
    history = engine.history(10, limit=10)
    assert history["anime_id"].tolist() == [101, 102]
    assert engine.poster_path(101).name == "poster-placeholder.svg"


def test_undeclared_poster_index_fails_fast(tmp_path: Path) -> None:
    bundle = _write_fixture(tmp_path)
    (bundle / "poster_index.json").write_text(
        json.dumps({"101": {"file": "posters/101.jpg"}}) + "\n", encoding="utf-8"
    )
    with pytest.raises(BundleError, match="not declared"):
        Recommender(bundle)


def test_unknown_user_and_invalid_top_k_fail(tmp_path: Path) -> None:
    engine = Recommender(_write_fixture(tmp_path))
    with pytest.raises(InferenceError):
        engine.recommend(999)
    with pytest.raises(ValueError):
        engine.recommend(10, top_k=0)


def test_manifest_hash_mismatch_fails_fast(tmp_path: Path) -> None:
    bundle = _write_fixture(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["item_mapping.parquet"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(BundleError, match="SHA-256 mismatch"):
        Recommender(bundle)


def test_manifest_parent_traversal_fails_fast(tmp_path: Path) -> None:
    bundle = _write_fixture(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["../outside.bin"] = {"bytes": 0, "sha256": "0" * 64}
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(BundleError, match="unsafe path"):
        Recommender(bundle)


def test_missing_metadata_fails_fast(tmp_path: Path) -> None:
    bundle = _write_fixture(tmp_path)
    (tmp_path / "demo" / "assets" / "anime_metadata.parquet").unlink()
    with pytest.raises(BundleError, match="manifest file is missing"):
        Recommender(bundle)


def test_percentile_tie_break_is_deterministic() -> None:
    frame = pd.DataFrame(
        {"user_id": [1, 1, 1], "item_id": [20, 10, 30], "label": [0, 0, 0], "score": [1.0, 1.0, 0.5]}
    )
    ranked = percentile_ranks(frame)
    assert ranked["item_id"].tolist() == [10, 20, 30]
    assert np.allclose(ranked["percentile"], [1.0, 0.5, 0.0])


def test_streamlit_entrypoint_imports_when_dependency_is_available() -> None:
    pytest.importorskip("streamlit")
    import demo.app  # noqa: F401
