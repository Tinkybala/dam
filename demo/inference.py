"""Offline CPU inference for the frozen Anime recommender demo."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.ensemble import percentile_ranks
from src.models.bpr import BPRMatrixFactorization
from src.models.neural import NeuMF

from .data_access import BundleError, load_bundle


class InferenceError(RuntimeError):
    """Raised for invalid demo requests."""


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.1 compatibility
        return torch.load(path, map_location="cpu")


def _fill_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("name", "genre", "type", "episodes"):
        if column in result:
            result[column] = result[column].astype(object).where(
                result[column].notna(), "Unknown"
            )
    if "name" in result:
        result["name"] = result["name"].map(
            lambda value: html.unescape(str(value)) if value != "Unknown" else value
        )
    for column in ("rating", "members"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


class Recommender:
    """Load frozen models once and score the complete warm catalog on CPU."""

    def __init__(
        self,
        bundle_dir: str | Path,
        metadata_path: str | Path | None = None,
        score_batch_size: int = 4096,
    ) -> None:
        if score_batch_size < 1:
            raise ValueError("score_batch_size must be positive")
        self.bundle_dir = Path(bundle_dir).expanduser().resolve()
        self.data = load_bundle(self.bundle_dir, metadata_path)
        manifest = self.data["manifest"]
        self.user_mapping = self.data["user_mapping"]
        self.item_mapping = self.data["item_mapping"]
        self.observed = self.data["observed"]
        self.metadata = _fill_metadata(self.data["metadata"])
        self.demo_user_table = self.data["demo_users"]
        self.poster_index = self.data["poster_index"]
        self.score_batch_size = score_batch_size
        self.user_to_index = dict(
            zip(self.user_mapping["user_id"].astype(int), self.user_mapping["user_index"].astype(int))
        )
        self.item_ids = self.item_mapping["anime_id"].astype(int).to_numpy()

        models = manifest["models"]
        bpr_spec = models["bpr_seed42"]
        neural_spec = models["weighted_neumf_seed42"]
        user_count = len(self.user_mapping)
        item_count = len(self.item_mapping)
        if (bpr_spec["user_count"], bpr_spec["item_count"]) != (user_count, item_count):
            raise BundleError("BPR manifest dimensions disagree with mappings")
        if (neural_spec["user_count"], neural_spec["item_count"]) != (user_count, item_count):
            raise BundleError("NeuMF manifest dimensions disagree with mappings")

        self.bpr = BPRMatrixFactorization(user_count, item_count, int(bpr_spec["embedding_dim"]))
        self.bpr.load_state_dict(_load_state(self.bundle_dir / bpr_spec["file"]))
        self.neural = NeuMF(
            user_count,
            item_count,
            int(neural_spec["embedding_dim"]),
            [int(width) for width in neural_spec["hidden_layers"]],
            float(neural_spec.get("dropout", 0.0)),
        )
        self.neural.load_state_dict(_load_state(self.bundle_dir / neural_spec["file"]))
        self.bpr.eval()
        self.neural.eval()
        self.bpr_weight = float(manifest["ensemble"]["bpr_weight"])
        self.neural_weight = float(manifest["ensemble"]["weighted_neumf_weight"])
        if not np.isclose(self.bpr_weight + self.neural_weight, 1.0):
            raise BundleError("ensemble weights must sum to one")

    @property
    def demo_users(self) -> pd.DataFrame:
        """Return the anonymous profiles available to the UI."""

        return self.demo_user_table.copy()

    @property
    def catalog_stats(self) -> dict[str, Any]:
        manifest = self.data["manifest"]
        return {
            "eligible_user_count": int(manifest["eligible_user_count"]),
            "catalog_count": int(manifest["catalog_count"]),
            "demo_user_count": int(manifest["demo_user_count"]),
            "source_commit": str(manifest["source_commit"]),
        }

    def _score(self, model: torch.nn.Module, user_index: int, item_indices: np.ndarray) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(item_indices), self.score_batch_size):
                batch = item_indices[start : start + self.score_batch_size]
                users = torch.full((len(batch),), user_index, dtype=torch.long)
                items = torch.as_tensor(batch, dtype=torch.long)
                chunks.append(model(users, items).detach().cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=float)

    def history(self, user_id: int, limit: int = 30) -> pd.DataFrame:
        """Return a deterministic, display-only history for one demo user."""

        self._require_user(user_id)
        if limit < 1:
            raise ValueError("limit must be positive")
        history = self.observed.loc[self.observed["user_id"] == int(user_id)].copy()
        history = history.merge(self.metadata, on="anime_id", how="left", validate="many_to_one")
        history = _fill_metadata(history)
        columns = [column for column in ["anime_id", "name", "genre", "type", "episodes", "rating"] if column in history]
        return history.sort_values(["name", "anime_id"], kind="mergesort")[columns].head(limit).reset_index(drop=True)

    def poster_path(self, anime_id: int) -> Path:
        """Return a cached poster or the committed offline placeholder."""

        record = self.poster_index.get(str(int(anime_id)))
        if record is not None:
            candidate = (self.bundle_dir / record["file"]).resolve()
            if candidate.is_file() and self.bundle_dir in candidate.parents:
                return candidate
        return Path(__file__).resolve().parent / "assets" / "poster-placeholder.svg"

    def recommend(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """Return unseen Top-K recommendations with both model explanations."""

        user_id = int(user_id)
        self._require_user(user_id)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        user_index = self.user_to_index[user_id]
        seen = set(self.observed.loc[self.observed["user_id"] == user_id, "anime_id"].astype(int))
        candidate = self.item_mapping.loc[~self.item_mapping["anime_id"].isin(seen)].copy()
        item_indices = candidate["item_index"].astype(int).to_numpy()
        item_ids = candidate["anime_id"].astype(int).to_numpy()
        bpr_scores = self._score(self.bpr, user_index, item_indices)
        neural_scores = self._score(self.neural, user_index, item_indices)
        base = pd.DataFrame(
            {
                "user_id": user_id,
                "item_id": item_ids,
                "label": 0,
                "bpr_score": bpr_scores,
                "neural_score": neural_scores,
            }
        )
        bpr_input = base.rename(columns={"bpr_score": "score"})[
            ["user_id", "item_id", "label", "score"]
        ]
        neural_input = base.rename(columns={"neural_score": "score"})[
            ["user_id", "item_id", "label", "score"]
        ]
        bpr_rank = percentile_ranks(bpr_input).rename(columns={"percentile": "bpr_percentile"})
        neural_rank = percentile_ranks(neural_input).rename(
            columns={"percentile": "neural_percentile"}
        )
        result = base.merge(bpr_rank, on=["user_id", "item_id"], validate="one_to_one").merge(
            neural_rank, on=["user_id", "item_id"], validate="one_to_one"
        )
        result["ensemble_score"] = (
            self.bpr_weight * result["bpr_percentile"]
            + self.neural_weight * result["neural_percentile"]
        )
        result = result.merge(
            self.metadata, left_on="item_id", right_on="anime_id", how="left", validate="many_to_one"
        ).drop(columns=["anime_id"], errors="ignore")
        result = _fill_metadata(result)
        result = result.sort_values(
            ["ensemble_score", "item_id"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        result["ensemble_rank"] = np.arange(1, len(result) + 1)
        bpr_order = result.sort_values(
            ["bpr_percentile", "item_id"], ascending=[False, True], kind="mergesort"
        ).index
        neural_order = result.sort_values(
            ["neural_percentile", "item_id"], ascending=[False, True], kind="mergesort"
        ).index
        result.loc[bpr_order, "bpr_rank"] = np.arange(1, len(result) + 1)
        result.loc[neural_order, "neural_rank"] = np.arange(1, len(result) + 1)
        return result.head(top_k).copy()

    def _require_user(self, user_id: int) -> None:
        if int(user_id) not in self.user_to_index:
            raise InferenceError(f"unknown user_id: {user_id}")
