"""Validated access to the local demo bundle.

The demo deliberately has a small data-access surface.  It reads only the
frozen inference bundle and curated catalog metadata; it never opens training,
validation, or test result files.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


class BundleError(RuntimeError):
    """Raised when the local demo bundle is missing or fails validation."""


REQUIRED_BUNDLE_FILES = (
    "bpr_seed42_model.pt",
    "weighted_neumf_seed42_model.pt",
    "user_mapping.parquet",
    "item_mapping.parquet",
    "observed_by_user.parquet",
    "demo_users.parquet",
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(project_root: Path, bundle_dir: Path, relative: str) -> Path:
    """Resolve a manifest path without permitting absolute paths."""

    path = Path(relative)
    if path.is_absolute() or path.drive:
        raise BundleError(f"manifest contains an absolute path: {relative}")
    if relative.startswith("demo/"):
        resolved = project_root / path
    else:
        resolved = bundle_dir / path
    return resolved


def load_and_validate_manifest(bundle_dir: Path) -> dict[str, Any]:
    """Load the manifest and verify every declared file hash and size."""

    bundle_dir = bundle_dir.expanduser().resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"missing demo manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"invalid demo manifest: {manifest_path}") from exc

    if manifest.get("schema_version") != 1:
        raise BundleError("unsupported demo manifest schema")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not _COMMIT_RE.fullmatch(source_commit):
        raise BundleError("manifest source_commit must be a 40-character hash")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise BundleError("manifest files must be an object")
    project_root = bundle_dir.parent
    for relative, record in files.items():
        if not isinstance(relative, str) or not isinstance(record, dict):
            raise BundleError("manifest file records are malformed")
        path = _manifest_path(project_root, bundle_dir, relative)
        if not path.is_file():
            raise BundleError(f"manifest file is missing: {relative}")
        expected_size = record.get("bytes")
        expected_hash = record.get("sha256")
        if not isinstance(expected_size, int) or not isinstance(expected_hash, str):
            raise BundleError(f"manifest record is incomplete: {relative}")
        if path.stat().st_size != expected_size:
            raise BundleError(f"size mismatch for {relative}")
        if _sha256(path) != expected_hash.lower():
            raise BundleError(f"SHA-256 mismatch for {relative}")

    missing = [name for name in REQUIRED_BUNDLE_FILES if name not in files]
    if missing:
        raise BundleError(f"manifest missing required files: {missing}")
    return manifest


def _read_table(path: Path, required: set[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise BundleError(f"missing {label}: {path}")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - backend-specific exception
        raise BundleError(f"cannot read {label}: {path}") from exc
    missing = required.difference(frame.columns)
    if missing:
        raise BundleError(f"{label} missing columns: {sorted(missing)}")
    return frame


def load_bundle(bundle_dir: str | Path, metadata_path: str | Path | None = None) -> dict[str, Any]:
    """Load validated mappings, demo history, and display metadata."""

    bundle = Path(bundle_dir).expanduser().resolve()
    manifest = load_and_validate_manifest(bundle)
    user_mapping = _read_table(
        bundle / "user_mapping.parquet", {"user_id", "user_index"}, "user mapping"
    )
    item_mapping = _read_table(
        bundle / "item_mapping.parquet", {"anime_id", "item_index"}, "item mapping"
    )
    observed = _read_table(
        bundle / "observed_by_user.parquet", {"user_id", "anime_id"}, "observed history"
    )
    demo_users = _read_table(
        bundle / "demo_users.parquet", {"user_id", "demo_label"}, "demo users"
    )

    if user_mapping["user_id"].duplicated().any() or user_mapping["user_index"].duplicated().any():
        raise BundleError("user mapping contains duplicate keys")
    if item_mapping["anime_id"].duplicated().any() or item_mapping["item_index"].duplicated().any():
        raise BundleError("item mapping contains duplicate keys")
    if set(user_mapping["user_index"]) != set(range(len(user_mapping))):
        raise BundleError("user mapping indices must be contiguous")
    if set(item_mapping["item_index"]) != set(range(len(item_mapping))):
        raise BundleError("item mapping indices must be contiguous")
    if not demo_users["user_id"].isin(set(user_mapping["user_id"])).all():
        raise BundleError("demo user is absent from user mapping")
    if not observed["user_id"].isin(set(demo_users["user_id"])).all():
        raise BundleError("observed history contains a non-demo user")
    if not observed["anime_id"].isin(set(item_mapping["anime_id"])).all():
        raise BundleError("observed history contains an item outside the warm catalog")
    if observed.duplicated(["user_id", "anime_id"]).any():
        raise BundleError("observed history contains duplicate pairs")

    root = bundle.parent
    if metadata_path is None:
        metadata_path = root / "demo" / "assets" / "anime_metadata.parquet"
    metadata = _read_table(Path(metadata_path), {"anime_id", "name"}, "anime metadata")
    if metadata["anime_id"].duplicated().any():
        raise BundleError("anime metadata contains duplicate anime_id values")

    return {
        "manifest": manifest,
        "user_mapping": user_mapping.sort_values("user_index").reset_index(drop=True),
        "item_mapping": item_mapping.sort_values("item_index").reset_index(drop=True),
        "observed": observed.sort_values(["user_id", "anime_id"]).reset_index(drop=True),
        "demo_users": demo_users.reset_index(drop=True),
        "metadata": metadata.copy(),
    }

