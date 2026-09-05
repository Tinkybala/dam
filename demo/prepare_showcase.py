"""Prepare anonymous demo users and cache only posters used by the showcase."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from src.data import load_anime_ratings, remove_ambiguous_pairs

from .data_access import load_and_validate_manifest


ANILIST_ENDPOINT = "https://graphql.anilist.co"
USER_AGENT = "dam-recommender-educational-demo/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=Path("demo_bundle"))
    parser.add_argument("--user-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, rows: int | None = None) -> dict[str, int | str]:
    record: dict[str, int | str] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if rows is not None:
        record["rows"] = rows
    return record


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_profiles(
    ratings_path: Path, bundle: Path, user_count: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if user_count < 1:
        raise ValueError("user_count must be positive")
    user_mapping = pd.read_parquet(bundle / "user_mapping.parquet")
    item_mapping = pd.read_parquet(bundle / "item_mapping.parquet")
    current = pd.read_parquet(bundle / "demo_users.parquet")
    if user_count < len(current):
        raise ValueError("user_count cannot remove existing anonymous users")

    ratings = load_anime_ratings(ratings_path)
    deduplicated, _ = remove_ambiguous_pairs(ratings)
    known_users = set(user_mapping["user_id"].astype(int))
    known_items = set(item_mapping["anime_id"].astype(int))
    observed = deduplicated.loc[
        deduplicated["user_id"].isin(known_users)
        & deduplicated["anime_id"].isin(known_items),
        ["user_id", "anime_id"],
    ].drop_duplicates()

    metadata = pd.read_parquet(bundle.parent / "demo" / "assets" / "anime_metadata.parquet")
    adult_items = set(
        metadata.loc[
            metadata["genre"].fillna("").str.split(",").apply(
                lambda values: any(value.strip() == "Hentai" for value in values)
            ),
            "anime_id",
        ].astype(int)
    )
    adult_users = set(observed.loc[observed["anime_id"].isin(adult_items), "user_id"].astype(int))
    selected = [
        user for user in current["user_id"].astype(int).tolist() if user not in adult_users
    ]
    needed = user_count - len(selected)
    if needed:
        counts = observed.groupby("user_id")["anime_id"].size()
        candidates = counts.loc[counts.between(20, 250)].index.to_numpy(dtype=np.int64)
        candidates = candidates[
            ~np.isin(candidates, np.asarray(selected, dtype=np.int64))
            & ~np.isin(candidates, np.asarray(sorted(adult_users), dtype=np.int64))
        ]
        if len(candidates) < needed:
            raise ValueError("not enough model-known users satisfy the demo history range")
        rng = np.random.default_rng(seed)
        selected.extend(int(value) for value in rng.choice(candidates, size=needed, replace=False))

    selected_observed = observed.loc[observed["user_id"].isin(selected)].copy()
    history_counts = selected_observed.groupby("user_id")["anime_id"].size().to_dict()
    demo_users = pd.DataFrame(
        {
            "user_id": selected,
            "demo_label": [f"Anonymous user {index:02d}" for index in range(1, len(selected) + 1)],
            "history_count": [int(history_counts[user]) for user in selected],
        }
    )
    return demo_users, selected_observed.sort_values(["user_id", "anime_id"]).reset_index(drop=True)


def _query_anilist(anime_ids: list[int], batch_size: int = 20) -> dict[int, str]:
    urls: dict[int, str] = {}
    for start in range(0, len(anime_ids), batch_size):
        batch = anime_ids[start : start + batch_size]
        fields = "\n".join(
            f"m{index}: Media(idMal: {anime_id}, type: ANIME) "
            "{ idMal coverImage { large medium } }"
            for index, anime_id in enumerate(batch)
        )
        payload = json.dumps({"query": f"query {{\n{fields}\n}}"}).encode("utf-8")
        request = urllib.request.Request(
            ANILIST_ENDPOINT,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT},
        )
        response_data: dict[str, object] | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    response_data = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 and exc.code < 500:
                    break
                wait = int(exc.headers.get("Retry-After", 2 ** attempt))
                time.sleep(max(1, wait))
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2 ** attempt)
        if response_data is not None:
            for media in dict(response_data.get("data") or {}).values():
                if not media:
                    continue
                images = media.get("coverImage") or {}
                url = images.get("large") or images.get("medium")
                if url:
                    urls[int(media["idMal"])] = str(url)
        time.sleep(2.1)
    return urls


def _download_posters(bundle: Path, anime_ids: list[int], urls: dict[int, str]) -> dict[str, dict[str, str]]:
    poster_dir = bundle / "posters"
    poster_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict[str, str]] = {}
    for anime_id in anime_ids:
        url = urls.get(anime_id)
        if not url:
            continue
        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        relative = Path("posters") / f"{anime_id}{suffix}"
        destination = bundle / relative
        if not destination.is_file():
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = response.read(5 * 1024 * 1024 + 1)
                if not payload or len(payload) > 5 * 1024 * 1024:
                    continue
                destination.write_bytes(payload)
            except (OSError, urllib.error.URLError, TimeoutError):
                continue
        index[str(anime_id)] = {
            "file": relative.as_posix(),
            "provider": "AniList",
            "source_url": url,
        }
    return index


def prepare(args: argparse.Namespace) -> dict[str, int]:
    bundle = args.bundle.resolve()
    ratings_path = args.ratings.resolve()
    manifest = load_and_validate_manifest(bundle)
    existing_poster_index: dict[str, dict[str, str]] = {}
    existing_index_path = bundle / "poster_index.json"
    if existing_index_path.is_file():
        existing_poster_index = json.loads(existing_index_path.read_text(encoding="utf-8"))
    demo_users, observed = _build_profiles(ratings_path, bundle, args.user_count, args.seed)
    demo_users_path = bundle / "demo_users.parquet"
    observed_path = bundle / "observed_by_user.parquet"
    demo_users.to_parquet(demo_users_path, index=False)
    observed.to_parquet(observed_path, index=False)

    files = dict(manifest["files"])
    files["demo_users.parquet"] = _file_record(demo_users_path, len(demo_users))
    files["observed_by_user.parquet"] = _file_record(observed_path, len(observed))
    manifest["files"] = files
    manifest["demo_user_count"] = len(demo_users)
    manifest["demo_observed_row_count"] = len(observed)
    _write_manifest(bundle / "manifest.json", manifest)

    from .inference import Recommender

    engine = Recommender(bundle)
    display_ids: set[int] = set()
    for user_id in demo_users["user_id"].astype(int):
        display_ids.update(engine.history(user_id, limit=5)["anime_id"].astype(int))
        display_ids.update(engine.recommend(user_id, top_k=10)["item_id"].astype(int))

    ordered_ids = sorted(display_ids)
    urls = {
        anime_id: existing_poster_index[str(anime_id)]["source_url"]
        for anime_id in ordered_ids
        if str(anime_id) in existing_poster_index
        and existing_poster_index[str(anime_id)].get("source_url")
    }
    missing_urls = [anime_id for anime_id in ordered_ids if anime_id not in urls]
    urls.update(_query_anilist(missing_urls))
    poster_index = _download_posters(bundle, ordered_ids, urls)
    poster_index_path = bundle / "poster_index.json"
    poster_index_path.write_text(
        json.dumps(poster_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    files = {
        key: value
        for key, value in dict(manifest["files"]).items()
        if key != "poster_index.json" and not key.startswith("posters/")
    }
    files["poster_index.json"] = _file_record(poster_index_path, len(poster_index))
    for record in poster_index.values():
        relative = record["file"]
        files[relative] = _file_record(bundle / relative)
    manifest["files"] = files
    manifest["poster_provider"] = "AniList public media metadata"
    manifest["poster_display_item_count"] = len(ordered_ids)
    manifest["poster_cached_count"] = len(poster_index)
    _write_manifest(bundle / "manifest.json", manifest)

    load_and_validate_manifest(bundle)
    return {
        "demo_users": len(demo_users),
        "observed_rows": len(observed),
        "display_items": len(ordered_ids),
        "cached_posters": len(poster_index),
        "missing_posters": len(ordered_ids) - len(poster_index),
    }


def main() -> None:
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
