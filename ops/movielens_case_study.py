"""Select deterministic sampled-test examples after the frozen campaign is complete."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

from src.evaluate import rank_candidates


COMMIT = "61863193e11650cdd579a102f172a51582502552"
SEED = 42


def titles_from_archive(path: Path) -> dict[int, str]:
    with zipfile.ZipFile(path) as archive:
        text = archive.read("ml-1m/movies.dat").decode("latin-1")
    titles = {}
    for line in text.splitlines():
        parts = line.split("::", 2)
        if len(parts) != 3:
            raise ValueError("invalid movies.dat row")
        titles[int(parts[0])] = parts[1]
    return titles


def ranked_positive(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(path)
    ranked = rank_candidates(predictions)
    assert ranked.groupby("user_id").size().eq(100).all()
    assert ranked.groupby("user_id").label.sum().eq(1).all()
    positive = ranked.loc[ranked.label.eq(1), ["user_id", "item_id", "rank"]]
    return ranked, positive.set_index("user_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--movie-archive", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results" / "movielens"
    summary = json.loads((results / "final_summary.json").read_text())
    assert summary["status"] == "complete" and summary["source_commit"] == COMMIT
    systems = {}
    positives = {}
    for model in ("bpr", "weighted_neumf", "ensemble"):
        ranked, positive = ranked_positive(results / f"{model}-seed-{SEED}" / "test_predictions.parquet")
        systems[model] = ranked
        positives[model] = positive
    require_users = set(positives["ensemble"].index)
    assert len(require_users) == 6034
    assert all(set(table.index) == require_users for table in positives.values())
    for model in ("bpr", "weighted_neumf"):
        assert positives[model].item_id.equals(positives["ensemble"].item_id)

    ensemble_rank = positives["ensemble"]["rank"]
    bpr_rank = positives["bpr"]["rank"]
    weighted_rank = positives["weighted_neumf"]["rank"]
    categories = {
        "rank_one_success": ensemble_rank.eq(1),
        "rank_ten_boundary": ensemble_rank.eq(10),
        "miss_after_ten": ensemble_rank.gt(10),
        "bpr_hit_weighted_miss": bpr_rank.le(10) & weighted_rank.gt(10),
        "weighted_hit_bpr_miss": weighted_rank.le(10) & bpr_rank.gt(10),
    }
    title_map = titles_from_archive(args.movie_archive)
    train = pd.read_parquet(root / "artifacts" / "movielens-1m-r4" / "train_positives.parquet")
    observed = pd.read_parquet(root / "artifacts" / "movielens-1m-r4" / "observed_interactions.parquet")
    selected = []
    for category, mask in categories.items():
        matching_users = sorted(ensemble_rank.index[mask].astype(int))
        for user_id in matching_users[:2]:
            held = positives["ensemble"].loc[user_id]
            history = train.loc[train.user_id.eq(user_id)].sort_values(["rating", "anime_id"], ascending=[False, True])
            top = systems["ensemble"].loc[systems["ensemble"].user_id.eq(user_id)].head(10)
            selected.append({
                "category": category,
                "source_user_id": int(user_id),
                "positive_movie_id": int(held.item_id),
                "positive_title": title_map.get(int(held.item_id), "[title unavailable]"),
                "positive_rank": {model: int(positives[model].loc[user_id, "rank"]) for model in positives},
                "training_positive_count": len(history),
                "observed_count": int(observed.user_id.eq(user_id).sum()),
                "training_history_preview": [title_map.get(int(item), "[title unavailable]")
                                             for item in history.anime_id.head(5)],
                "top_ten_sampled_candidate_titles": [title_map.get(int(item), "[title unavailable]")
                                                      for item in top.item_id],
            })
    record = {"status": "complete", "source_commit": COMMIT, "seed": SEED,
              "selection_rule": "lowest source user IDs, up to 2 per category",
              "protocol": "one test positive plus 99 fixed negatives per warm user",
              "category_counts": {key: int(mask.sum()) for key, mask in categories.items()},
              "cases": selected}
    with (results / "case_study_internal.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    public = {**record, "cases": []}
    for index, item in enumerate(selected, start=1):
        public["cases"].append({key: value for key, value in item.items() if key != "source_user_id"} |
                               {"case_label": f"Case {index:02d}"})
    with (results / "case_study_anonymous.json").open("x", encoding="utf-8") as handle:
        json.dump(public, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": "complete", "case_count": len(selected),
                      "category_counts": record["category_counts"]}, indent=2))


if __name__ == "__main__":
    main()
