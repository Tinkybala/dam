"""Build report figures from archived, sanitized Anime metrics; never train models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .anime_figure_layout import model_comparison, fusion_gains, alpha_analysis
except ImportError:  # Direct script invocation.
    from anime_figure_layout import model_comparison, fusion_gains, alpha_analysis


LABELS = {
    "popular": "Popular", "mlp": "MLP", "gmf": "GMF", "neumf": "NeuMF",
    "bpr": "BPR (standalone)", "weighted_neumf": "Weighted NeuMF",
    "bpr_ensemble_component": "BPR component", "ensemble_w0p7": "Fixed ensemble",
}
COLORS = {key: "#587895" for key in LABELS}
COLORS.update(popular="#77838F", weighted_neumf="#C57528",
              bpr_ensemble_component="#3976A8", ensemble_w0p7="#247A64")
SEEDS = [42, 43, 44]
METRICS = ["test_ndcg@10", "test_hit_rate@10"]
PROTOCOL = "Anime | 60,384 warm users | 1 held-out positive + 99 fixed unseen negatives per user"


def summarize(data):
    """Recompute mean/sample SD from the seed records, rejecting missing runs."""
    result = {}
    for model in LABELS:
        rows = data["final"]["models"][model]["per_seed"]
        expected = [42] if model == "popular" else SEEDS
        if sorted(row["seed"] for row in rows) != expected:
            raise ValueError(f"unexpected seeds for {model}")
        result[model] = {}
        for metric in METRICS:
            values = [row[metric] for row in rows]
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise ValueError(f"invalid {metric} for {model}")
            result[model][metric] = (statistics.mean(values),
                                      statistics.stdev(values) if len(values) > 1 else None)
    return result


def paired_differences(data, model, baseline, metric="test_ndcg@10"):
    """Pair fixed training seeds; never subtract unrelated row positions."""
    models = data["final"]["models"]
    for name in (model, baseline):
        seeds = [r["seed"] for r in models[name]["per_seed"]]
        if sorted(seeds) != SEEDS:
            raise ValueError(f"unexpected seeds for {name}")
    reference = {r["seed"]: r for r in models[baseline]["per_seed"]}
    selected = {r["seed"]: r for r in models[model]["per_seed"]}
    return [selected[s][metric] - reference[s][metric] for s in SEEDS]


def paired_gains(data, component, metric="test_ndcg@10"):
    return paired_differences(data, "ensemble_w0p7", component, metric)


def build(data, output):
    summary = summarize(data)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "bold",
                         "text.color": "#213343", "axes.labelcolor": "#213343",
                         "xtick.color": "#455767", "ytick.color": "#213343",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    models = ["popular", "mlp", "gmf", "neumf", "bpr", "weighted_neumf", "ensemble_w0p7"]
    model_comparison(data, output, summary, LABELS, COLORS, paired_differences)
    fusion_gains(data, output, paired_differences)
    alpha_analysis(data, output)
    records = sorted(data["alpha_sweep"]["records"], key=lambda row: row["alpha"])
    baseline = next(row for row in records if row["alpha"] == 0)

    with (output / "baseline_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "baseline", "metric", "seed", "difference"])
        for model in ["mlp", "gmf", "neumf", "weighted_neumf", "ensemble_w0p7"]:
            for metric in METRICS:
                for seed, delta in zip(SEEDS, paired_differences(data, model, "bpr", metric)):
                    writer.writerow([model, "bpr", metric, seed, delta])

    with (output / "model_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "split", "model", "role", "seeds", "ndcg_mean", "ndcg_sample_sd", "hr_mean", "hr_sample_sd"])
        for model in [*models, "bpr_ensemble_component"]:
            writer.writerow(["Anime", "test", model, "component" if model == "bpr_ensemble_component" else "system",
                             "42" if model == "popular" else "42;43;44", *summary[model][METRICS[0]], *summary[model][METRICS[1]]])
    with (output / "alpha_scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "seed", "alpha", "ndcg", "hit_rate", "ndcg_delta_from_zero", "hr_delta_from_zero"])
        for r in records:
            writer.writerow(["validation", r["seed"], r["alpha"], r["validation_ndcg@10"], r["validation_hit_rate@10"],
                             r["validation_ndcg@10"] - baseline["validation_ndcg@10"], r["validation_hit_rate@10"] - baseline["validation_hit_rate@10"]])
    with (output / "paired_component_gains.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "comparison", "ndcg_delta", "hr_delta"])
        for model in ["bpr_ensemble_component", "weighted_neumf"]:
            for seed, ndcg, hr in zip(SEEDS, paired_gains(data, model), paired_gains(data, model, "test_hit_rate@10")):
                writer.writerow([seed, f"ensemble_minus_{model}", ndcg, hr])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("evidence/anime_report_figure_data_20260926.json"))
    parser.add_argument("--output", type=Path, default=Path("results/anime_report_figures_20260926_v2"))
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty figure directory")
    args.output.mkdir(parents=True, exist_ok=True)
    build(data, args.output)
    metadata = {"input_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "layout_sha256": hashlib.sha256(Path(__file__).with_name("anime_figure_layout.py").read_bytes()).hexdigest(),
                "matplotlib_version": matplotlib.__version__,
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())}}
    (args.output / "figure_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Created 3 figures in PNG/SVG/PDF, 4 data tables, and manifest: {args.output}")


if __name__ == "__main__":
    main()
