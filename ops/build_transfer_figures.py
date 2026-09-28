"""Render Anime and MovieLens report panels from frozen, archived summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


LABELS = {"popular": "Popular", "bpr": "BPR", "gmf": "GMF", "mlp": "MLP",
          "neumf": "NeuMF", "weighted_neumf": "Weighted NeuMF", "ensemble_w0p7": "Fixed ensemble"}
BLUE, ORANGE, GREEN = "#3976A8", "#B96820", "#247A64"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def chart_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.titlesize": 10, "axes.titleweight": "bold",
                         "text.color": "#213343", "axes.labelcolor": "#213343",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})


def save(fig, target: Path):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(target.with_suffix("." + extension), dpi=300, facecolor="white")
    plt.close(fig)


def anime_value(anime, model, metric):
    rows = anime["final"]["models"][model]["per_seed"]
    vals = [r[f"test_{metric}"] for r in rows]
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


def plot_comparison(anime, movie, output):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 8.2))
    for row, (dataset, models) in enumerate([
        ("Anime", ["popular", "gmf", "mlp", "neumf", "bpr", "weighted_neumf", "ensemble_w0p7"]),
        ("MovieLens 1M", ["popular", "gmf", "neumf", "bpr", "weighted_neumf", "ensemble_w0p7"]),
    ]):
        for col, (metric, label) in enumerate([("ndcg@10", "NDCG@10"), ("hit_rate@10", "Hit Rate@10")]):
            ax = axes[row, col]
            if dataset == "Anime":
                values = [anime_value(anime, model, metric) for model in models]
            else:
                values = [(movie["systems"][model][f"test_{metric}"]["mean"],
                           movie["systems"][model][f"test_{metric}"]["sample_sd"] or 0) for model in models]
            means, sds = zip(*values)
            colors = [GREEN if model == "ensemble_w0p7" else ORANGE if model == "weighted_neumf" else
                      "#77838F" if model == "popular" else BLUE for model in models]
            ax.barh(range(len(models)), means, xerr=sds, color=colors, height=.58,
                    error_kw={"ecolor": "#213343", "capsize": 2, "elinewidth": 1})
            for i, value in enumerate(means):
                ax.text(.025, i, f"{value:.4f}", va="center", color="white", fontsize=8.5)
            ax.set_yticks(range(len(models)), [LABELS[model] for model in models] if col == 0 else [])
            ax.set_ylim(len(models) - .5, -.5)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.set_title(f"{dataset}: {label}", loc="left", pad=9)
            ax.set_xlabel("Within-dataset test score")
            ax.grid(axis="x", color="#E5E9ED", linewidth=.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
    fig.text(.20, .98, "Model comparison within each dataset", ha="left", va="top", fontsize=13, weight="bold")
    fig.text(.20, .95, "One held-out positive + 99 fixed negatives per warm user", ha="left", va="top", fontsize=8.5)
    fig.text(.20, .039, "Mean ± sample SD across seeds 42/43/44; Popular has one run.\n"
             "Positive threshold: Anime rating ≥7; MovieLens rating ≥4.\n"
             "Users and catalogs differ. Compare model order and gains within panels.", fontsize=7.5)
    fig.subplots_adjust(left=.20, right=.98, top=.89, bottom=.15, hspace=.43, wspace=.20)
    save(fig, output / "model_comparison")


def plot_fusion(movie, output):
    runs = movie["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.9))
    specs = [("bpr", "vs BPR", BLUE), ("weighted_neumf", "vs Weighted\nNeuMF", ORANGE)]
    markers = ["o", "s", "^"]
    for col, (metric, label) in enumerate([("ndcg@10", "NDCG@10"), ("hit_rate@10", "Hit Rate@10")]):
        ax = axes[col]
        ax.axhline(0, color="#213343", lw=.9)
        for x, (component, _, color) in enumerate(specs):
            values = [runs[f"ensemble-seed-{seed}"]["test"][metric] -
                      runs[f"{component}-seed-{seed}"]["test"][metric] for seed in (42, 43, 44)]
            for seed_index, value in enumerate(values):
                ax.scatter(x + [-.09, 0, .09][seed_index], value, s=40,
                           color=color, marker=markers[seed_index], zorder=3,
                           edgecolor="white", linewidth=.5)
            ax.hlines(statistics.mean(values), x - .20, x + .20, color="#213343", lw=1.7)
            ax.text(x, max(values) + .0011, f"mean +{statistics.mean(values):.4f}",
                    ha="center", fontsize=8, weight="bold")
        ax.set_xticks([0, 1], [spec[1] for spec in specs])
        ax.set_xlim(-.5, 1.5)
        ax.set_ylim(-.001, .022)
        ax.set_yticks([0, .005, .010, .015, .020])
        ax.set_ylabel(f"Δ test {label}")
        ax.set_title(label, loc="left", pad=10)
        ax.grid(axis="y", color="#E5E9ED", linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.10, .98, "MovieLens: fixed ensemble gains by seed", va="top", fontsize=13, weight="bold")
    fig.text(.10, .94, "0.7 BPR percentile rank + 0.3 Weighted NeuMF percentile rank", va="top", fontsize=8.5)
    handles = [Line2D([], [], marker=marker, color="#77838F", linestyle="", label=f"Seed {seed}")
               for marker, seed in zip(markers, (42, 43, 44))]
    handles.append(Line2D([], [], color="#213343", lw=1.7, label="Mean"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .10), ncol=4, frameon=False, fontsize=8)
    fig.text(.10, .035, "One positive + 99 unseen negatives/user. Differences are paired by seed.\n"
             "Three seeds describe training variation; these are not confidence intervals.", fontsize=8)
    fig.subplots_adjust(left=.10, right=.98, top=.81, bottom=.28, wspace=.40)
    save(fig, output / "component_ablation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anime", type=Path, default=Path("evidence/anime_report_figure_data_20260926.json"))
    parser.add_argument("--movie", type=Path, default=Path("evidence/movielens_transfer_summary_20260928.json"))
    parser.add_argument("--output", type=Path, default=Path("results/movielens/report_20260928"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty figure directory")
    anime = json.loads(args.anime.read_text(encoding="utf-8"))
    movie = json.loads(args.movie.read_text(encoding="utf-8"))
    if movie["status"] != "complete" or movie["source_commit"] != "61863193e11650cdd579a102f172a51582502552":
        raise ValueError("MovieLens final summary provenance mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    chart_style()
    plot_comparison(anime, movie, args.output)
    plot_fusion(movie, args.output)
    files = {path.name: digest(path) for path in sorted(args.output.iterdir())}
    with (args.output / "figure_manifest.json").open("x", encoding="utf-8") as output:
        json.dump({"status": "complete", "anime_input_sha256": digest(args.anime),
                   "movielens_input_sha256": digest(args.movie),
                   "script_sha256": digest(Path(__file__)), "files": files}, output, indent=2)
        output.write("\n")
    print("Created two report figures in PNG/SVG/PDF from frozen summaries.")


if __name__ == "__main__":
    main()
