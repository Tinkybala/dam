"""Report-sized layouts for the archived Anime analysis (no data mutation)."""

import statistics

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


BLUE = "#3976A8"
ORANGE = "#B96820"
GREEN = "#247A64"
INK = "#213343"
GREY = "#77838F"


def clean(ax, axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCD5DD")
    ax.tick_params(length=0, pad=5)
    ax.grid(axis=axis, color="#E5E9ED", linewidth=.65)
    ax.set_axisbelow(True)


def title(fig, heading, subtitle, left=.20):
    fig.text(left, .977, heading, ha="left", va="top", fontsize=13, weight="bold")
    fig.text(left, .943, subtitle, ha="left", va="top", fontsize=8.5, color="#455767")


def save(fig, output, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def model_comparison(data, output, summary, labels, colors, differences):
    """Full-range scores above; paired baseline differences below."""
    models = ["popular", "mlp", "gmf", "neumf", "bpr", "weighted_neumf", "ensemble_w0p7"]
    compared = ["mlp", "gmf", "neumf", "weighted_neumf", "ensemble_w0p7"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 8.0), gridspec_kw={"height_ratios": [1.15, 1]})
    for j, (metric, name) in enumerate([("test_ndcg@10", "NDCG@10"), ("test_hit_rate@10", "Hit Rate@10")]):
        ax = axes[0, j]
        means = [summary[m][metric][0] for m in models]
        sds = [summary[m][metric][1] or 0 for m in models]
        ax.barh(range(len(models)), means, height=.59, color=[colors[m] for m in models],
                xerr=sds, error_kw={"ecolor": INK, "capsize": 2, "elinewidth": .9})
        for i, mean in enumerate(means):
            ax.text(.035, i, f"{mean:.4f}", va="center", color="white", fontsize=9)
        ax.set_yticks(range(len(models)), [labels[m] for m in models] if j == 0 else [])
        ax.set_ylim(len(models) - .5, -.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_title(f"{'A' if j == 0 else 'B'}  {name}", loc="left", pad=10)
        ax.set_xlabel("Absolute test score")
        clean(ax)

        ax = axes[1, j]
        ax.axvline(0, color=INK, lw=.9, ls="--")
        for i, model in enumerate(compared):
            ds = differences(data, model, "bpr", metric)
            mean, sd = statistics.mean(ds), statistics.stdev(ds)
            ax.errorbar(mean, i, xerr=sd, fmt="D", ms=4.5, color=colors[model],
                        capsize=3, elinewidth=1.1)
            ax.annotate(f"{mean:+.4f}", (mean, i), xytext=(0, -13),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set_yticks(range(len(compared)), [labels[m] for m in compared] if j == 0 else [])
        ax.set_ylim(len(compared) - .35, -.5)
        ax.set_xlim((-.072, .047) if j == 0 else (-.026, .011))
        ax.set_xticks([-.06, -.03, 0, .03] if j == 0 else [-.02, -.01, 0, .01])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.2f}" if v else "0"))
        ax.set_title(f"{'C' if j == 0 else 'D'}  Gain over BPR", loc="left", pad=11)
        ax.set_xlabel(f"Δ test {name}")
        clean(ax)
    title(fig, "Which systems improve on standalone BPR?",
          "Anime test · 60,384 warm users · 1 positive + 99 fixed negatives")
    fig.text(.20, .042, "Top: mean ± sample SD; Popular has one run. Bottom: mean ± sample SD of\n"
             "paired seed differences (42/43/44), not confidence intervals. Positive = better.\n"
             "The five alternatives below share BPR's seeds; Popular is shown only above.", fontsize=8)
    fig.subplots_adjust(left=.20, right=.98, top=.875, bottom=.15, hspace=.43, wspace=.23)
    save(fig, output, "model_comparison")


def fusion_gains(data, output, differences):
    """Two endpoints, actual components, visible seed observations and their mean."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.9))
    comparisons = [("bpr_ensemble_component", "vs BPR\ncomponent", BLUE),
                   ("weighted_neumf", "vs Weighted\nNeuMF", ORANGE)]
    markers, offsets = ["o", "s", "^"], [-.075, 0, .075]
    for j, (metric, label) in enumerate([("test_ndcg@10", "NDCG@10"), ("test_hit_rate@10", "Hit Rate@10")]):
        ax = axes[j]
        for x, (model, _, color) in enumerate(comparisons):
            ds = differences(data, "ensemble_w0p7", model, metric)
            for delta, marker, offset in zip(ds, markers, offsets):
                ax.scatter(x + offset, delta, marker=marker, color=color, s=34, zorder=4,
                           edgecolors="white", linewidth=.5)
            mean = statistics.mean(ds)
            ax.hlines(mean, x - .20, x + .20, color=INK, lw=1.7, zorder=3)
            ax.text(x, max(ds) + .00125, f"mean +{mean:.4f}", ha="center", fontsize=8.5, weight="bold")
        ax.axhline(0, color=INK, lw=1)
        ax.set_xlim(-.5, 1.5)
        ax.set_ylim(-.001, .021)
        ax.set_yticks([0, .005, .010, .015, .020])
        ax.set_xticks([0, 1], [entry[1] for entry in comparisons])
        ax.set_ylabel(f"Δ test {label}")
        ax.set_title(f"{'A' if j == 0 else 'B'}  {label}", loc="left", pad=12)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.3f}" if v else "0"))
        clean(ax, "y")
    title(fig, "Does fusion improve both components on both metrics?",
          "Fixed rank blend: 0.7 BPR component + 0.3 Weighted NeuMF", left=.10)
    handles = [Line2D([], [], marker=m, color=GREY, linestyle="", label=f"Seed {s}")
               for m, s in zip(markers, [42, 43, 44])]
    handles.append(Line2D([], [], color=INK, lw=1.7, label="Mean"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .10),
               ncol=4, frameon=False, fontsize=8)
    fig.text(.10, .035, "All 12 paired gains are positive. Points are training seeds, not individual users.\n"
             "Anime test: 60,384 warm users; 100 candidates/user. No significance test is implied.", fontsize=8)
    fig.subplots_adjust(left=.10, right=.98, top=.82, bottom=.28, wspace=.43)
    save(fig, output, "component_ablation")


def alpha_analysis(data, output):
    records = sorted(data["alpha_sweep"]["records"], key=lambda row: row["alpha"])
    baseline = next(row for row in records if row["alpha"] == 0)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.9))
    for ax, metric, label, color, limits, ticks in [
        (axes[0], "validation_ndcg@10", "NDCG@10", BLUE, (-.00225, .00245), [-.002, -.001, 0, .001, .002]),
        (axes[1], "validation_hit_rate@10", "Hit Rate@10", ORANGE, (-.0011, .0082), [0, .002, .004, .006, .008]),
    ]:
        xs = [r["alpha"] for r in records]
        ys = [r[metric] - baseline[metric] for r in records]
        ax.axvspan(.4, .6, color=GREEN, alpha=.10, linewidth=0)
        ax.axhline(0, color=GREY, ls="--", lw=.9)
        ax.plot(xs, ys, color=color, marker="o", ms=4, lw=1.5)
        selected = next(i for i, x in enumerate(xs) if x == .5)
        ax.scatter([.5], [ys[selected]], color=GREEN, marker="D", s=45, zorder=4)
        for i, (x, y) in enumerate(zip(xs, ys)):
            if i == selected or x in [1.5, 2]:
                ax.annotate(f"{y:+.4f}", (x, y), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=8)
        ax.set_xlim(-.15, 2.15)
        ax.set_ylim(*limits)
        ax.set_xticks(xs)
        ax.set_yticks(ticks)
        ax.set_xlabel("Confidence weight alpha")
        ax.set_ylabel(f"Δ validation {label}")
        ax.set_title(f"{'A' if metric.endswith('ndcg@10') else 'B'}  {label}", loc="left", pad=23)
        ax.text(0, 1.025, f"alpha=0: {baseline[metric]:.6f}", transform=ax.transAxes, fontsize=8.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.3f}" if v else "0"))
        clean(ax, "y")
    title(fig, "Why select alpha = 0.5?",
          "Phase D validation · 10,000 users · seed 42 · 100 candidates/user", left=.10)
    fig.text(.10, .12, "Green diamond/band: alpha=0.5, selected by the primary metric NDCG.\n"
             "HR peaks at alpha=1.5, where NDCG is below the unweighted baseline.", fontsize=8.5)
    fig.text(.10, .038, "Axes show changes from alpha=0. Other configured hyperparameters are fixed;\n"
             "early-stopping epochs vary. One seed per alpha; no replicated uncertainty estimate.", fontsize=8)
    fig.subplots_adjust(left=.10, right=.98, top=.77, bottom=.29, wspace=.43)
    save(fig, output, "parameter_analysis")
