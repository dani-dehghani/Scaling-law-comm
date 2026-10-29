"""Regenerate comparison tables and figures from packaged evidence; no training."""
from pathlib import Path
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vit_tiny_comparison_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "artifacts"
OUT = HERE.parent / "reports" / "figures" / "historical_comparison"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = [
    ("before", "baselines/formal_lr_003", "Before: manual LR 0.003", "#c66528"),
    ("after", "cifar100_main", "After: HPO-selected recipe", "#087e8b"),
    ("manual001", "baselines/formal_lr_001", "Other manual run: LR 0.001", "#7968a8"),
]
data = {}
for key, path, _, _ in RUNS:
    rows = list(csv.DictReader((REPORTS / path / "aggregated_by_D.csv").open()))
    data[key] = {int(r["D"]): {k: float(v) for k, v in r.items()} for r in rows}

comparison = []
for D in sorted(data["before"].keys() & data["after"].keys()):
    before, after = data["before"][D], data["after"][D]
    comparison.append({
        "D": D,
        "before_loss": before["test_loss_mean"],
        "after_loss": after["test_loss_mean"],
        "loss_reduction_percent": 100 * (before["test_loss_mean"] - after["test_loss_mean"]) / before["test_loss_mean"],
        "before_accuracy_percent": 100 * before["test_accuracy_mean"],
        "after_accuracy_percent": 100 * after["test_accuracy_mean"],
        "accuracy_change_percentage_points": 100 * (after["test_accuracy_mean"] - before["test_accuracy_mean"]),
    })
with (OUT / "comparison_by_D.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
    writer.writeheader()
    writer.writerows(comparison)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
for use_log in [False, True]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for key, _, label, color in RUNS:
        rows = [data[key][D] for D in sorted(data[key])]
        D = [r["D"] for r in rows]
        for ax, metric, factor in [(axes[0], "test_loss", 1), (axes[1], "test_accuracy", 100)]:
            ax.errorbar(D, [factor*r[metric+"_mean"] for r in rows],
                        yerr=[factor*r[metric+"_std"] for r in rows],
                        color=color, marker="o", markersize=4, capsize=3,
                        linewidth=1.8, label=label, alpha=0.9)
    for ax in axes:
        ax.set_xlabel("Training images D (more data →)")
        ax.grid(True, which="both", alpha=0.18)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
        if use_log:
            ax.set_xscale("log")
            ax.set_xticks([200, 1000, 5000, 10000, 45000])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
        else:
            ax.set_xlim(0, 47000)
            ax.set_xticks([0, 10000, 20000, 30000, 40000])
    axes[0].set_ylabel("Test loss (lower is better)")
    axes[1].set_ylabel("Correct test predictions (%) — higher is better")
    axes[0].set_title("A. Prediction loss")
    axes[1].set_title("B. Classification accuracy")
    axes[1].set_ylim(25, 82)
    if use_log:
        axes[0].set_yscale("log")
        axes[0].set_yticks([0.8, 1, 1.5, 2, 3])
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:g}"))
        scales = "Loss: logarithmic X and Y. Accuracy: logarithmic X, ordinary linear Y."
    else:
        axes[0].set_ylim(0.6, 3.4)
        scales = "Both charts use ordinary linear X and Y axes."
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=3, frameon=False)
    fig.suptitle("CIFAR-100: historical manual recipes vs the post-HPO main sweep", fontsize=15, y=0.99)
    fig.text(0.5, 0.02, scales + "\nDots: means of 10 seeds. Bars: ±1 seed SD. Lines only connect measurements; no fitted curves shown.",
             ha="center", fontsize=10)
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.07, right=0.98, wspace=0.24)
    fig.savefig(OUT / ("comparison_original_scales.png" if use_log else "comparison_linear.png"), dpi=180)
    plt.close(fig)

print("Comparison CSV and figures regenerated from packaged summaries.")
