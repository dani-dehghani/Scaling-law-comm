"""Regenerate report figures from compact saved tables; no training or evaluation."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


def main() -> None:
    experiment = Path(__file__).resolve().parents[1]
    output = experiment / "reports" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    datasets = [("cifar100_main", "CIFAR-100", "official test"),
                ("places365_main", "Places365", "official validation used for final evaluation")]
    records = []
    for name, label, split in datasets:
        folder = experiment / "artifacts" / name
        aggregate = pd.read_csv(folder / "aggregated_by_D.csv").sort_values("D")
        runs = pd.read_csv(folder / "summary.csv")
        fit = json.loads((folder / "scaling_fit.json").read_text())
        assert aggregate.D.is_unique and len(aggregate) == runs.D.nunique(), "Mismatched size tables"
        assert (aggregate.n_seeds == 10).all(), "Report captions require ten seeds"
        assert np.isfinite(runs[["D", "examples_seen", "steps_ran"]].to_numpy()).all(), "Invalid exposure data"
        records.append((label, aggregate, runs, fit))
        figure = Figure(figsize=(11, 4.8), constrained_layout=True)
        left, right = figure.subplots(1, 2)
        d = aggregate.D.to_numpy(float)
        left.errorbar(d, aggregate.test_loss_mean, yerr=aggregate.test_loss_std,
                      fmt="o", capsize=3, label="Mean ± 1 seed SD", color="#176b87")
        dense = np.geomspace(d.min(), d.max(), 200)
        left.plot(dense, fit["L_inf"] + fit["b"] * dense ** -fit["beta"], color="#a95814",
                  label=f"Recorded fit: beta={fit['beta']:.4f}")
        left.set_ylabel("Final-evaluation cross-entropy (nats)")
        left.set_title("Loss and fitted curve")
        left.legend()
        right.errorbar(d, 100 * aggregate.test_accuracy_mean, yerr=100 * aggregate.test_accuracy_std,
                       fmt="o-", capsize=3, label="Mean ± 1 seed SD", color="#176b87")
        right.set_ylabel("Final-evaluation top-1 accuracy (%)")
        right.set_title("Accuracy across training sizes")
        right.legend()
        for axis in (left, right):
            axis.set_xscale("log")
            axis.set_xlabel("Distinct labeled training examples D (log scale)")
            axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
            axis.grid(alpha=0.18)
        figure.suptitle(f"{label}: {split}\nFrozen ViT-Tiny, 10 seeds per size; vertical axes are linear")
        figure.savefig(output / f"{name}.png", dpi=180)

    figure = Figure(figsize=(11, 4.3), constrained_layout=True)
    axes = figure.subplots(1, 2)
    for axis, (label, aggregate, _, fit) in zip(axes, records):
        d = aggregate.D.to_numpy(float)
        measured = aggregate.test_loss_mean.to_numpy(float)
        predicted = fit["L_inf"] + fit["b"] * d ** -fit["beta"]
        axis.axhline(0, color="#666666", linewidth=1)
        axis.plot(d, 100 * (measured - predicted) / measured, "o-", color="#176b87")
        axis.set_xscale("log")
        axis.set_xlabel("Distinct training examples D (log scale)")
        axis.set_ylabel("(Observed - fitted) / observed (%)")
        axis.set_title(label)
        axis.grid(alpha=0.18)
    figure.suptitle("In-sample residuals of the recorded loss fits; positive = fit below observations")
    figure.savefig(output / "fit_residuals.png", dpi=180)

    figure = Figure(figsize=(11, 4.3), constrained_layout=True)
    axes = figure.subplots(1, 2)
    for axis, (label, _, runs, _) in zip(axes, records):
        grouped = runs.groupby("D")[["steps_ran", "examples_seen"]].mean()
        axis.plot(grouped.index, grouped.examples_seen, "o-", color="#176b87")
        axis.plot(grouped.index, grouped.index, "--", color="#a95814", label="One pass through D")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Distinct training examples D")
        axis.set_ylabel("Mean training-example presentations")
        axis.set_title(label)
        axis.grid(alpha=0.18)
        axis.legend(["Recorded mean examples seen", "One pass through D"])
    figure.suptitle("Unique data and training exposure are different; both axes use log spacing")
    figure.savefig(output / "data_vs_exposure.png", dpi=180)
    print(f"Four report figures written to {output}")


if __name__ == "__main__":
    main()
