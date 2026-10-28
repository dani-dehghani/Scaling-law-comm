from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.io_utils import atomic_json_dump


REQUIRED_AGGREGATE_COLUMNS = {
    "D",
    "test_loss_mean",
    "test_loss_std",
    "test_accuracy_mean",
    "test_accuracy_std",
    "n_seeds",
}


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Run must have the form NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Run name cannot be empty")
    return name, Path(raw_path).expanduser()


def _column_prefix(name: str) -> str:
    prefix = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    if not prefix:
        raise ValueError(f"Cannot create a column prefix for run name {name!r}")
    return prefix


def _load_run(name: str, run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    aggregate_path = run_dir / "aggregated_by_D.csv"
    fit_path = run_dir / "scaling_fit.json"
    if not aggregate_path.exists() or not fit_path.exists():
        raise FileNotFoundError(
            f"{name}: expected {aggregate_path} and {fit_path}"
        )
    frame = pd.read_csv(aggregate_path)
    missing = REQUIRED_AGGREGATE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: aggregate is missing columns {sorted(missing)}")
    if frame["D"].duplicated().any():
        raise ValueError(f"{name}: aggregate contains duplicate D values")
    with fit_path.open("r", encoding="utf-8") as f:
        fit = json.load(f)
    for key in ["L_inf", "b", "beta", "rmse_log_loss"]:
        if key not in fit:
            raise ValueError(f"{name}: scaling fit is missing {key}")
    return frame.sort_values("D"), fit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare completed data-scaling sweeps and fitted scaling laws"
    )
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        required=True,
        help="Run specification NAME=PATH; the first run is the reference",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write comparison tables without importing matplotlib",
    )
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("At least two --run arguments are required")

    names = [name for name, _ in args.run]
    if len(names) != len(set(names)):
        parser.error("Run names must be unique")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded: list[tuple[str, pd.DataFrame, dict[str, Any]]] = []
    combined: pd.DataFrame | None = None

    for name, run_dir in args.run:
        frame, fit = _load_run(name, run_dir)
        loaded.append((name, frame, fit))
        prefix = _column_prefix(name)
        renamed = frame.rename(
            columns={column: f"{prefix}_{column}" for column in frame.columns if column != "D"}
        )
        combined = renamed if combined is None else combined.merge(renamed, on="D", how="outer")

    assert combined is not None
    reference_prefix = _column_prefix(loaded[0][0])
    for name, _, _ in loaded[1:]:
        prefix = _column_prefix(name)
        combined[f"{prefix}_loss_minus_reference"] = (
            combined[f"{prefix}_test_loss_mean"]
            - combined[f"{reference_prefix}_test_loss_mean"]
        )
        combined[f"{prefix}_accuracy_minus_reference"] = (
            combined[f"{prefix}_test_accuracy_mean"]
            - combined[f"{reference_prefix}_test_accuracy_mean"]
        )
    combined.sort_values("D").to_csv(output_dir / "comparison_by_D.csv", index=False)

    fit_summary = {
        name: {
            "source": str(run_dir),
            **fit,
        }
        for (name, run_dir), (_, _, fit) in zip(args.run, loaded)
    }
    atomic_json_dump(fit_summary, output_dir / "scaling_comparison.json")

    if not args.no_plots:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for name, frame, fit in loaded:
            ax.errorbar(
                frame["D"],
                frame["test_loss_mean"],
                yerr=frame["test_loss_std"].fillna(0),
                marker="o",
                capsize=3,
                label=f"{name} (beta={float(fit['beta']):.3f})",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training-set size D")
        ax.set_ylabel("Test cross-entropy loss")
        ax.set_title("Data-scaling sweep comparison")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "loss_comparison.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for name, frame, _ in loaded:
            ax.errorbar(
                frame["D"],
                frame["test_accuracy_mean"],
                yerr=frame["test_accuracy_std"].fillna(0),
                marker="o",
                capsize=3,
                label=name,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Training-set size D")
        ax.set_ylabel("Test accuracy")
        ax.set_title("Accuracy comparison")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "accuracy_comparison.png", dpi=180)
        plt.close(fig)

    print(f"Reference run: {loaded[0][0]}")
    for name, _, fit in loaded:
        print(
            f"{name}: beta={float(fit['beta']):.6f}, "
            f"L_inf={float(fit['L_inf']):.6f}, "
            f"rmse_log_loss={float(fit['rmse_log_loss']):.6f}"
        )
    print(f"Comparison outputs: {output_dir}")


if __name__ == "__main__":
    main()
