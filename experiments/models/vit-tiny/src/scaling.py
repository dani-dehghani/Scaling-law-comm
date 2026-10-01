from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def scaling_law(D: np.ndarray, L_inf: float, b: float, beta: float) -> np.ndarray:
    return L_inf + b * np.power(D, -beta)


def _fit_power_law(D: np.ndarray, loss: np.ndarray) -> dict[str, float | int]:
    from scipy.optimize import least_squares

    D = np.asarray(D, dtype=float)
    loss = np.asarray(loss, dtype=float)
    if D.ndim != 1 or loss.ndim != 1 or len(D) != len(loss):
        raise ValueError("D and loss must be one-dimensional arrays of equal length")
    if len(D) < 4 or len(np.unique(D)) < 4:
        raise ValueError("At least four distinct dataset sizes are required")
    if not np.all(np.isfinite(D)) or not np.all(np.isfinite(loss)):
        raise ValueError("D and loss must contain only finite values")
    if np.any(D <= 0) or np.any(loss <= 0):
        raise ValueError("D and loss must be strictly positive")

    order = np.argsort(D)
    D = D[order]
    loss = loss[order]

    # Positive parameters via exponentiation. Residuals are in log-loss space,
    # matching standard scaling-law practice and reducing domination by large losses.
    def residuals(p: np.ndarray) -> np.ndarray:
        L_inf, b, beta = np.exp(p)
        pred = scaling_law(D, L_inf, b, beta)
        return np.log(pred) - np.log(loss)

    fits = []
    for floor_fraction in [0.1, 0.5, 0.9]:
        for initial_beta in [0.1, 0.3, 0.6, 1.0]:
            initial_floor = max(loss.min() * floor_fraction, 1e-8)
            initial_b = max(
                (loss.max() - initial_floor) * np.power(D.min(), initial_beta),
                1e-8,
            )
            initial = np.log([initial_floor, initial_b, initial_beta])
            candidate = least_squares(
                residuals,
                initial,
                loss="huber",
                f_scale=0.05,
                max_nfev=10000,
            )
            if candidate.success and np.all(np.isfinite(candidate.x)):
                fits.append(candidate)
    if not fits:
        raise RuntimeError("Scaling-law optimization did not converge")

    fit = min(fits, key=lambda candidate: candidate.cost)
    L_inf, b, beta = np.exp(fit.x)
    pred = scaling_law(D, L_inf, b, beta)
    rmse_log = float(np.sqrt(np.mean((np.log(pred) - np.log(loss)) ** 2)))
    return {
        "L_inf": float(L_inf),
        "b": float(b),
        "beta": float(beta),
        "rmse_log_loss": rmse_log,
        "n_points": int(len(D)),
    }


def analyze_results(
    summary_csv: Path,
    output_dir: Path,
    experiment_label: str = "Frozen linear probe",
) -> dict[str, float | int]:
    import matplotlib.pyplot as plt
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError("summary.csv is empty")
    required_columns = {"D", "seed", "test_loss", "test_accuracy"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"summary.csv is missing columns: {sorted(missing_columns)}")
    numeric = df[list(required_columns)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("summary.csv contains non-numeric or missing metric values")
    if (numeric["D"] <= 0).any() or (numeric["test_loss"] <= 0).any():
        raise ValueError("D and test_loss must be strictly positive")
    df[list(required_columns)] = numeric

    agg = (
        df.groupby("D", as_index=False)
        .agg(
            test_loss_mean=("test_loss", "mean"),
            test_loss_std=("test_loss", "std"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("D")
    )
    agg.to_csv(output_dir / "aggregated_by_D.csv", index=False)

    fit = _fit_power_law(agg["D"].to_numpy(), agg["test_loss_mean"].to_numpy())
    with (output_dir / "scaling_fit.json").open("w", encoding="utf-8") as f:
        json.dump(fit, f, indent=2)
        f.write("\n")

    D = agg["D"].to_numpy(dtype=float)
    dense_D = np.geomspace(D.min(), D.max(), 300)
    fitted = scaling_law(dense_D, fit["L_inf"], fit["b"], fit["beta"])

    fig, ax = plt.subplots(figsize=(7, 5))
    yerr = agg["test_loss_std"].fillna(0).to_numpy()
    ax.errorbar(D, agg["test_loss_mean"], yerr=yerr, fmt="o", capsize=3, label="Measured")
    ax.plot(dense_D, fitted, label=f"Fit: beta={fit['beta']:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training-set size D")
    ax.set_ylabel("Test cross-entropy loss")
    ax.set_title(f"{experiment_label}: loss vs dataset size")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_vs_D.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    yerr = agg["test_accuracy_std"].fillna(0).to_numpy()
    ax.errorbar(D, agg["test_accuracy_mean"], yerr=yerr, fmt="o-", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("Training-set size D")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"{experiment_label}: accuracy vs dataset size")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_D.png", dpi=180)
    plt.close(fig)

    return fit
