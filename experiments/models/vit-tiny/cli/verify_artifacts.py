"""Check compact recorded evidence without training or opening final-evaluation data."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_sweep(folder: Path, refit: bool = False) -> dict:
    cfg = read_json(folder / "config_snapshot.json")
    rows = read_csv(folder / "summary.csv")
    identities = [(int(row["D"]), int(row["seed"])) for row in rows]
    expected = set(itertools.product(cfg["data"]["dataset_sizes"], cfg["data"]["seeds"]))
    require(len(identities) == len(set(identities)), f"{folder.name}: duplicate D/seed")
    require(set(identities) == expected, f"{folder.name}: incomplete or unexpected grid")
    for row in rows:
        loss, accuracy = float(row["test_loss"]), float(row["test_accuracy"])
        require(math.isfinite(loss) and loss > 0, f"{folder.name}: invalid loss")
        require(math.isfinite(accuracy) and 0 <= accuracy <= 1, f"{folder.name}: invalid accuracy")
        for field in ("learning_rate", "weight_decay", "batch_size"):
            if field in row:
                require(float(row[field]) == float(cfg["probe"][field]), f"{folder.name}: mixed {field}")
    stored = read_csv(folder / "aggregated_by_D.csv")
    require(len(stored) == len(cfg["data"]["dataset_sizes"]), f"{folder.name}: aggregate size count")
    require(len({int(row["D"]) for row in stored}) == len(stored), f"{folder.name}: duplicate aggregate D")
    differences = []
    for aggregate in stored:
        group = [row for row in rows if int(row["D"]) == int(aggregate["D"])]
        require(bool(group), f"{folder.name}: aggregate D absent from runs")
        require(int(aggregate["n_seeds"]) == len(group), f"{folder.name}: wrong seed count")
        for metric in ("test_loss", "test_accuracy"):
            values = [float(row[metric]) for row in group]
            for suffix, value in (("mean", mean(values)), ("std", stdev(values) if len(values) > 1 else 0.0)):
                gap = abs(float(aggregate[f"{metric}_{suffix}"]) - value)
                require(gap < 1e-12, f"{folder.name}: incorrect {metric}_{suffix}")
                differences.append(gap)
    completion_file = folder / "completion.json"
    if completion_file.exists():
        completion = read_json(completion_file)
        require(completion.get("training_complete") is True and completion.get("test_evaluation_complete") is True,
                f"{folder.name}: final sweep not marked complete")
        require(completion["evaluated_runs"] == len(rows) == completion["expected_runs"], f"{folder.name}: completion count")
    fit = read_json(folder / "scaling_fit.json")
    require(all(math.isfinite(float(fit[k])) and float(fit[k]) > 0 for k in ("L_inf", "b", "beta")),
            f"{folder.name}: invalid fit parameters")
    result = {"runs": len(rows), "sizes": len(stored), "max_aggregate_difference": max(differences),
              "saved_beta": fit["beta"], "completion_record": completion_file.exists()}
    if refit:
        import numpy as np
        from src.scaling import _fit_power_law
        recalculated = _fit_power_law(np.array([float(r["D"]) for r in stored]),
                                     np.array([float(r["test_loss_mean"]) for r in stored]))
        for field in ("L_inf", "b", "beta"):
            require(math.isclose(recalculated[field], fit[field], rel_tol=1e-5, abs_tol=1e-7),
                    f"{folder.name}: saved fit differs in {field}")
        result["refitted_beta"] = recalculated["beta"]
    return result


def check_hpo(folder: Path) -> dict:
    cfg = read_json(folder / "config_snapshot.json")
    rows = read_csv(folder / "hpo_trials.csv")
    fields = ("learning_rate", "weight_decay", "schedule", "batch_size")

    def recipe(row: dict) -> tuple:
        return float(row["learning_rate"]), float(row["weight_decay"]), row["schedule"], int(row.get("batch_size", cfg["probe"]["batch_size"]))

    hpo = cfg["hpo"]
    candidates = list(itertools.product(map(float, hpo["learning_rates"]), map(float, hpo["weight_decays"]),
                                       hpo["schedules"], hpo.get("batch_sizes", [cfg["probe"]["batch_size"]])))
    expected = {(int(d), int(s), *c) for d in cfg["data"]["dataset_sizes"] for s in cfg["data"]["seeds"] for c in candidates}
    keys = [(int(r["D"]), int(r["seed"]), *recipe(r)) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, f"{folder.name}: incomplete/duplicate HPO grid")
    grouped = {}
    for row in rows:
        loss, accuracy = float(row["best_val_loss"]), float(row["best_val_accuracy"])
        require(math.isfinite(loss) and loss > 0 and math.isfinite(accuracy) and 0 <= accuracy <= 1,
                f"{folder.name}: invalid validation metric")
        grouped.setdefault((int(row["D"]), *recipe(row)), []).append((loss, accuracy))
    table = read_csv(folder / "hpo_aggregated.csv")
    aggregate_keys = [(int(r["D"]), *recipe(r)) for r in table]
    require(len(aggregate_keys) == len(set(aggregate_keys)) and set(aggregate_keys) == set(grouped),
            f"{folder.name}: aggregate grid mismatch")
    means = {}
    max_gap = 0.0
    for row in table:
        key = (int(row["D"]), *recipe(row))
        values = grouped[key]
        require(int(row["n_seeds"]) == len(values), f"{folder.name}: aggregate seed count")
        losses = [v[0] for v in values]
        calculated = {"val_loss_mean": mean(losses), "val_loss_std": stdev(losses),
                      "val_accuracy_mean": mean(v[1] for v in values)}
        for field, value in calculated.items():
            gap = abs(float(row[field]) - value)
            require(gap < 1e-12, f"{folder.name}: aggregate {field} disagrees")
            max_gap = max(max_gap, gap)
        means[key] = calculated["val_loss_mean"]
    best = {int(d): min(v for k, v in means.items() if k[0] == int(d)) for d in cfg["data"]["dataset_sizes"]}
    regret = {c: mean(means[(int(d), *c)] / best[int(d)] - 1 for d in cfg["data"]["dataset_sizes"]) for c in candidates}
    winner = min(candidates, key=lambda c: (regret[c], c))
    selected = read_json(folder / "selected_global.json")
    require(recipe(selected) == winner, f"{folder.name}: wrong global selection")
    require(math.isclose(selected["mean_relative_regret"], regret[winner], abs_tol=1e-12, rel_tol=0),
            f"{folder.name}: wrong relative regret")
    completion = read_json(folder / "completion.json")
    require(completion.get("complete") is True and completion["completed_trials"] == len(rows) == completion["expected_trials"],
            f"{folder.name}: completion mismatch")
    require(all(completion[k] == 0 for k in ("missing_trials", "unexpected_trials", "duplicate_trials")),
            f"{folder.name}: completion reports grid defects")
    return {"trials": len(rows), "aggregate_cells": len(table), "max_aggregate_difference": max_gap,
            "selected": dict(zip(fields, winner)), "mean_relative_regret": regret[winner]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit", action="store_true", help="Recalculate saved fits with the repository fitter; requires SciPy")
    parser.add_argument("--output", type=Path, help="Optional verification JSON outside artifacts")
    args = parser.parse_args()
    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    if args.output:
        require(not args.output.resolve().is_relative_to(artifacts.resolve()), "Verification output must not replace original artifacts")
    sweeps = ["cifar100_main", "places365_main", "baselines/formal_lr_001", "baselines/formal_lr_003", "baselines/initial_formal_v1"]
    searches = ["cifar100_core_hpo", "places365_global_hpo", "places365_refinement_hpo", "places365_boundary_hpo"]
    result = {"sweeps": {n: check_sweep(artifacts / n, args.refit) for n in sweeps},
              "hpo": {n: check_hpo(artifacts / n) for n in searches},
              "scope": "Recorded tables and fit consistency only; no training, inference, image/cache audit, or historical chronology proof."}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
