from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_boundary(value: float, values: list[float]) -> bool:
    ordered = sorted(set(map(float, values)))
    return len(ordered) > 1 and value in {ordered[0], ordered[-1]}


def assess_hpo(results_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Return the selected recipe and every failed promotion condition."""
    required = [
        "completion.json",
        "selected_global.json",
        "config_snapshot.json",
        "hpo_aggregated.csv",
    ]
    missing = [name for name in required if not (results_dir / name).exists()]
    if missing:
        return {}, [f"Missing result files: {missing}"]

    completion = _load_json(results_dir / "completion.json")
    selection = _load_json(results_dir / "selected_global.json")
    snapshot = _load_json(results_dir / "config_snapshot.json")
    failures: list[str] = []

    if not bool(completion.get("complete", False)):
        failures.append("HPO completion.json does not report complete=true")
    if int(completion.get("completed_trials", -1)) != int(
        completion.get("expected_trials", -2)
    ):
        failures.append("Completed trial count does not equal expected trial count")
    for key in ["missing_trials", "unexpected_trials", "duplicate_trials"]:
        if int(completion.get(key, -1)) != 0:
            failures.append(f"completion.json reports {key}={completion.get(key)}")

    hpo = snapshot["hpo"]
    lr = float(selection["learning_rate"])
    wd = float(selection["weight_decay"])
    if _is_boundary(lr, list(map(float, hpo["learning_rates"]))):
        failures.append(f"Selected learning rate {lr:g} is a search boundary")
    if _is_boundary(wd, list(map(float, hpo["weight_decays"]))):
        failures.append(f"Selected weight decay {wd:g} is a search boundary")

    selected_rows: list[dict[str, str]] = []
    with (results_dir / "hpo_aggregated.csv").open(
        "r", encoding="utf-8", newline=""
    ) as f:
        for row in csv.DictReader(f):
            if (
                math.isclose(float(row["learning_rate"]), lr)
                and math.isclose(float(row["weight_decay"]), wd)
                and str(row["schedule"]) == str(selection["schedule"])
                and int(row["batch_size"]) == int(selection["batch_size"])
            ):
                selected_rows.append(row)

    expected_D = int(selection["dataset_sizes_compared"])
    if len(selected_rows) != expected_D:
        failures.append(
            f"Found {len(selected_rows)} selected-global aggregate rows; expected {expected_D}"
        )
    for row in selected_rows:
        D = int(row["D"])
        numeric_fields = [
            "val_loss_mean",
            "val_loss_std",
            "val_accuracy_mean",
            "best_step_mean",
            "steps_ran_mean",
        ]
        if any(not math.isfinite(float(row[field])) for field in numeric_fields):
            failures.append(f"D={D} contains a non-finite selected-global metric")
        if float(row["fraction_best_near_step_limit"]) >= 0.5:
            failures.append(f"D={D} has a selected-global training-budget warning")

    return selection, failures


def _write_promoted_config(
    path: Path, selection: dict[str, Any], *, smoke_only: bool
) -> None:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    experiment = cfg["experiment"]
    experiment["protocol_status"] = "smoke_only" if smoke_only else "approved"
    experiment.pop("protocol_block_reason", None)
    probe = cfg["probe"]
    probe["learning_rate"] = float(selection["learning_rate"])
    probe["weight_decay"] = float(selection["weight_decay"])
    probe["schedule"] = str(selection["schedule"])
    probe["batch_size"] = int(selection["batch_size"])

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Promote an accepted Places365 HPO recipe into final sweep configs"
    )
    parser.add_argument(
        "--hpo-results",
        default="results/hpo_places365_wd_boundary_8d_10seeds_30k",
    )
    parser.add_argument(
        "--final-config", default="configs/data_sweep_places365_final.yaml"
    )
    parser.add_argument(
        "--smoke-config", default="configs/data_sweep_places365_final_smoke.yaml"
    )
    args = parser.parse_args()

    results_dir = project_root / args.hpo_results
    selection, failures = assess_hpo(results_dir)
    if failures:
        print("Promotion refused. The HPO acceptance gate did not pass:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    final_path = project_root / args.final_config
    smoke_path = project_root / args.smoke_config
    _write_promoted_config(final_path, selection, smoke_only=False)
    _write_promoted_config(smoke_path, selection, smoke_only=True)
    print(f"Promoted recipe: {selection}")
    print(f"Approved final config: {final_path}")
    print(f"Updated train-only smoke config: {smoke_path}")


if __name__ == "__main__":
    main()
