from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import load_config
from src.feature_cache import load_feature_cache, prepare_feature_cache
from src.hpo import aggregate_trials, config_key, select_best_by_D, select_global_config
from src.io_utils import atomic_json_dump
from src.probe import train_probe_trial
from src.sampling import build_nested_balanced_subsets, build_nested_stratified_subsets


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _number_token(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace("+", "").replace(".", "p")


def _save_numpy_array_atomic(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = np.load(path)
        if not np.array_equal(existing, array):
            raise ValueError(f"Existing subset disagrees with the current run: {path}")
        return

    fd, temporary_name = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.remove(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _build_subsets(
    cfg: dict[str, Any],
    original_indices: np.ndarray,
    original_targets: np.ndarray,
    seed: int,
) -> dict[int, np.ndarray]:
    sizes = list(map(int, cfg["data"]["dataset_sizes"]))
    sampling = str(cfg["data"].get("sampling", "balanced"))
    if sampling == "balanced":
        return build_nested_balanced_subsets(
            train_pool_indices=original_indices,
            targets=original_targets,
            dataset_sizes=sizes,
            seed=seed,
        )
    if sampling == "stratified_proportional":
        return build_nested_stratified_subsets(
            train_pool_indices=original_indices,
            targets=original_targets,
            dataset_sizes=sizes,
            seed=seed,
        )
    raise ValueError(f"Unsupported sampling strategy: {sampling}")


def _trial_id(
    learning_rate: float,
    weight_decay: float,
    schedule: str,
    batch_size: int,
    include_batch_size: bool,
) -> str:
    if not include_batch_size:
        return f"lr_{learning_rate:.0e}_wd_{weight_decay:.0e}_{schedule}"
    return (
        f"lr_{_number_token(learning_rate)}_"
        f"wd_{_number_token(weight_decay)}_{schedule}_bs_{batch_size}"
    )


def _configurations(cfg: dict[str, Any]) -> list[tuple[float, float, str, int]]:
    hpo_cfg = cfg["hpo"]
    return list(
        itertools.product(
            map(float, hpo_cfg["learning_rates"]),
            map(float, hpo_cfg["weight_decays"]),
            map(str, hpo_cfg["schedules"]),
            map(int, hpo_cfg.get("batch_sizes", [cfg["probe"]["batch_size"]])),
        )
    )


def _expected_count(cfg: dict[str, Any]) -> int:
    return (
        len(cfg["data"]["dataset_sizes"])
        * len(cfg["data"]["seeds"])
        * len(_configurations(cfg))
    )


def _metric_key(row: dict[str, Any]) -> tuple[int, int, float, float, str, int]:
    return (int(row["D"]), int(row["seed"]), *config_key(row))


def _expected_keys(cfg: dict[str, Any]) -> set[tuple[int, int, float, float, str, int]]:
    return {
        (int(D), int(seed), learning_rate, weight_decay, schedule, batch_size)
        for D in cfg["data"]["dataset_sizes"]
        for seed in cfg["data"]["seeds"]
        for learning_rate, weight_decay, schedule, batch_size in _configurations(cfg)
    }


def _ensure_config_snapshot(cfg: dict[str, Any], output_dir: Path) -> None:
    snapshot_path = output_dir / "config_snapshot.json"
    if snapshot_path.exists():
        with snapshot_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != cfg:
            raise ValueError(
                f"{output_dir} already contains results from a different configuration. "
                "Use a new experiment.output_dir instead of mixing experiments."
            )
        return
    atomic_json_dump(cfg, snapshot_path)


def _collect_metrics(output_dir: Path, default_batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = "*/D_*_seed_*/trial_metrics.json"
    for path in sorted((output_dir / "trials").glob(pattern)):
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        row.setdefault("batch_size", int(default_batch_size))
        rows.append(row)
    return rows


def _boundary_label(value: float | int, values: list[float] | list[int]) -> str:
    unique = sorted(set(values))
    if len(unique) == 1:
        return "fixed"
    if value == unique[0]:
        return "lower"
    if value == unique[-1]:
        return "upper"
    return "interior"


def _finalize_results(cfg: dict[str, Any], output_dir: Path) -> None:
    rows = _collect_metrics(output_dir, int(cfg["probe"]["batch_size"]))
    expected_keys = _expected_keys(cfg)
    found_keys = [_metric_key(row) for row in rows]
    unique_found = set(found_keys)
    duplicates = len(found_keys) - len(unique_found)
    missing = expected_keys - unique_found
    unexpected = unique_found - expected_keys

    completion = {
        "expected_trials": len(expected_keys),
        "completed_trials": len(unique_found & expected_keys),
        "missing_trials": len(missing),
        "unexpected_trials": len(unexpected),
        "duplicate_trials": duplicates,
        "complete": not missing and not unexpected and duplicates == 0,
    }
    atomic_json_dump(completion, output_dir / "completion.json")

    if rows:
        rows.sort(
            key=lambda row: (
                int(row["D"]),
                float(row["learning_rate"]),
                float(row["weight_decay"]),
                str(row["schedule"]),
                int(row["batch_size"]),
                int(row["seed"]),
            )
        )
        _write_csv(rows, output_dir / "hpo_trials_partial.csv")

    if not completion["complete"]:
        raise RuntimeError(
            "HPO is incomplete; final selections were not written. "
            f"Completed {completion['completed_trials']}/{completion['expected_trials']} "
            f"expected trials. See {output_dir / 'completion.json'}."
        )

    _write_csv(rows, output_dir / "hpo_trials.csv")
    aggregated = aggregate_trials(rows)
    _write_csv(aggregated, output_dir / "hpo_aggregated.csv")

    hpo_cfg = cfg["hpo"]
    best_by_D = select_best_by_D(aggregated)
    batch_sizes = list(
        map(int, hpo_cfg.get("batch_sizes", [cfg["probe"]["batch_size"]]))
    )

    for row in best_by_D:
        lr = float(row["learning_rate"])
        wd = float(row["weight_decay"])
        batch_size = int(row["batch_size"])
        row["lr_boundary"] = _boundary_label(
            lr, list(map(float, hpo_cfg["learning_rates"]))
        )
        row["weight_decay_boundary"] = _boundary_label(
            wd, list(map(float, hpo_cfg["weight_decays"]))
        )
        row["batch_size_boundary"] = _boundary_label(batch_size, batch_sizes)
        row["training_budget_warning"] = (
            float(row["fraction_best_near_step_limit"]) >= 0.5
        )

    _write_csv(best_by_D, output_dir / "hpo_best_by_D.csv")
    atomic_json_dump(
        {str(row["D"]): row for row in best_by_D},
        output_dir / "selected_by_D.json",
    )
    global_selection = select_global_config(aggregated)
    atomic_json_dump(global_selection, output_dir / "selected_global.json")

    print("\nHPO aggregation complete. Test features were never loaded.")
    print(f"Global selection: {global_selection}")
    for row in best_by_D:
        warnings = []
        for field in ["lr_boundary", "weight_decay_boundary", "batch_size_boundary"]:
            if row[field] not in {"interior", "fixed"}:
                warnings.append(f"{field}={row[field]}")
        if row["training_budget_warning"]:
            warnings.append("training_budget_warning=True")
        if warnings:
            print(f"WARNING D={row['D']}: {', '.join(warnings)}")
    print(f"Results: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune a frozen ViT-Tiny linear probe using validation loss only"
    )
    parser.add_argument("--config", default="configs/hpo_lr_stage1.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num-shards)")
    if args.aggregate_only and args.force:
        parser.error("--aggregate-only and --force cannot be used together")

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / args.config)
    configurations = _configurations(cfg)
    total_trials = _expected_count(cfg)
    shard_trials = sum(
        task_index % args.num_shards == args.shard_index
        for task_index in range(total_trials)
    )

    print(f"Configurations: {len(configurations)}")
    print(f"Dataset sizes: {len(cfg['data']['dataset_sizes'])}")
    print(f"Seeds: {len(cfg['data']['seeds'])}")
    print(f"Total trials: {total_trials}")
    print(f"Shard {args.shard_index}/{args.num_shards}: {shard_trials} assigned trials")
    if args.dry_run:
        return

    output_dir = project_root / cfg["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_config_snapshot(cfg, output_dir)
    if args.aggregate_only:
        _finalize_results(cfg, output_dir)
        return

    cache_dir = project_root / cfg["experiment"]["cache_dir"]
    if args.prepare_cache or not (cache_dir / "metadata.json").exists():
        prepare_feature_cache(cfg, project_root, force=False)

    # HPO deliberately cannot access the test tensors.
    tensors, cache_metadata = load_feature_cache(cache_dir, include_test=False)
    original_indices = tensors["train_original_indices"].numpy()
    original_targets = np.full(int(original_indices.max()) + 1, -1, dtype=np.int64)
    original_targets[original_indices] = tensors["train_labels"].numpy()
    original_to_cache_row = {int(idx): row for row, idx in enumerate(original_indices)}
    include_batch_size = "batch_sizes" in cfg["hpo"]
    task_index = 0

    for seed in map(int, cfg["data"]["seeds"]):
        subset_dir = output_dir / "subsets" / f"seed_{seed}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        subset_paths = {
            int(D): subset_dir / f"D_{int(D):05d}_original_indices.npy"
            for D in cfg["data"]["dataset_sizes"]
        }
        if all(path.exists() for path in subset_paths.values()):
            subsets = {D: np.load(path) for D, path in subset_paths.items()}
        else:
            subsets = _build_subsets(
                cfg=cfg,
                original_indices=original_indices,
                original_targets=original_targets,
                seed=seed,
            )

        for D, original_subset in subsets.items():
            _save_numpy_array_atomic(
                original_subset,
                subset_paths[int(D)],
            )
            cache_rows = torch.tensor(
                [original_to_cache_row[int(i)] for i in original_subset], dtype=torch.long
            )
            train_features = tensors["train_features"].index_select(0, cache_rows)
            train_labels = tensors["train_labels"].index_select(0, cache_rows)

            for learning_rate, weight_decay, schedule, batch_size in configurations:
                assigned = task_index % args.num_shards == args.shard_index
                task_index += 1
                if not assigned:
                    continue

                trial_id = _trial_id(
                    learning_rate,
                    weight_decay,
                    schedule,
                    batch_size,
                    include_batch_size,
                )
                run_dir = output_dir / "trials" / trial_id / f"D_{D:05d}_seed_{seed}"
                metrics_path = run_dir / "trial_metrics.json"
                if metrics_path.exists() and not args.force:
                    print(f"SKIP {trial_id} D={D} seed={seed}")
                    continue

                probe_cfg = {
                    **cfg["probe"],
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "schedule": schedule,
                    "batch_size": batch_size,
                }
                print(f"RUN {trial_id} D={D} seed={seed}")
                metrics = train_probe_trial(
                    train_features=train_features,
                    train_labels=train_labels,
                    val_features=tensors["val_features"],
                    val_labels=tensors["val_labels"],
                    probe_cfg=probe_cfg,
                    runtime_cfg=cfg["runtime"],
                    seed=seed,
                    run_dir=run_dir,
                    run_metadata={
                        "experiment": cfg["experiment"]["name"],
                        "model_name": cfg["model"]["name"],
                        "D": int(D),
                        "trial_id": trial_id,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "schedule": schedule,
                        "batch_size": batch_size,
                        "backbone_total_parameters": cache_metadata[
                            "backbone_total_parameters"
                        ],
                        "frozen_backbone": True,
                    },
                )
                print(f"  best_val_loss={metrics['best_val_loss']:.6f}")

    atomic_json_dump(
        {
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "assigned_trials": shard_trials,
            "finished": True,
        },
        output_dir / "shards" / f"shard_{args.shard_index:04d}_of_{args.num_shards:04d}.json",
    )

    if args.num_shards == 1:
        _finalize_results(cfg, output_dir)
    else:
        print(
            "Shard complete. After every shard finishes, run the same command with "
            "--aggregate-only to verify completeness and create final selections."
        )


if __name__ == "__main__":
    main()
