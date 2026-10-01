from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import load_config
from src.feature_cache import load_feature_cache, prepare_feature_cache
from src.io_utils import atomic_json_dump
from src.probe import evaluate_probe_checkpoint, train_probe_trial
from src.sampling import build_nested_balanced_subsets, build_nested_stratified_subsets
from src.scaling import analyze_results
from src.sweep import budget_warning


def _ensure_config_snapshot(cfg: dict[str, Any], output_dir: Path) -> None:
    snapshot_path = output_dir / "config_snapshot.json"
    if snapshot_path.exists():
        with snapshot_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != cfg:
            raise RuntimeError(
                "The output directory already contains results from a different "
                "configuration. Choose a new experiment.output_dir."
            )
        return
    atomic_json_dump(cfg, snapshot_path)


def _enforce_protocol_status(
    cfg: dict[str, Any], *, dry_run: bool, train_only: bool
) -> None:
    """Prevent provisional configs from opening the final evaluation split."""
    experiment_cfg = cfg["experiment"]
    status = str(experiment_cfg.get("protocol_status", "approved"))
    reason = str(experiment_cfg.get("protocol_block_reason", "")).strip()
    print(f"Protocol status: {status}")
    if reason:
        print(f"Protocol note: {reason}")
    if dry_run or status == "approved":
        return
    if status == "smoke_only":
        if train_only:
            return
        raise RuntimeError(
            "This smoke configuration may only run with --train-only. "
            "The locked final evaluation split remains inaccessible."
        )
    raise RuntimeError(
        "This sweep configuration is blocked pending HPO acceptance. "
        "Dry-run is allowed, but training and final evaluation are disabled."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen ViT-Tiny dataset-size sweep")
    parser.add_argument("--config", default="configs/data_sweep_vit_tiny.yaml")
    parser.add_argument("--force", action="store_true", help="Rerun completed D/seed configurations")
    parser.add_argument(
        "--prepare-cache",
        action="store_true",
        help="Create the feature cache first if needed",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train and validate every run, but keep the test split locked",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and print the planned run count",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    cfg = load_config(project_root / args.config)
    expected_runs = len(cfg["data"]["dataset_sizes"]) * len(cfg["data"]["seeds"])
    print(f"Dataset sizes: {len(cfg['data']['dataset_sizes'])}")
    print(f"Seeds: {len(cfg['data']['seeds'])}")
    print(f"Planned runs: {expected_runs}")
    _enforce_protocol_status(cfg, dry_run=args.dry_run, train_only=args.train_only)
    if args.dry_run:
        return

    cache_dir = project_root / cfg["experiment"]["cache_dir"]
    if args.prepare_cache or not (cache_dir / "metadata.json").exists():
        prepare_feature_cache(cfg, project_root, force=False)

    # Keep test tensors inaccessible until every training run is complete and
    # the convergence gate below has passed.
    tensors, cache_metadata = load_feature_cache(cache_dir, include_test=False)
    output_dir = project_root / cfg["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_config_snapshot(cfg, output_dir)

    # Map cached rows back to the source training split so subset membership is
    # reproducible independently of feature-cache row order.
    original_indices = tensors["train_original_indices"].numpy()
    train_labels_np = tensors["train_labels"].numpy()
    original_targets = np.full(int(original_indices.max()) + 1, -1, dtype=np.int64)
    original_targets[original_indices] = train_labels_np

    original_to_cache_row = {int(idx): row for row, idx in enumerate(original_indices)}
    all_trial_metrics: list[dict[str, Any]] = []

    for seed in map(int, cfg["data"]["seeds"]):
        sampling = str(cfg["data"].get("sampling", "balanced"))
        builder = (
            build_nested_balanced_subsets
            if sampling == "balanced"
            else build_nested_stratified_subsets
        )
        subsets = builder(
            train_pool_indices=original_indices,
            targets=original_targets,
            dataset_sizes=list(map(int, cfg["data"]["dataset_sizes"])),
            seed=seed,
        )

        seed_dir = output_dir / "subsets" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for D, original_subset in subsets.items():
            run_dir = output_dir / "runs" / f"D_{D:05d}_seed_{seed}"
            trial_metrics_path = run_dir / "trial_metrics.json"
            subset_path = seed_dir / f"D_{D:05d}_original_indices.npy"
            np.save(subset_path, original_subset)

            if trial_metrics_path.exists() and not args.force:
                with trial_metrics_path.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
                all_trial_metrics.append(metrics)
                print(f"SKIP trained D={D}, seed={seed}")
                continue

            cache_rows = torch.tensor(
                [original_to_cache_row[int(i)] for i in original_subset], dtype=torch.long
            )
            train_features = tensors["train_features"].index_select(0, cache_rows)
            train_labels = tensors["train_labels"].index_select(0, cache_rows)

            run_metadata = {
                "experiment": cfg["experiment"]["name"],
                "model_name": cfg["model"]["name"],
                "D": int(D),
                "backbone_total_parameters": cache_metadata["backbone_total_parameters"],
                "frozen_backbone": True,
                "learning_rate": float(cfg["probe"]["learning_rate"]),
                "weight_decay": float(cfg["probe"]["weight_decay"]),
                "schedule": str(cfg["probe"].get("schedule", "cosine")),
                "batch_size": int(cfg["probe"]["batch_size"]),
                "channel": "clean/full-width/noiseless (not yet inserted in this D-only sweep)",
            }
            print(f"RUN D={D}, seed={seed}")
            metrics = train_probe_trial(
                train_features=train_features,
                train_labels=train_labels,
                val_features=tensors["val_features"],
                val_labels=tensors["val_labels"],
                probe_cfg=cfg["probe"],
                runtime_cfg=cfg["runtime"],
                seed=seed,
                run_dir=run_dir,
                run_metadata=run_metadata,
            )
            all_trial_metrics.append(metrics)
            print(f"  best_val_loss={metrics['best_val_loss']:.6f}")

    unique_keys = {(int(row["D"]), int(row["seed"])) for row in all_trial_metrics}
    budget_warnings = [
        {"D": int(row["D"]), "seed": int(row["seed"]), "best_step": int(row["best_step"])}
        for row in all_trial_metrics
        if budget_warning(row)
    ]
    training_complete = len(unique_keys) == expected_runs
    completion = {
        "expected_runs": expected_runs,
        "trained_runs": len(unique_keys),
        "training_complete": training_complete,
        "budget_warning_runs": budget_warnings,
        "test_evaluation_complete": False,
    }
    atomic_json_dump(completion, output_dir / "completion.json")
    if not training_complete:
        raise RuntimeError(
            f"Training is incomplete: found {len(unique_keys)}/{expected_runs} unique runs."
        )
    if args.train_only:
        print("Training complete. Test features remain locked because --train-only was used.")
        print(f"Convergence warnings: {len(budget_warnings)}")
        return
    if budget_warnings:
        raise RuntimeError(
            "Test features remain locked because one or more runs selected a checkpoint "
            "near the step limit. Increase the shared max_steps budget and rerun. See "
            f"{output_dir / 'completion.json'}."
        )

    # The protocol is now complete and convergence-checked, so final test
    # evaluation can happen without feeding test results back into selection.
    test_features = torch.load(
        cache_dir / "test_features.pt", map_location="cpu", weights_only=True
    )
    test_labels = torch.load(
        cache_dir / "test_labels.pt", map_location="cpu", weights_only=True
    )
    all_metrics: list[dict[str, Any]] = []
    for trial_metrics in sorted(all_trial_metrics, key=lambda row: (row["D"], row["seed"])):
        D = int(trial_metrics["D"])
        seed = int(trial_metrics["seed"])
        run_dir = output_dir / "runs" / f"D_{D:05d}_seed_{seed}"
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists() and not args.force:
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            result = evaluate_probe_checkpoint(
                checkpoint_path=run_dir / "linear_probe.pt",
                features=test_features,
                labels=test_labels,
                batch_size=int(cfg["probe"]["batch_size"]),
                runtime_cfg=cfg["runtime"],
            )
            metrics = {
                **trial_metrics,
                "test_loss": float(result.loss),
                "test_accuracy": float(result.accuracy),
            }
            atomic_json_dump(metrics, metrics_path)
        all_metrics.append(metrics)
        print(
            f"TEST D={D}, seed={seed}: loss={metrics['test_loss']:.4f}, "
            f"accuracy={100 * metrics['test_accuracy']:.2f}%"
        )

    import pandas as pd

    summary = pd.DataFrame(all_metrics).sort_values(["D", "seed"])
    summary.to_csv(output_dir / "summary.csv", index=False)
    label = str(cfg["experiment"]["name"]).replace("_", " ")
    fit = analyze_results(output_dir / "summary.csv", output_dir, label)
    completion["test_evaluation_complete"] = True
    completion["evaluated_runs"] = len(all_metrics)
    atomic_json_dump(completion, output_dir / "completion.json")
    print("\nSweep complete.")
    print(f"Estimated beta = {fit['beta']:.4f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
