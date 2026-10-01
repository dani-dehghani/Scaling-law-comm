from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .io_utils import atomic_json_dump
from .reproducibility import environment_metadata, seed_everything, select_device


@dataclass
class EvalResult:
    loss: float
    accuracy: float


def _validate_supervised_split(
    features: torch.Tensor, labels: torch.Tensor, name: str
) -> None:
    if features.ndim != 2:
        raise ValueError(f"{name}_features must be a two-dimensional tensor")
    if labels.ndim != 1:
        raise ValueError(f"{name}_labels must be a one-dimensional tensor")
    if len(features) == 0 or len(labels) == 0:
        raise ValueError(f"{name} split must not be empty")
    if len(features) != len(labels):
        raise ValueError(f"{name} features and labels have different lengths")
    if not torch.isfinite(features).all():
        raise ValueError(f"{name}_features contains NaN or infinity")
    if torch.any(labels < 0):
        raise ValueError(f"{name}_labels contains a negative class id")


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        batch_n = y.numel()
        total_loss += float(loss.item()) * batch_n
        correct += int((logits.argmax(dim=1) == y).sum().item())
        n += batch_n
    if n == 0:
        raise ValueError("Cannot evaluate an empty data loader")
    return EvalResult(loss=total_loss / n, accuracy=correct / n)


def train_probe_trial(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    probe_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    seed: int,
    run_dir: Path,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    trial_started_at = time.perf_counter()
    _validate_supervised_split(train_features, train_labels, "train")
    _validate_supervised_split(val_features, val_labels, "val")
    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError("Train and validation feature dimensions differ")
    train_classes = set(map(int, torch.unique(train_labels).tolist()))
    val_classes = set(map(int, torch.unique(val_labels).tolist()))
    missing_classes = val_classes - train_classes
    if missing_classes:
        raise ValueError(
            f"Training subset is missing validation classes: {sorted(missing_classes)}"
        )

    seed_everything(seed, deterministic=bool(runtime_cfg["deterministic"]))
    device = select_device(runtime_cfg["device"])

    feature_dim = int(train_features.shape[1])
    num_classes = int(
        torch.maximum(torch.max(train_labels), torch.max(val_labels)).item() + 1
    )
    model = nn.Linear(feature_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    learning_rate = float(probe_cfg["learning_rate"])
    weight_decay = float(probe_cfg["weight_decay"])
    batch_size = int(probe_cfg["batch_size"])
    min_delta = float(probe_cfg["min_delta"])
    if learning_rate <= 0 or weight_decay < 0 or batch_size <= 0 or min_delta < 0:
        raise ValueError("Invalid probe learning rate, weight decay, batch size, or min_delta")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    step_mode = "max_steps" in probe_cfg or "max_examples" in probe_cfg
    schedule = str(probe_cfg.get("schedule", "cosine"))
    if schedule not in {"constant", "cosine"}:
        raise ValueError("probe.schedule must be 'constant' or 'cosine'")

    if step_mode:
        if "max_examples" in probe_cfg:
            max_examples = int(probe_cfg["max_examples"])
            eval_every_examples = int(probe_cfg["eval_every_examples"])
            steps_per_epoch = math.ceil(len(train_features) / batch_size)
            epochs_budget = math.ceil(max_examples / len(train_features))
            eval_every_epochs = max(
                1, math.ceil(eval_every_examples / len(train_features))
            )
            max_steps = epochs_budget * steps_per_epoch
            eval_every_steps = eval_every_epochs * steps_per_epoch
        else:
            max_examples = None
            eval_every_examples = None
            max_steps = int(probe_cfg["max_steps"])
            eval_every_steps = int(probe_cfg["eval_every_steps"])
        patience_evals = int(probe_cfg["patience_evals"])
        if max_steps <= 0 or eval_every_steps <= 0 or patience_evals <= 0:
            raise ValueError(
                "max_steps, eval_every_steps, and patience_evals must be positive"
            )
        schedule_length = max_steps
    else:
        max_epochs = int(probe_cfg["epochs"])
        if max_epochs <= 0:
            raise ValueError("probe.epochs must be positive")
        patience_evals = int(probe_cfg["patience"])
        schedule_length = max_epochs

    scheduler = None
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=schedule_length
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(val_features, val_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_state = None
    best_epoch = 0
    best_step = 0
    stale_evals = 0
    global_step = 0
    examples_seen = 0
    history: list[dict[str, float | int]] = []

    def record_validation(
        train_loss: float, epoch: int, learning_rate_used: float
    ) -> bool:
        nonlocal best_val_loss, best_val_accuracy, best_state
        nonlocal best_epoch, best_step, stale_evals

        val_result = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "step": global_step,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_result.loss,
                "val_accuracy": val_result.accuracy,
                "learning_rate": learning_rate_used,
            }
        )

        improvement = best_val_loss - val_result.loss
        if improvement > min_delta:
            best_val_loss = val_result.loss
            best_val_accuracy = val_result.accuracy
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_step = global_step
            stale_evals = 0
        else:
            stale_evals += 1
        return stale_evals >= patience_evals

    if step_mode:
        stop = False
        interval_loss = 0.0
        interval_examples = 0
        while global_step < max_steps and not stop:
            for x, y in train_loader:
                model.train()
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                learning_rate_used = float(optimizer.param_groups[0]["lr"])
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                batch_n = y.numel()
                examples_seen += batch_n
                interval_loss += float(loss.item()) * batch_n
                interval_examples += batch_n
                global_step += 1

                should_evaluate = (
                    global_step % eval_every_steps == 0 or global_step == max_steps
                )
                if should_evaluate:
                    epoch = int(np.ceil(global_step / len(train_loader)))
                    stop = record_validation(
                        train_loss=interval_loss / interval_examples,
                        epoch=epoch,
                        learning_rate_used=learning_rate_used,
                    )
                    interval_loss = 0.0
                    interval_examples = 0

                if stop or global_step >= max_steps:
                    break
    else:
        for epoch in range(1, max_epochs + 1):
            model.train()
            total_train_loss = 0.0
            n_train = 0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                learning_rate_used = float(optimizer.param_groups[0]["lr"])
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                global_step += 1

                batch_n = y.numel()
                examples_seen += batch_n
                total_train_loss += float(loss.item()) * batch_n
                n_train += batch_n

            stop = record_validation(
                train_loss=total_train_loss / n_train,
                epoch=epoch,
                learning_rate_used=learning_rate_used,
            )
            if scheduler is not None:
                scheduler.step()
            if stop:
                break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "linear_probe.pt")
    atomic_json_dump({"history": history}, run_dir / "history.json")

    metrics = {
        **run_metadata,
        "seed": int(seed),
        "feature_dim": feature_dim,
        "probe_trainable_parameters": int(sum(p.numel() for p in model.parameters())),
        "best_epoch": int(best_epoch),
        "best_step": int(best_step),
        "steps_ran": int(global_step),
        "evaluations_ran": int(len(history)),
        "examples_seen": int(examples_seen),
        "effective_epochs": float(examples_seen / len(train_labels)),
        "wall_time_seconds": float(time.perf_counter() - trial_started_at),
        "optimization_unit": "step" if step_mode else "epoch",
        "best_val_loss": float(best_val_loss),
        "best_val_accuracy": float(best_val_accuracy),
        "environment": environment_metadata(),
    }
    if not step_mode:
        metrics["epochs_ran"] = int(len(history))
        metrics["configured_budget"] = int(max_epochs)
        metrics["reached_budget_limit"] = bool(len(history) == max_epochs)
    else:
        metrics["configured_budget"] = int(max_steps)
        metrics["reached_budget_limit"] = bool(global_step == max_steps)
        if max_examples is not None:
            metrics["configured_examples_budget"] = int(max_examples)
            metrics["eval_every_examples"] = int(eval_every_examples)
    atomic_json_dump(metrics, run_dir / "trial_metrics.json")
    return metrics


def evaluate_probe_checkpoint(
    checkpoint_path: Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    runtime_cfg: dict[str, Any],
) -> EvalResult:
    """Evaluate a selected linear-probe checkpoint on a held-out split."""
    _validate_supervised_split(features, labels, "evaluation")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    device = select_device(runtime_cfg["device"])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "weight" not in state or "bias" not in state:
        raise ValueError("Checkpoint is not a linear-probe state dictionary")
    out_features, checkpoint_feature_dim = state["weight"].shape
    if checkpoint_feature_dim != int(features.shape[1]):
        raise ValueError("Checkpoint and evaluation feature dimensions differ")
    if int(torch.max(labels).item()) >= out_features:
        raise ValueError("Evaluation labels exceed the checkpoint output dimension")
    model = nn.Linear(checkpoint_feature_dim, out_features)
    model.load_state_dict(state)
    model.to(device)

    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    return evaluate(model, loader, nn.CrossEntropyLoss(), device)


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    probe_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    seed: int,
    run_dir: Path,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Backward-compatible final run: train on validation, then evaluate test."""
    metrics = train_probe_trial(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        probe_cfg=probe_cfg,
        runtime_cfg=runtime_cfg,
        seed=seed,
        run_dir=run_dir,
        run_metadata=run_metadata,
    )
    test_result = evaluate_probe_checkpoint(
        checkpoint_path=run_dir / "linear_probe.pt",
        features=test_features,
        labels=test_labels,
        batch_size=int(probe_cfg["batch_size"]),
        runtime_cfg=runtime_cfg,
    )
    metrics = {
        **metrics,
        "test_loss": float(test_result.loss),
        "test_accuracy": float(test_result.accuracy),
    }
    atomic_json_dump(metrics, run_dir / "metrics.json")
    return metrics
