from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev
from typing import Any, Iterable


CONFIG_FIELDS = ("learning_rate", "weight_decay", "schedule", "batch_size")


def config_key(row: dict[str, Any]) -> tuple[float, float, str, int]:
    return (
        float(row["learning_rate"]),
        float(row["weight_decay"]),
        str(row["schedule"]),
        int(row.get("batch_size", 256)),
    )


def aggregate_trials(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average validation results across seeds for each D/configuration."""
    groups: dict[tuple[int, float, float, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (int(row["D"]), *config_key(row))
        groups[key].append(row)

    aggregated = []
    for (D, learning_rate, weight_decay, schedule, batch_size), group in sorted(
        groups.items()
    ):
        losses = [float(row["best_val_loss"]) for row in group]
        accuracies = [float(row["best_val_accuracy"]) for row in group]
        best_steps = [int(row["best_step"]) for row in group]
        steps_ran = [int(row["steps_ran"]) for row in group]
        configured_budgets = [
            int(row.get("configured_budget", row["steps_ran"])) for row in group
        ]
        aggregated.append(
            {
                "D": D,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "schedule": schedule,
                "batch_size": batch_size,
                "val_loss_mean": mean(losses),
                "val_loss_std": stdev(losses) if len(losses) > 1 else 0.0,
                "val_accuracy_mean": mean(accuracies),
                "best_step_mean": mean(best_steps),
                "steps_ran_mean": mean(steps_ran),
                "fraction_best_near_step_limit": mean(
                    best >= 0.9 * budget
                    for best, budget in zip(best_steps, configured_budgets)
                ),
                "fraction_reached_budget_limit": mean(
                    bool(row.get("reached_budget_limit", False)) for row in group
                ),
                "examples_seen_mean": mean(
                    int(row.get("examples_seen", 0)) for row in group
                ),
                "wall_time_seconds_mean": mean(
                    float(row.get("wall_time_seconds", 0.0)) for row in group
                ),
                "n_seeds": len({int(row["seed"]) for row in group}),
            }
        )
    return aggregated


def select_best_by_D(aggregated: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the lowest mean validation loss independently at each D."""
    by_D: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregated:
        by_D[int(row["D"])].append(row)
    return [
        min(by_D[D], key=lambda row: (float(row["val_loss_mean"]), config_key(row)))
        for D in sorted(by_D)
    ]


def select_global_config(aggregated: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Choose one config by mean relative regret, giving every D equal weight."""
    rows = list(aggregated)
    best_loss_by_D: dict[int, float] = {}
    for row in rows:
        D = int(row["D"])
        loss = float(row["val_loss_mean"])
        best_loss_by_D[D] = min(best_loss_by_D.get(D, float("inf")), loss)

    regrets: dict[tuple[float, float, str, int], list[float]] = defaultdict(list)
    templates: dict[tuple[float, float, str, int], dict[str, Any]] = {}
    for row in rows:
        key = config_key(row)
        D = int(row["D"])
        regrets[key].append(float(row["val_loss_mean"]) / best_loss_by_D[D] - 1.0)
        templates[key] = row

    expected_D_count = len(best_loss_by_D)
    incomplete = [key for key, values in regrets.items() if len(values) != expected_D_count]
    if incomplete:
        raise ValueError("Every configuration must be evaluated at every dataset size")

    best_key = min(regrets, key=lambda key: (mean(regrets[key]), key))
    return {
        **{field: templates[best_key][field] for field in CONFIG_FIELDS},
        "mean_relative_regret": mean(regrets[best_key]),
        "dataset_sizes_compared": expected_D_count,
    }
