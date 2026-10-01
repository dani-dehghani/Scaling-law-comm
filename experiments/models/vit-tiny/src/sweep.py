from __future__ import annotations

from typing import Any


def budget_warning(metrics: dict[str, Any]) -> bool:
    """Return whether a step-based run may have stopped before convergence."""
    if metrics.get("optimization_unit") != "step":
        return False
    budget = int(metrics["configured_budget"])
    return bool(metrics.get("reached_budget_limit", False)) and int(
        metrics["best_step"]
    ) >= 0.9 * budget
