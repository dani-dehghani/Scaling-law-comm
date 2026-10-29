from src.sweep import budget_warning
from cli.run_sweep import _enforce_protocol_status


def test_budget_warning_requires_step_limit_and_late_best_checkpoint():
    metrics = {
        "optimization_unit": "step",
        "configured_budget": 10000,
        "reached_budget_limit": True,
        "best_step": 9500,
    }
    assert budget_warning(metrics)


def test_budget_warning_allows_converged_or_epoch_runs():
    converged = {
        "optimization_unit": "step",
        "configured_budget": 10000,
        "reached_budget_limit": False,
        "best_step": 9500,
    }
    epoch_run = {
        "optimization_unit": "epoch",
        "configured_budget": 100,
        "reached_budget_limit": True,
        "best_step": 10000,
    }
    assert not budget_warning(converged)
    assert not budget_warning(epoch_run)


def test_blocked_protocol_allows_dry_run_but_rejects_training():
    cfg = {
        "experiment": {
            "protocol_status": "blocked",
            "protocol_block_reason": "Pending HPO",
        }
    }
    _enforce_protocol_status(cfg, dry_run=True, train_only=False)
    import pytest

    with pytest.raises(RuntimeError, match="blocked pending HPO"):
        _enforce_protocol_status(cfg, dry_run=False, train_only=True)


def test_smoke_protocol_requires_train_only():
    cfg = {"experiment": {"protocol_status": "smoke_only"}}
    _enforce_protocol_status(cfg, dry_run=False, train_only=True)
    import pytest

    with pytest.raises(RuntimeError, match="only run with --train-only"):
        _enforce_protocol_status(cfg, dry_run=False, train_only=False)
