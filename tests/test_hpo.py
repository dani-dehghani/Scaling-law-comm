import pytest

from src.hpo import aggregate_trials, select_best_by_D, select_global_config


def _trial(D, seed, lr, loss):
    return {
        "D": D,
        "seed": seed,
        "learning_rate": lr,
        "weight_decay": 0.0,
        "schedule": "constant",
        "batch_size": 256,
        "best_val_loss": loss,
        "best_val_accuracy": 1.0 - loss / 10.0,
        "best_step": 50,
        "steps_ran": 100,
    }


def test_hpo_aggregation_and_selection():
    rows = [
        _trial(100, 0, 0.001, 2.0),
        _trial(100, 1, 0.001, 2.2),
        _trial(100, 0, 0.003, 1.5),
        _trial(100, 1, 0.003, 1.7),
        _trial(1000, 0, 0.001, 0.8),
        _trial(1000, 1, 0.001, 1.0),
        _trial(1000, 0, 0.003, 1.1),
        _trial(1000, 1, 0.003, 1.3),
    ]

    aggregated = aggregate_trials(rows)
    selected = select_best_by_D(aggregated)
    assert [row["learning_rate"] for row in selected] == [0.003, 0.001]

    global_selection = select_global_config(aggregated)
    assert global_selection["learning_rate"] == pytest.approx(0.001)
    assert global_selection["batch_size"] == 256
    assert global_selection["dataset_sizes_compared"] == 2


def test_global_selection_rejects_incomplete_search_grid():
    rows = [
        _trial(100, 0, 0.001, 2.0),
        _trial(100, 0, 0.003, 1.5),
        _trial(1000, 0, 0.001, 0.8),
    ]
    with pytest.raises(ValueError, match="Every configuration"):
        select_global_config(aggregate_trials(rows))
