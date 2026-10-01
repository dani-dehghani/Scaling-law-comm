from copy import deepcopy
import json

import numpy as np
import pytest

from run_hpo import (
    _configurations,
    _expected_count,
    _finalize_results,
    _save_numpy_array_atomic,
    _trial_id,
)


def _config():
    return {
        "data": {"dataset_sizes": [200, 1000], "seeds": [0, 1, 2]},
        "probe": {"batch_size": 256},
        "hpo": {
            "learning_rates": [0.0001, 0.001],
            "weight_decays": [0.0, 0.0001],
            "schedules": ["constant", "cosine"],
            "batch_sizes": [64, 256],
        },
    }


def test_full_hpo_trial_count_includes_batch_size():
    cfg = _config()
    assert len(_configurations(cfg)) == 16
    assert _expected_count(cfg) == 96


def test_hpo_uses_probe_batch_size_when_search_list_is_absent():
    cfg = deepcopy(_config())
    del cfg["hpo"]["batch_sizes"]
    assert {row[3] for row in _configurations(cfg)} == {256}


def test_trial_ids_do_not_round_distinct_learning_rates_together():
    first = _trial_id(0.001, 0.0001, "constant", 256, True)
    second = _trial_id(0.0015, 0.0001, "constant", 256, True)
    assert first != second


def test_atomic_subset_save_reuses_only_identical_data(tmp_path):
    path = tmp_path / "subset.npy"
    expected = np.array([3, 1, 4], dtype=np.int64)
    _save_numpy_array_atomic(expected, path)
    _save_numpy_array_atomic(expected.copy(), path)
    assert np.array_equal(np.load(path), expected)

    with pytest.raises(ValueError, match="Existing subset disagrees"):
        _save_numpy_array_atomic(np.array([9], dtype=np.int64), path)


def test_finalization_refuses_incomplete_results_and_accepts_complete_results(tmp_path):
    cfg = _config()
    cfg["data"] = {"dataset_sizes": [200], "seeds": [0, 1]}
    cfg["hpo"] = {
        "learning_rates": [0.001],
        "weight_decays": [0.0001],
        "schedules": ["constant"],
        "batch_sizes": [256],
    }
    output_dir = tmp_path / "results"

    def write_metric(seed, loss):
        directory = output_dir / "trials" / "trial" / f"D_00200_seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        row = {
            "D": 200,
            "seed": seed,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "schedule": "constant",
            "batch_size": 256,
            "best_val_loss": loss,
            "best_val_accuracy": 0.5,
            "best_step": 100,
            "steps_ran": 300,
        }
        (directory / "trial_metrics.json").write_text(json.dumps(row))

    write_metric(0, 1.0)
    with pytest.raises(RuntimeError, match="Completed 1/2"):
        _finalize_results(cfg, output_dir)
    assert not (output_dir / "selected_global.json").exists()

    write_metric(1, 1.2)
    _finalize_results(cfg, output_dir)
    completion = json.loads((output_dir / "completion.json").read_text())
    assert completion["complete"] is True
    assert (output_dir / "selected_global.json").exists()
