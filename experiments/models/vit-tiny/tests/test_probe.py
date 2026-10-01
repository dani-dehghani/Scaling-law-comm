from pathlib import Path

import torch

from src.probe import evaluate_probe_checkpoint, train_linear_probe, train_probe_trial


def _separable_features(n_per_class: int = 12):
    class_zero = torch.tensor([[-1.0, -1.0]]) + 0.05 * torch.randn(n_per_class, 2)
    class_one = torch.tensor([[1.0, 1.0]]) + 0.05 * torch.randn(n_per_class, 2)
    features = torch.cat([class_zero, class_one])
    labels = torch.tensor([0] * n_per_class + [1] * n_per_class)
    return features, labels


def test_hpo_trial_is_validation_only_and_step_based(tmp_path: Path):
    torch.manual_seed(7)
    train_features, train_labels = _separable_features()
    val_features, val_labels = _separable_features(n_per_class=6)

    probe_cfg = {
        "batch_size": 8,
        "learning_rate": 0.05,
        "weight_decay": 0.0,
        "schedule": "constant",
        "max_steps": 20,
        "eval_every_steps": 5,
        "patience_evals": 10,
        "min_delta": 0.0,
    }
    runtime_cfg = {"device": "cpu", "deterministic": True}
    run_dir = tmp_path / "trial"

    metrics = train_probe_trial(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        probe_cfg=probe_cfg,
        runtime_cfg=runtime_cfg,
        seed=0,
        run_dir=run_dir,
        run_metadata={"D": len(train_labels)},
    )

    assert "test_loss" not in metrics
    assert "test_accuracy" not in metrics
    assert metrics["optimization_unit"] == "step"
    assert metrics["steps_ran"] == 20
    assert metrics["examples_seen"] > 0
    assert metrics["effective_epochs"] > 0
    assert metrics["wall_time_seconds"] >= 0
    assert metrics["configured_budget"] == 20
    assert metrics["reached_budget_limit"] is True
    assert metrics["best_step"] in {5, 10, 15, 20}
    assert (run_dir / "trial_metrics.json").exists()
    assert not (run_dir / "metrics.json").exists()

    held_out = evaluate_probe_checkpoint(
        checkpoint_path=run_dir / "linear_probe.pt",
        features=val_features,
        labels=val_labels,
        batch_size=8,
        runtime_cfg=runtime_cfg,
    )
    assert held_out.loss >= 0.0
    assert held_out.accuracy >= 0.9


def test_existing_epoch_sweep_still_writes_final_test_metrics(tmp_path: Path):
    torch.manual_seed(11)
    train_features, train_labels = _separable_features()
    val_features, val_labels = _separable_features(n_per_class=6)
    test_features, test_labels = _separable_features(n_per_class=6)

    metrics = train_linear_probe(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
        probe_cfg={
            "batch_size": 8,
            "epochs": 10,
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "patience": 5,
            "min_delta": 0.0,
        },
        runtime_cfg={"device": "cpu", "deterministic": True},
        seed=0,
        run_dir=tmp_path / "final_run",
        run_metadata={"D": len(train_labels)},
    )

    assert "test_loss" in metrics
    assert "test_accuracy" in metrics
    assert metrics["optimization_unit"] == "epoch"
    assert (tmp_path / "final_run" / "metrics.json").exists()


def test_example_budget_equalizes_data_exposure_across_batch_sizes(tmp_path: Path):
    torch.manual_seed(13)
    train_features, train_labels = _separable_features()
    val_features, val_labels = _separable_features(n_per_class=6)

    seen = []
    for batch_size in [8, 16]:
        metrics = train_probe_trial(
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            probe_cfg={
                "batch_size": batch_size,
                "learning_rate": 0.05,
                "weight_decay": 0.0,
                "schedule": "constant",
                "max_examples": 48,
                "eval_every_examples": 24,
                "patience_evals": 10,
                "min_delta": 0.0,
            },
            runtime_cfg={"device": "cpu", "deterministic": True},
            seed=0,
            run_dir=tmp_path / f"batch_{batch_size}",
            run_metadata={"D": len(train_labels)},
        )
        seen.append(metrics["examples_seen"])
        assert metrics["configured_examples_budget"] == 48
        assert metrics["reached_budget_limit"] is True

    assert seen == [48, 48]


def test_same_seed_reproduces_the_same_probe_checkpoint(tmp_path: Path):
    torch.manual_seed(19)
    train_features, train_labels = _separable_features()
    val_features, val_labels = _separable_features(n_per_class=6)
    probe_cfg = {
        "batch_size": 8,
        "learning_rate": 0.05,
        "weight_decay": 0.0,
        "schedule": "constant",
        "max_steps": 15,
        "eval_every_steps": 5,
        "patience_evals": 10,
        "min_delta": 0.0,
    }
    runtime_cfg = {"device": "cpu", "deterministic": True}

    results = []
    states = []
    for run_name in ["first", "second"]:
        run_dir = tmp_path / run_name
        results.append(
            train_probe_trial(
                train_features=train_features,
                train_labels=train_labels,
                val_features=val_features,
                val_labels=val_labels,
                probe_cfg=probe_cfg,
                runtime_cfg=runtime_cfg,
                seed=3,
                run_dir=run_dir,
                run_metadata={"D": len(train_labels)},
            )
        )
        states.append(
            torch.load(run_dir / "linear_probe.pt", map_location="cpu", weights_only=True)
        )

    assert results[0]["best_val_loss"] == results[1]["best_val_loss"]
    assert results[0]["best_step"] == results[1]["best_step"]
    assert torch.equal(states[0]["weight"], states[1]["weight"])
    assert torch.equal(states[0]["bias"], states[1]["bias"])
