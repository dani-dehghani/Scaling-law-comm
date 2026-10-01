from copy import deepcopy
from pathlib import Path

import pytest

from src.config import load_config, validate_config


def _valid_config():
    return {
        "experiment": {"name": "test", "output_dir": "results/x", "cache_dir": "cache/x"},
        "model": {"name": "vit", "pretrained": True, "num_classes": 100},
        "data": {
            "root": "data",
            "dataset": "CIFAR100",
            "validation_per_class": 50,
            "dataset_sizes": [200, 1000, 45000],
            "seeds": [0, 1, 2],
            "num_workers": 0,
            "feature_batch_size": 32,
        },
        "probe": {
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "patience": 3,
            "min_delta": 0.0,
        },
        "runtime": {
            "device": "cpu",
            "amp_feature_extraction": False,
            "deterministic": True,
        },
    }


def test_checked_in_configs_are_valid():
    root = Path(__file__).resolve().parents[1]
    load_config(root / "configs" / "data_sweep_vit_tiny.yaml")
    load_config(root / "configs" / "hpo_lr_stage1.yaml")
    load_config(root / "configs" / "hpo_core_12d_10seeds.yaml")
    load_config(root / "configs" / "hpo_places365_global_8d_10seeds.yaml")
    load_config(root / "configs" / "hpo_places365_refinement_8d_10seeds_30k.yaml")
    load_config(root / "configs" / "hpo_places365_wd_boundary_8d_10seeds_30k.yaml")
    load_config(root / "configs" / "data_sweep_places365_final.yaml")
    load_config(root / "configs" / "data_sweep_places365_final_smoke.yaml")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cfg: cfg["data"].update(dataset_sizes=[250]), "divisible by the number"),
        (lambda cfg: cfg["data"].update(seeds=[0, 0]), "Seeds must be unique"),
        (lambda cfg: cfg["probe"].update(learning_rate=0), "learning_rate"),
        (lambda cfg: cfg["model"].update(num_classes=10), "num_classes"),
    ],
)
def test_invalid_configs_are_rejected(mutation, message):
    cfg = deepcopy(_valid_config())
    mutation(cfg)
    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


def test_hpo_batch_sizes_must_be_positive_and_unique():
    cfg = _valid_config()
    cfg["hpo"] = {
        "learning_rates": [0.001],
        "weight_decays": [0.0],
        "schedules": ["constant"],
        "batch_sizes": [128, 128],
    }
    with pytest.raises(ValueError, match="batch_sizes.*unique"):
        validate_config(cfg)

    cfg["hpo"]["batch_sizes"] = [0]
    with pytest.raises(ValueError, match="batch_sizes.*positive"):
        validate_config(cfg)


def test_places365_proportional_config_accepts_full_training_pool():
    cfg = _valid_config()
    cfg["model"]["num_classes"] = 365
    cfg["data"].update(
        dataset="Places365",
        places_variant="standard",
        sampling="stratified_proportional",
        validation_per_class=50,
        dataset_sizes=[3650, 58400, 1785210],
    )
    validate_config(cfg)


def test_places365_rejects_wrong_class_count():
    cfg = _valid_config()
    cfg["data"].update(
        dataset="Places365",
        sampling="stratified_proportional",
        dataset_sizes=[3650],
    )
    with pytest.raises(ValueError, match="num_classes must be 365"):
        validate_config(cfg)


def test_protocol_status_must_be_known():
    cfg = _valid_config()
    cfg["experiment"]["protocol_status"] = "maybe"
    with pytest.raises(ValueError, match="protocol_status"):
        validate_config(cfg)
