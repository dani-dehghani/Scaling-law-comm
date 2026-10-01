from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DATASET_SPECS = {
    "CIFAR100": {
        "num_classes": 100,
        "train_size": 50_000,
        "smallest_train_class": 500,
    },
    "Places365": {
        "num_classes": 365,
        "train_size": 1_803_460,
        "smallest_train_class": 3_068,
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")

    required = {"experiment", "model", "data", "probe", "runtime"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    for section in required:
        if not isinstance(cfg[section], dict):
            raise ValueError(f"Config section '{section}' must be a mapping")

    data_cfg = cfg["data"]
    experiment_cfg = cfg["experiment"]
    model_cfg = cfg["model"]
    probe_cfg = cfg["probe"]
    runtime_cfg = cfg["runtime"]

    protocol_status = str(experiment_cfg.get("protocol_status", "approved"))
    if protocol_status not in {"approved", "blocked", "smoke_only"}:
        raise ValueError(
            "experiment.protocol_status must be 'approved', 'blocked', or "
            "'smoke_only'"
        )

    dataset = str(data_cfg.get("dataset"))
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    spec = DATASET_SPECS[dataset]
    if int(model_cfg.get("num_classes", -1)) != int(spec["num_classes"]):
        raise ValueError(
            f"model.num_classes must be {spec['num_classes']} for {dataset}"
        )
    if dataset == "Places365" and str(data_cfg.get("places_variant", "standard")) != "standard":
        raise ValueError("Only Places365-Standard is supported by this protocol")

    sizes = data_cfg["dataset_sizes"]
    seeds = data_cfg["seeds"]
    val_per_class = int(data_cfg["validation_per_class"])

    if not sizes or any(int(d) <= 0 for d in sizes):
        raise ValueError("dataset_sizes must contain positive integers")
    if sorted(set(map(int, sizes))) != list(map(int, sizes)):
        raise ValueError("dataset_sizes must be strictly increasing and unique")
    sampling = str(data_cfg.get("sampling", "balanced"))
    if sampling not in {"balanced", "stratified_proportional"}:
        raise ValueError("data.sampling must be 'balanced' or 'stratified_proportional'")
    if sampling == "balanced" and any(
        int(d) % int(spec["num_classes"]) != 0 for d in sizes
    ):
        raise ValueError(
            "Every dataset size must be divisible by the number of classes "
            "for exact class balance"
        )
    if sampling == "stratified_proportional" and any(
        int(d) < int(spec["num_classes"]) for d in sizes
    ):
        raise ValueError(
            "Every stratified-proportional dataset size must be at least the "
            "number of classes"
        )
    if not seeds:
        raise ValueError("At least one seed is required")
    integer_seeds = list(map(int, seeds))
    if len(set(integer_seeds)) != len(integer_seeds):
        raise ValueError("Seeds must be unique")
    if not (1 <= val_per_class < int(spec["smallest_train_class"])):
        raise ValueError(
            "validation_per_class must leave at least one training example in "
            f"the smallest {dataset} class"
        )

    max_train = int(spec["train_size"]) - val_per_class * int(spec["num_classes"])
    if max(map(int, sizes)) > max_train:
        raise ValueError(
            f"Largest D exceeds the post-validation training pool ({max_train})"
        )

    if sampling == "balanced":
        max_balanced = (
            int(spec["smallest_train_class"]) - val_per_class
        ) * int(spec["num_classes"])
        if max(map(int, sizes)) > max_balanced:
            raise ValueError(
                "Largest D exceeds the exactly balanced post-validation pool "
                f"({max_balanced})"
            )

    required_probe = {"batch_size", "learning_rate", "weight_decay", "min_delta"}
    missing_probe = required_probe - set(probe_cfg)
    if missing_probe:
        raise ValueError(f"Missing probe settings: {sorted(missing_probe)}")
    if int(probe_cfg["batch_size"]) <= 0:
        raise ValueError("probe.batch_size must be positive")
    if float(probe_cfg["learning_rate"]) <= 0:
        raise ValueError("probe.learning_rate must be positive")
    if float(probe_cfg["weight_decay"]) < 0:
        raise ValueError("probe.weight_decay cannot be negative")
    if float(probe_cfg["min_delta"]) < 0:
        raise ValueError("probe.min_delta cannot be negative")

    schedule = str(probe_cfg.get("schedule", "cosine"))
    if schedule not in {"constant", "cosine"}:
        raise ValueError("probe.schedule must be 'constant' or 'cosine'")
    if "max_steps" in probe_cfg and "max_examples" in probe_cfg:
        raise ValueError("probe must define only one of max_steps and max_examples")
    if "max_examples" in probe_cfg:
        for key in ["max_examples", "eval_every_examples", "patience_evals"]:
            if key not in probe_cfg or int(probe_cfg[key]) <= 0:
                raise ValueError(f"probe.{key} must be present and positive")
    elif "max_steps" in probe_cfg:
        for key in ["max_steps", "eval_every_steps", "patience_evals"]:
            if key not in probe_cfg or int(probe_cfg[key]) <= 0:
                raise ValueError(f"probe.{key} must be present and positive")
    else:
        for key in ["epochs", "patience"]:
            if key not in probe_cfg or int(probe_cfg[key]) <= 0:
                raise ValueError(f"probe.{key} must be present and positive")

    required_runtime = {"device", "amp_feature_extraction", "deterministic"}
    missing_runtime = required_runtime - set(runtime_cfg)
    if missing_runtime:
        raise ValueError(f"Missing runtime settings: {sorted(missing_runtime)}")

    if "hpo" in cfg:
        hpo = cfg["hpo"]
        for key in ["learning_rates", "weight_decays", "schedules"]:
            if key not in hpo or not hpo[key]:
                raise ValueError(f"hpo.{key} must contain at least one value")
        batch_sizes = hpo.get("batch_sizes", [probe_cfg["batch_size"]])
        if not batch_sizes:
            raise ValueError("hpo.batch_sizes must contain at least one value")
        if any(float(lr) <= 0 for lr in hpo["learning_rates"]):
            raise ValueError("hpo.learning_rates must be positive")
        if any(float(wd) < 0 for wd in hpo["weight_decays"]):
            raise ValueError("hpo.weight_decays cannot be negative")
        if any(int(batch_size) <= 0 for batch_size in batch_sizes):
            raise ValueError("hpo.batch_sizes must be positive")
        invalid_schedules = set(hpo["schedules"]) - {"constant", "cosine"}
        if invalid_schedules:
            raise ValueError(f"Unsupported HPO schedules: {sorted(invalid_schedules)}")

        search_values: dict[str, list[Any]] = {
            "learning_rates": list(map(float, hpo["learning_rates"])),
            "weight_decays": list(map(float, hpo["weight_decays"])),
            "schedules": list(map(str, hpo["schedules"])),
            "batch_sizes": list(map(int, batch_sizes)),
        }
        for key, values in search_values.items():
            if len(values) != len(set(values)):
                raise ValueError(f"hpo.{key} must contain unique values")


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path
