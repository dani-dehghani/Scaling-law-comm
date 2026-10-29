from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .io_utils import atomic_json_dump
from .reproducibility import environment_metadata, select_device
from .sampling import stratified_train_val_split


SPLIT_SEED = 12345

PLACES365_STANDARD_TRAIN_SIZE = 1_803_460
PLACES365_VALIDATION_SIZE = 36_500


def _expected_cache_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg["data"]
    dataset = str(data_cfg["dataset"])
    validation_per_class = int(data_cfg["validation_per_class"])
    common = {
        "dataset": dataset,
        "model_name": cfg["model"]["name"],
        "pretrained": bool(cfg["model"]["pretrained"]),
        "validation_per_class": validation_per_class,
        "amp_feature_extraction": bool(cfg["runtime"]["amp_feature_extraction"]),
    }
    if dataset == "CIFAR100":
        return {
            **common,
            "train_pool_size": (500 - validation_per_class) * 100,
            "validation_size": validation_per_class * 100,
            "test_size": 10_000,
            "evaluation_split": "cifar100_test",
        }
    if dataset == "Places365":
        return {
            **common,
            "places_variant": str(data_cfg.get("places_variant", "standard")),
            "places_small": bool(data_cfg.get("places_small", True)),
            "train_pool_size": (
                PLACES365_STANDARD_TRAIN_SIZE - validation_per_class * 365
            ),
            "validation_size": validation_per_class * 365,
            "test_size": PLACES365_VALIDATION_SIZE,
            "evaluation_split": "places365_official_validation",
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def cache_metadata_mismatches(
    metadata: dict[str, Any], cfg: dict[str, Any]
) -> list[str]:
    """Return config fields that are incompatible with an existing cache."""
    expected = _expected_cache_metadata(cfg)
    return [
        key for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    ]


def _validate_loaded_tensors(
    tensors: dict[str, torch.Tensor], metadata: dict[str, Any]
) -> None:
    for split in ["train", "val", "test"]:
        feature_key = f"{split}_features"
        label_key = f"{split}_labels"
        if feature_key not in tensors:
            continue
        features = tensors[feature_key]
        labels = tensors[label_key]
        if features.ndim != 2 or labels.ndim != 1:
            raise ValueError(f"Malformed {split} cache tensor dimensions")
        if len(features) != len(labels):
            raise ValueError(f"{split} cache features and labels have different lengths")
        index_key = f"{split}_original_indices"
        if index_key in tensors and len(tensors[index_key]) != len(labels):
            raise ValueError(f"{split} cache indices and labels have different lengths")

        expected_size = metadata.get(f"{split}_pool_size", metadata.get(f"{split}_size"))
        if expected_size is not None and len(labels) != int(expected_size):
            raise ValueError(f"{split} cache size disagrees with metadata")
        if "feature_dim" in metadata and features.shape[1] != int(metadata["feature_dim"]):
            raise ValueError(f"{split} feature dimension disagrees with metadata")


def load_feature_cache(cache_dir: Path, include_test: bool = True):
    """Load cached tensors, optionally keeping the test split inaccessible."""
    names = [
        "train_features",
        "train_labels",
        "train_original_indices",
        "val_features",
        "val_labels",
        "val_original_indices",
    ]
    if include_test:
        names.extend(["test_features", "test_labels", "test_original_indices"])

    tensors = {
        name: torch.load(cache_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        for name in names
    }
    with (cache_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    _validate_loaded_tensors(tensors, metadata)
    return tensors, metadata


@torch.inference_mode()
def _extract(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from tqdm import tqdm

    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    original_indices: list[torch.Tensor] = []

    use_amp = amp and device.type == "cuda"
    for images, target, index in tqdm(loader, desc="Extracting features", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            z = model(images)
        features.append(z.float().cpu())
        labels.append(target.long().cpu())
        original_indices.append(index.long().cpu())

    return (
        torch.cat(features, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(original_indices, dim=0),
    )


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: torch.utils.data.Dataset, indices: np.ndarray):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        original_index = int(self.indices[item])
        image, target = self.base_dataset[original_index]
        return image, target, original_index


def prepare_feature_cache(cfg: dict[str, Any], project_root: Path, force: bool = False) -> Path:
    import timm
    from torchvision.datasets import CIFAR100, Places365

    exp_cfg = cfg["experiment"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    runtime_cfg = cfg["runtime"]

    cache_dir = Path(exp_cfg["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = cache_dir / "metadata.json"
    required_files = [
        cache_dir / "train_features.pt",
        cache_dir / "train_labels.pt",
        cache_dir / "train_original_indices.pt",
        cache_dir / "val_features.pt",
        cache_dir / "val_labels.pt",
        cache_dir / "val_original_indices.pt",
        cache_dir / "test_features.pt",
        cache_dir / "test_labels.pt",
        cache_dir / "test_original_indices.pt",
    ]
    if not force and metadata_path.exists() and all(p.exists() for p in required_files):
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        mismatches = cache_metadata_mismatches(metadata, cfg)
        if mismatches:
            raise RuntimeError(
                "Existing feature cache is incompatible with the current config "
                f"for fields: {mismatches}. Use --force to rebuild it."
            )
        print(f"Feature cache already exists: {cache_dir}")
        return cache_dir

    device = select_device(runtime_cfg["device"])
    print(f"Feature extraction device: {device}")
    print(f"Loading pretrained model: {model_cfg['name']}")

    model = timm.create_model(
        model_cfg["name"], pretrained=bool(model_cfg["pretrained"]), num_classes=0
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    data_root = Path(data_cfg["root"])
    if not data_root.is_absolute():
        data_root = project_root / data_root

    dataset_name = str(data_cfg["dataset"])
    if dataset_name == "CIFAR100":
        train_full = CIFAR100(
            root=data_root, train=True, download=True, transform=transform
        )
        test_ds = CIFAR100(
            root=data_root, train=False, download=True, transform=transform
        )
        train_targets = np.asarray(train_full.targets, dtype=np.int64)
        evaluation_split = "cifar100_test"
    elif dataset_name == "Places365":
        download = bool(data_cfg.get("download", False))
        small = bool(data_cfg.get("places_small", True))
        size_token = "256" if small else "large"
        train_images_dir = data_root / f"data_{size_token}_standard"
        val_images_dir = data_root / f"val_{size_token}"
        train_full = Places365(
            root=data_root,
            split="train-standard",
            small=small,
            # torchvision raises when download=True and an extracted image
            # directory already exists, so resume must disable downloading for
            # each split independently once extraction has completed.
            download=download and not train_images_dir.exists(),
            transform=transform,
        )
        # Places365's public test labels are unavailable. Its official labeled
        # validation set is kept locked here as the final evaluation split;
        # HPO uses a fixed internal validation split carved from training.
        test_ds = Places365(
            root=data_root,
            split="val",
            small=small,
            download=download and not val_images_dir.exists(),
            transform=transform,
        )
        train_targets = np.asarray(train_full.targets, dtype=np.int64)
        evaluation_split = "places365_official_validation"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_pool_idx, val_idx = stratified_train_val_split(
        train_targets,
        validation_per_class=int(data_cfg["validation_per_class"]),
        seed=SPLIT_SEED,
    )

    train_ds = IndexedDataset(train_full, train_pool_idx)
    val_ds = IndexedDataset(train_full, val_idx)
    test_indices = np.arange(len(test_ds), dtype=np.int64)
    test_indexed = IndexedDataset(test_ds, test_indices)

    loader_kwargs = dict(
        batch_size=int(data_cfg["feature_batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(data_cfg["num_workers"]) > 0,
    )

    train_features, train_labels, train_original_indices = _extract(
        model,
        DataLoader(train_ds, **loader_kwargs),
        device,
        bool(runtime_cfg["amp_feature_extraction"]),
    )
    val_features, val_labels, val_original_indices = _extract(
        model,
        DataLoader(val_ds, **loader_kwargs),
        device,
        bool(runtime_cfg["amp_feature_extraction"]),
    )
    test_features, test_labels, test_original_indices = _extract(
        model,
        DataLoader(test_indexed, **loader_kwargs),
        device,
        bool(runtime_cfg["amp_feature_extraction"]),
    )

    torch.save(train_features, cache_dir / "train_features.pt")
    torch.save(train_labels, cache_dir / "train_labels.pt")
    torch.save(train_original_indices, cache_dir / "train_original_indices.pt")
    torch.save(val_features, cache_dir / "val_features.pt")
    torch.save(val_labels, cache_dir / "val_labels.pt")
    torch.save(val_original_indices, cache_dir / "val_original_indices.pt")
    torch.save(test_features, cache_dir / "test_features.pt")
    torch.save(test_labels, cache_dir / "test_labels.pt")
    torch.save(test_original_indices, cache_dir / "test_original_indices.pt")

    metadata = {
        "model_name": model_cfg["name"],
        "dataset": dataset_name,
        "pretrained": bool(model_cfg["pretrained"]),
        "feature_dim": int(train_features.shape[1]),
        "backbone_total_parameters": int(sum(p.numel() for p in model.parameters())),
        "train_pool_size": int(len(train_features)),
        "validation_size": int(len(val_features)),
        "test_size": int(len(test_features)),
        "validation_per_class": int(data_cfg["validation_per_class"]),
        "evaluation_split": evaluation_split,
        "input_size": list(data_config.get("input_size", ())),
        "mean": list(data_config.get("mean", ())),
        "std": list(data_config.get("std", ())),
        "amp_feature_extraction": bool(runtime_cfg["amp_feature_extraction"]),
        "split_seed": SPLIT_SEED,
        "transform_kind": "timm_deterministic_evaluation",
        "environment": environment_metadata(),
    }
    if dataset_name == "Places365":
        metadata["places_variant"] = str(data_cfg.get("places_variant", "standard"))
        metadata["places_small"] = bool(data_cfg.get("places_small", True))
    atomic_json_dump(metadata, metadata_path)
    print(f"Saved feature cache to: {cache_dir}")
    return cache_dir
