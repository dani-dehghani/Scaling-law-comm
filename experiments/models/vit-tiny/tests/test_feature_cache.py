import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.feature_cache import _extract, cache_metadata_mismatches, load_feature_cache


def _config():
    return {
        "model": {"name": "vit-test", "pretrained": True},
        "data": {"dataset": "CIFAR100", "validation_per_class": 50},
        "runtime": {"amp_feature_extraction": False},
    }


def _write_small_cache(cache_dir):
    cache_dir.mkdir()
    tensors = {
        "train_features": torch.randn(4, 2),
        "train_labels": torch.tensor([0, 1, 0, 1]),
        "train_original_indices": torch.arange(4),
        "val_features": torch.randn(2, 2),
        "val_labels": torch.tensor([0, 1]),
        "val_original_indices": torch.arange(2),
        "test_features": torch.randn(2, 2),
        "test_labels": torch.tensor([0, 1]),
        "test_original_indices": torch.arange(2),
    }
    for name, tensor in tensors.items():
        torch.save(tensor, cache_dir / f"{name}.pt")
    metadata = {
        "feature_dim": 2,
        "train_pool_size": 4,
        "validation_size": 2,
        "test_size": 2,
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_hpo_cache_loading_excludes_test_tensors(tmp_path):
    cache_dir = tmp_path / "cache"
    _write_small_cache(cache_dir)
    tensors, _ = load_feature_cache(cache_dir, include_test=False)
    assert "train_features" in tensors
    assert "val_features" in tensors
    assert all(not name.startswith("test_") for name in tensors)


def test_cache_shape_disagreement_is_rejected(tmp_path):
    cache_dir = tmp_path / "cache"
    _write_small_cache(cache_dir)
    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_dim"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="feature dimension"):
        load_feature_cache(cache_dir, include_test=False)


def test_cache_config_mismatch_is_reported():
    metadata = {
        "dataset": "CIFAR100",
        "model_name": "different-vit",
        "pretrained": True,
        "validation_per_class": 50,
        "amp_feature_extraction": False,
        "train_pool_size": 45000,
        "validation_size": 5000,
        "test_size": 10000,
        "evaluation_split": "cifar100_test",
    }
    assert cache_metadata_mismatches(metadata, _config()) == ["model_name"]


def test_feature_extraction_preserves_labels_and_original_indices():
    images = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    labels = torch.tensor([2, 1, 0])
    indices = torch.tensor([9, 4, 7])
    loader = DataLoader(TensorDataset(images, labels, indices), batch_size=2)

    features, extracted_labels, extracted_indices = _extract(
        torch.nn.Identity(), loader, torch.device("cpu"), amp=False
    )
    assert torch.equal(features, images)
    assert torch.equal(extracted_labels, labels)
    assert torch.equal(extracted_indices, indices)
