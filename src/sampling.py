from __future__ import annotations

from collections import Counter

import numpy as np


def stratified_train_val_split(
    targets: list[int] | np.ndarray,
    validation_per_class: int,
    seed: int = 12345,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed train-pool and validation indices for CIFAR-style labels."""
    y = np.asarray(targets, dtype=np.int64)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("targets must be a non-empty one-dimensional array")
    if validation_per_class <= 0:
        raise ValueError("validation_per_class must be positive")

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    smallest_class = min(int(np.sum(y == cls)) for cls in classes)
    if validation_per_class >= smallest_class:
        raise ValueError(
            "validation_per_class must leave at least one training example per class"
        )

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for cls in classes:
        cls_idx: np.ndarray = np.asarray(
            np.flatnonzero(y == cls), dtype=np.int64
        ).copy()
        rng.shuffle(cls_idx)
        val_parts.append(cls_idx[:validation_per_class])
        train_parts.append(cls_idx[validation_per_class:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_nested_balanced_subsets(
    train_pool_indices: np.ndarray,
    targets: list[int] | np.ndarray,
    dataset_sizes: list[int],
    seed: int,
) -> dict[int, np.ndarray]:
    """
    Build class-balanced, nested subsets.

    For CIFAR-100 we require D to be divisible by the number of classes. This
    makes every subset exactly balanced and makes the smaller subset a strict
    subset of every larger subset for a given seed.
    """
    y = np.asarray(targets, dtype=np.int64)
    train_pool_indices = np.asarray(train_pool_indices, dtype=np.int64)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("targets must be a non-empty one-dimensional array")
    if train_pool_indices.ndim != 1 or len(train_pool_indices) == 0:
        raise ValueError("train_pool_indices must be non-empty and one-dimensional")
    if np.any(train_pool_indices < 0) or np.any(train_pool_indices >= len(y)):
        raise ValueError("train_pool_indices contains an out-of-range index")
    if len(np.unique(train_pool_indices)) != len(train_pool_indices):
        raise ValueError("train_pool_indices must not contain duplicates")
    if not dataset_sizes or any(int(d) <= 0 for d in dataset_sizes):
        raise ValueError("dataset_sizes must contain positive integers")
    integer_sizes = list(map(int, dataset_sizes))
    if integer_sizes != sorted(set(integer_sizes)):
        raise ValueError("dataset_sizes must be strictly increasing and unique")

    classes = np.unique(y[train_pool_indices])
    n_classes = len(classes)

    for d in integer_sizes:
        if d % n_classes != 0:
            raise ValueError(
                f"D={d} is not divisible by {n_classes}; exact class balance is impossible."
            )

    rng = np.random.default_rng(seed)
    ordered_by_class: dict[int, np.ndarray] = {}
    for cls in classes:
        cls_pool = train_pool_indices[y[train_pool_indices] == cls].copy()
        rng.shuffle(cls_pool)
        ordered_by_class[int(cls)] = cls_pool

    largest_per_class = integer_sizes[-1] // n_classes
    smallest_pool = min(len(indices) for indices in ordered_by_class.values())
    if largest_per_class > smallest_pool:
        raise ValueError(
            f"Largest D needs {largest_per_class} examples per class, but the "
            f"smallest class pool contains only {smallest_pool}"
        )

    result: dict[int, np.ndarray] = {}
    for d in integer_sizes:
        per_class = d // n_classes
        chosen = np.concatenate(
            [ordered_by_class[int(cls)][:per_class] for cls in classes]
        )
        # Shuffle presentation order without changing membership.
        local_rng = np.random.default_rng(seed + int(d))
        local_rng.shuffle(chosen)
        if len(chosen) != d:
            raise RuntimeError(f"Internal sampling error: requested D={d}, got {len(chosen)}")
        result[int(d)] = chosen

    return result


def build_nested_stratified_subsets(
    train_pool_indices: np.ndarray,
    targets: list[int] | np.ndarray,
    dataset_sizes: list[int],
    seed: int,
) -> dict[int, np.ndarray]:
    """Build nested subsets that closely preserve an imbalanced class distribution.

    Each class is shuffled independently. Its examples receive evenly spaced random
    priorities in [0, 1), and all classes are merged by priority. Prefixes of this
    master ordering are therefore nested and track the full pool's class proportions
    without discarding examples from larger classes. The complete-pool endpoint uses
    every example exactly once.
    """
    y = np.asarray(targets, dtype=np.int64)
    train_pool_indices = np.asarray(train_pool_indices, dtype=np.int64)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("targets must be a non-empty one-dimensional array")
    if train_pool_indices.ndim != 1 or len(train_pool_indices) == 0:
        raise ValueError("train_pool_indices must be non-empty and one-dimensional")
    if np.any(train_pool_indices < 0) or np.any(train_pool_indices >= len(y)):
        raise ValueError("train_pool_indices contains an out-of-range index")
    if len(np.unique(train_pool_indices)) != len(train_pool_indices):
        raise ValueError("train_pool_indices must not contain duplicates")

    integer_sizes = list(map(int, dataset_sizes))
    if not integer_sizes or any(d <= 0 for d in integer_sizes):
        raise ValueError("dataset_sizes must contain positive integers")
    if integer_sizes != sorted(set(integer_sizes)):
        raise ValueError("dataset_sizes must be strictly increasing and unique")
    if integer_sizes[-1] > len(train_pool_indices):
        raise ValueError(
            f"Largest D={integer_sizes[-1]} exceeds train pool size "
            f"{len(train_pool_indices)}"
        )

    classes = np.unique(y[train_pool_indices])
    if integer_sizes[0] < len(classes):
        raise ValueError(
            "Smallest D must be at least the number of classes for stratified sampling"
        )

    rng = np.random.default_rng(seed)
    index_parts: list[np.ndarray] = []
    priority_parts: list[np.ndarray] = []
    tie_parts: list[np.ndarray] = []
    for cls in classes:
        cls_pool = train_pool_indices[y[train_pool_indices] == cls].copy()
        rng.shuffle(cls_pool)
        n_cls = len(cls_pool)
        priorities = (np.arange(n_cls, dtype=np.float64) + rng.random(n_cls)) / n_cls
        index_parts.append(cls_pool)
        priority_parts.append(priorities)
        tie_parts.append(rng.random(n_cls))

    all_indices = np.concatenate(index_parts)
    priorities = np.concatenate(priority_parts)
    tie_breakers = np.concatenate(tie_parts)
    order = np.lexsort((tie_breakers, priorities))
    master_order = all_indices[order]

    result: dict[int, np.ndarray] = {}
    for d in integer_sizes:
        subset = master_order[:d].copy()
        local_rng = np.random.default_rng(seed + d)
        local_rng.shuffle(subset)
        result[d] = subset
    return result


def class_counts(indices: np.ndarray, targets: list[int] | np.ndarray) -> Counter:
    y = np.asarray(targets, dtype=np.int64)
    return Counter(map(int, y[np.asarray(indices, dtype=np.int64)]))
