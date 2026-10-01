import numpy as np
import pytest

from src.sampling import (
    build_nested_balanced_subsets,
    build_nested_stratified_subsets,
    class_counts,
    stratified_train_val_split,
)


def fake_cifar100_targets():
    return np.repeat(np.arange(100), 500)


def test_train_val_split_is_balanced():
    y = fake_cifar100_targets()
    train_idx, val_idx = stratified_train_val_split(y, validation_per_class=50, seed=12345)
    assert len(train_idx) == 45000
    assert len(val_idx) == 5000
    assert set(class_counts(val_idx, y).values()) == {50}
    assert set(class_counts(train_idx, y).values()) == {450}
    assert set(train_idx).isdisjoint(set(val_idx))
    train_again, val_again = stratified_train_val_split(
        y, validation_per_class=50, seed=12345
    )
    assert np.array_equal(train_idx, train_again)
    assert np.array_equal(val_idx, val_again)


def test_subsets_are_balanced_and_nested():
    y = fake_cifar100_targets()
    train_idx, _ = stratified_train_val_split(y, validation_per_class=50, seed=12345)
    subsets = build_nested_balanced_subsets(train_idx, y, [500, 1000, 5000], seed=0)

    assert set(class_counts(subsets[500], y).values()) == {5}
    assert set(class_counts(subsets[1000], y).values()) == {10}
    assert set(class_counts(subsets[5000], y).values()) == {50}
    assert set(subsets[500]).issubset(set(subsets[1000]))
    assert set(subsets[1000]).issubset(set(subsets[5000]))
    assert all(len(subset) == D for D, subset in subsets.items())


def test_split_rejects_validation_that_empties_a_class():
    with pytest.raises(ValueError, match="leave at least one"):
        stratified_train_val_split([0, 0, 1, 1], validation_per_class=2)


def test_subsets_reject_insufficient_class_pool():
    targets = np.repeat(np.arange(2), 2)
    with pytest.raises(ValueError, match="smallest class pool"):
        build_nested_balanced_subsets(
            np.arange(4), targets, dataset_sizes=[2, 6], seed=0
        )


def test_stratified_subsets_are_nested_and_preserve_the_full_pool():
    targets = np.repeat(np.arange(3), [20, 30, 50])
    pool = np.arange(len(targets))
    subsets = build_nested_stratified_subsets(
        pool, targets, dataset_sizes=[15, 40, 100], seed=7
    )

    assert set(subsets[15]).issubset(set(subsets[40]))
    assert set(subsets[40]).issubset(set(subsets[100]))
    assert set(subsets[100]) == set(pool)
    assert all(len(subsets[D]) == D for D in subsets)
    assert set(class_counts(subsets[15], targets)) == {0, 1, 2}


def test_stratified_subsets_are_reproducible():
    targets = np.repeat(np.arange(2), [10, 30])
    pool = np.arange(len(targets))
    first = build_nested_stratified_subsets(pool, targets, [10, 20], seed=3)
    second = build_nested_stratified_subsets(pool, targets, [10, 20], seed=3)
    assert np.array_equal(first[10], second[10])
    assert np.array_equal(first[20], second[20])
