"""Stratified split on a fake labels array: no CIFAR-10 download needed."""

import json
from collections import Counter

import numpy as np
import pytest

from src.data.cifar10 import load_or_create_split, make_stratified_split

N_CLASSES = 10
PER_CLASS = 60
VAL_SIZE = 100  # 10 per class


def _fake_labels(seed: int = 123) -> np.ndarray:
    """Balanced, shuffled labels: PER_CLASS of each of 10 classes."""
    labels = np.repeat(np.arange(N_CLASSES), PER_CLASS)
    return np.random.default_rng(seed).permutation(labels)


def test_stratified():
    labels = _fake_labels()
    train, val = make_stratified_split(labels, VAL_SIZE, split_seed=42)
    assert Counter(labels[val].tolist()) == {c: VAL_SIZE // N_CLASSES for c in range(N_CLASSES)}
    expected_train = PER_CLASS - VAL_SIZE // N_CLASSES
    assert Counter(labels[train].tolist()) == {c: expected_train for c in range(N_CLASSES)}


def test_disjoint_and_complete():
    labels = _fake_labels()
    train, val = make_stratified_split(labels, VAL_SIZE, split_seed=42)
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(len(labels)))
    assert len(val) == VAL_SIZE


def test_deterministic_and_seed_dependent():
    labels = _fake_labels()
    a = make_stratified_split(labels, VAL_SIZE, split_seed=42)
    b = make_stratified_split(labels, VAL_SIZE, split_seed=42)
    c = make_stratified_split(labels, VAL_SIZE, split_seed=7)
    assert a == b
    assert a[1] != c[1]


def test_unaffected_by_global_rng():
    labels = _fake_labels()
    np.random.seed(0)
    a = make_stratified_split(labels, VAL_SIZE, split_seed=42)
    np.random.seed(2)
    b = make_stratified_split(labels, VAL_SIZE, split_seed=42)
    assert a == b


def test_val_size_not_divisible_raises():
    with pytest.raises(ValueError, match="divisible"):
        make_stratified_split(_fake_labels(), 105, split_seed=42)


def test_file_round_trip(tmp_path):
    path = tmp_path / "split.json"
    labels = _fake_labels()
    first = load_or_create_split(path, lambda: labels, VAL_SIZE, 42)
    data = json.loads(path.read_text())
    assert set(data) == {"split_seed", "val_size", "val_indices", "train_indices"}
    assert (data["split_seed"], data["val_size"]) == (42, VAL_SIZE)

    def must_not_be_called():
        raise AssertionError("labels must not be reloaded when the file exists")

    second = load_or_create_split(path, must_not_be_called, VAL_SIZE, 42)
    assert first == second


@pytest.mark.parametrize("val_size, split_seed", [(VAL_SIZE, 7), (200, 42)])
def test_mismatch_raises_and_keeps_file(tmp_path, val_size, split_seed):
    path = tmp_path / "split.json"
    labels = _fake_labels()
    load_or_create_split(path, lambda: labels, VAL_SIZE, 42)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="fixed"):
        load_or_create_split(path, lambda: labels, val_size, split_seed)
    assert path.read_bytes() == before
