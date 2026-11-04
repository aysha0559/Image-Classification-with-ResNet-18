"""Clean CIFAR-10: the fixed stratified train/validation split, normalisation stats, loaders.

The split (DECISIONS D1) and the stats (D2) are computed once and stored in `splits/`;
every later call reads those files, so all runs use the identical split and values.
The CIFAR-10 test set is only returned by `get_test_dataset` (final evaluation only).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from src.constants import NUM_CLASSES
from src.data.transforms import build_eval_transform, build_train_transform
from src.utils import REPO_ROOT, env_path, get_device, make_generator, seed_worker

SPLITS_DIR = REPO_ROOT / "splits"
SPLIT_FILE = SPLITS_DIR / "cifar10_val_indices.json"
STATS_FILE = SPLITS_DIR / "cifar10_stats.json"

N_TRAIN_FULL = 50_000
N_TEST = 10_000


# ---------------------------------------------------------------- raw data


def _load_cifar10(data_dir: str | Path, train: bool, transform: Callable | None = None) -> CIFAR10:
    """Load (downloading if needed) a CIFAR-10 split and check its size."""
    ds = CIFAR10(root=str(data_dir), train=train, download=True, transform=transform)
    expected = N_TRAIN_FULL if train else N_TEST
    if len(ds) != expected:
        raise RuntimeError(
            f"CIFAR-10 {'train' if train else 'test'} set has {len(ds)} images, expected {expected}"
        )
    return ds


# ---------------------------------------------------------------- split


def make_stratified_split(
    labels: Sequence[int] | np.ndarray, val_size: int, split_seed: int
) -> tuple[list[int], list[int]]:
    """Return sorted (train_indices, val_indices) with `val_size / n_classes` val images per class.

    Depends only on `labels`, `val_size` and `split_seed`.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if val_size % len(classes) != 0:
        raise ValueError(f"val_size {val_size} is not divisible by {len(classes)} classes")
    per_class = val_size // len(classes)
    rng = np.random.default_rng(split_seed)
    val: list[int] = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        if len(idx) < per_class:
            raise ValueError(f"class {c} has {len(idx)} images, fewer than {per_class}")
        val.extend(rng.permutation(idx)[:per_class].tolist())
    val_set = set(val)
    train = [i for i in range(len(labels)) if i not in val_set]
    return train, sorted(val)


def load_or_create_split(
    path: str | Path,
    labels_fn: Callable[[], Sequence[int] | np.ndarray],
    val_size: int,
    split_seed: int,
) -> tuple[list[int], list[int]]:
    """Read the split from `path`; if missing, build it from `labels_fn()` and write it.

    Raises ValueError if the file's `split_seed` or `val_size` differ from the arguments
    (the file is never overwritten in that case).
    """
    path = Path(path)
    if path.exists():
        data = json.loads(path.read_text())
        if data["split_seed"] != split_seed or data["val_size"] != val_size:
            raise ValueError(
                f"{path} holds split_seed={data['split_seed']}, val_size={data['val_size']}, "
                f"but split_seed={split_seed}, val_size={val_size} was requested. The split is "
                "fixed (DECISIONS D1); fix the config instead of the file."
            )
        return data["train_indices"], data["val_indices"]
    train, val = make_stratified_split(labels_fn(), val_size, split_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_seed": split_seed,
        "val_size": val_size,
        "val_indices": val,
        "train_indices": train,
    }
    path.write_text(json.dumps(payload) + "\n")
    return train, val


def get_split_indices(
    data_dir: str, val_size: int = 5000, split_seed: int = 42
) -> tuple[list[int], list[int]]:
    """Return (train_indices, val_indices) into the CIFAR-10 train set (CONTRACTS C10, D1)."""
    return load_or_create_split(
        SPLIT_FILE,
        lambda: _load_cifar10(data_dir, train=True).targets,
        val_size,
        split_seed,
    )


# ---------------------------------------------------------------- normalisation stats


def compute_normalization_stats(
    data_dir: str, val_size: int = 5000, split_seed: int = 42
) -> tuple[list[float], list[float]]:
    """Compute per-channel mean/std over the training split only (pixels in [0,1]); write the file.

    Std is the population std (ddof=0) over all training pixels of each channel.
    """
    train_idx, _ = get_split_indices(data_dir, val_size, split_seed)
    pixels = _load_cifar10(data_dir, train=True).data[train_idx].astype(np.float64) / 255.0
    mean = pixels.mean(axis=(0, 1, 2)).tolist()
    std = pixels.std(axis=(0, 1, 2)).tolist()
    payload = {
        "mean": mean,
        "std": std,
        "n_images": len(train_idx),
        "split_seed": split_seed,
        "val_size": val_size,
        "computed_on": "CIFAR-10 train split (train_indices of cifar10_val_indices.json), /255",
    }
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps(payload, indent=2) + "\n")
    return mean, std


def get_normalization_stats() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return (mean, std) per RGB channel from `splits/cifar10_stats.json` (DECISIONS D2)."""
    if not STATS_FILE.exists():
        raise FileNotFoundError(
            f"{STATS_FILE} is missing; run "
            "`python -c 'from src.data.cifar10 import compute_normalization_stats as c; "
            "c(\"./data\")'` once"
        )
    data = json.loads(STATS_FILE.read_text())
    return tuple(data["mean"]), tuple(data["std"])


# ---------------------------------------------------------------- exploration helpers


def split_class_counts(
    data_dir: str, val_size: int = 5000, split_seed: int = 42
) -> dict[str, list[int]]:
    """Return per-class image counts (class index order) for "train", "val" and "test"."""
    train_idx, val_idx = get_split_indices(data_dir, val_size, split_seed)
    train_labels = np.asarray(_load_cifar10(data_dir, train=True).targets)
    test_labels = np.asarray(get_test_dataset(data_dir).targets)
    return {
        name: np.bincount(labels, minlength=NUM_CLASSES).tolist()
        for name, labels in (
            ("train", train_labels[train_idx]),
            ("val", train_labels[val_idx]),
            ("test", test_labels),
        )
    }


def examples_per_class(
    data_dir: str, n_per_class: int = 10, val_size: int = 5000, split_seed: int = 42
) -> np.ndarray:
    """Return uint8 images of shape (10, n_per_class, 32, 32, 3): the first training-split
    images of each class, in index order (no randomness)."""
    train_idx, _ = get_split_indices(data_dir, val_size, split_seed)
    ds = _load_cifar10(data_dir, train=True)
    labels = np.asarray(ds.targets)[train_idx]
    train_idx = np.asarray(train_idx)
    return np.stack(
        [ds.data[train_idx[labels == c][:n_per_class]] for c in range(NUM_CLASSES)]
    )


# ---------------------------------------------------------------- datasets


class TransformedSubset(Dataset):
    """Items `indices` of `base` (which yields (PIL image, label)), with `transform` on the image."""

    def __init__(self, base: Dataset, indices: Sequence[int], transform: Callable | None) -> None:
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[Any, int]:
        img, label = self.base[self.indices[i]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_train_val_datasets(
    cfg: dict, train_transform: Callable | None = None, data_dir: str | None = None
) -> tuple[Dataset, Dataset]:
    """Return (train, val) datasets from the fixed split; honours `cfg["subset"]` (first N each).

    `train_transform` defaults to the pipeline of `cfg["strategy"]`.
    Validation always uses `build_eval_transform()`.
    """
    data_dir = data_dir or str(env_path("DATA_DIR"))
    train_idx, val_idx = get_split_indices(data_dir, cfg["val_size"], cfg["split_seed"])
    if cfg.get("subset"):
        n = cfg["subset"]
        train_idx, val_idx = train_idx[:n], val_idx[:n]
    if train_transform is None:
        train_transform = build_train_transform(cfg["strategy"], cfg["strategy_params"])
    base = _load_cifar10(data_dir, train=True)
    return (
        TransformedSubset(base, train_idx, train_transform),
        TransformedSubset(base, val_idx, build_eval_transform()),
    )


def get_test_dataset(data_dir: str) -> Dataset:
    """Return the 10,000-image CIFAR-10 test set with the eval transform (final evaluation only)."""
    return _load_cifar10(data_dir, train=False, transform=build_eval_transform())


def make_loaders(
    cfg: dict, train_transform: Callable | None = None, data_dir: str | None = None
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader). Train is shuffled with a generator seeded by cfg seed."""
    train_ds, val_ds = get_train_val_datasets(cfg, train_transform, data_dir)
    common = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
        "pin_memory": get_device(cfg["device"]).type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": cfg["num_workers"] > 0,
    }
    train_loader = DataLoader(
        train_ds, shuffle=True, generator=make_generator(cfg["seed"]), **common
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader
