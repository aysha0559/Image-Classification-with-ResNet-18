"""Build the demo's bundled images (Phase 11): python -m app.build_assets [--data-dir DIR].

Writes, from the CIFAR-10 test set and CIFAR-10-C (both test-only data, RULES section 3):
- `app/assets/samples.npz`: the first 10 test images of each class (100 images, chosen by index
  only, never by any result): `images` uint8 (100, 32, 32, 3), `labels`, `test_indices`.
- `app/assets/samples_c.npz`: their CIFAR-10-C versions for the 15 corruptions and severities
  1-5, `images` uint8 (15, 5, 100, 32, 32, 3), plus `corruptions`, `severities`, `test_indices`.
  If that file is over 25 MB, it is rebuilt with severities 1, 3 and 5 only (Phase 11 BUILD).

This module runs only where the data is (not in the Docker image) and is the demo asset builder
that RULES section 3 allows to import `src.data.cifar10c`. Nothing here feeds training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.constants import CLASS_NAMES, CORRUPTIONS, SEVERITIES
from src.data.cifar10 import get_test_dataset
from src.data.cifar10c import load_cifar10c
from src.utils import env_path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
PER_CLASS = 10
MAX_BYTES = 25 * 1024 * 1024
FALLBACK_SEVERITIES = (1, 3, 5)


def pick_indices(labels: np.ndarray, per_class: int = PER_CLASS) -> np.ndarray:
    """Return the first `per_class` test indices of each class, class by class."""
    picked = [np.flatnonzero(labels == c)[:per_class] for c in range(len(CLASS_NAMES))]
    if any(len(p) < per_class for p in picked):
        raise ValueError(f"fewer than {per_class} images in some class")
    return np.concatenate(picked).astype(np.int64)


def build_corrupted(
    data_dir: str, indices: np.ndarray, labels: np.ndarray, severities
) -> np.ndarray:
    """Return the CIFAR-10-C images of `indices`, uint8 (15, len(severities), N, 32, 32, 3)."""
    out = np.empty((len(CORRUPTIONS), len(severities), len(indices), 32, 32, 3), dtype=np.uint8)
    for i, corruption in enumerate(CORRUPTIONS):
        for j, severity in enumerate(severities):
            images, c_labels = load_cifar10c(data_dir, corruption, severity)
            if not np.array_equal(c_labels[indices], labels):
                raise ValueError(f"{corruption} s{severity}: labels differ from the test set")
            out[i, j] = images[indices]
    return out


def write_npz(path: Path, **arrays) -> int:
    """Write a compressed npz and return its size in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path.stat().st_size


def build(data_dir: str, assets_dir: Path = ASSETS_DIR) -> None:
    """Write samples.npz and samples_c.npz (see the module docstring)."""
    test = get_test_dataset(data_dir)
    all_labels = np.asarray(test.targets, dtype=np.int64)
    indices = pick_indices(all_labels)
    images = np.asarray(test.data)[indices]
    labels = all_labels[indices]
    size = write_npz(assets_dir / "samples.npz", images=images, labels=labels, test_indices=indices)
    print(f"wrote {assets_dir / 'samples.npz'} ({size / 1e6:.2f} MB, {len(indices)} images)")

    for severities in (tuple(SEVERITIES), FALLBACK_SEVERITIES):
        corrupted = build_corrupted(data_dir, indices, labels, severities)
        size = write_npz(
            assets_dir / "samples_c.npz",
            images=corrupted,
            corruptions=np.array(CORRUPTIONS),
            severities=np.array(severities, dtype=np.int64),
            test_indices=indices,
        )
        print(
            f"wrote {assets_dir / 'samples_c.npz'} ({size / 1e6:.2f} MB, severities {severities})"
        )
        if size <= MAX_BYTES:
            return
        print(f"over {MAX_BYTES / 1e6:.1f} MB; rebuilding with severities {FALLBACK_SEVERITIES}")
    raise RuntimeError("samples_c.npz is still over the size limit")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None, help="default: $DATA_DIR or ./data")
    args = parser.parse_args(argv)
    build(args.data_dir or str(env_path("DATA_DIR")))


if __name__ == "__main__":
    main()
