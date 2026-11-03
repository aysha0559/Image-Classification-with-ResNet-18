"""CIFAR-10-C loader.

Test-only. Never import from training code.

CIFAR-10-C (Hendrycks & Dietterich, 2019) is a test-only benchmark (RULES section 3). This
module is imported only by `src/evaluate.py`, `scripts/make_sample_grid.py` and the Phase 11
demo asset builder. `tests/test_no_leakage.py` enforces that.

Layout under `DATA_DIR`: `CIFAR-10-C/{corruption}.npy`, uint8 (50000, 32, 32, 3), severity s
in rows `(s-1)*10000 .. s*10000-1`, plus `CIFAR-10-C/labels.npy` (50000,).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.constants import CORRUPTIONS, EXTRA_CORRUPTIONS, SEVERITIES
from src.utils import env_path

CIFAR10C_DIRNAME = "CIFAR-10-C"
N_PER_SEVERITY = 10_000  # images per severity; tests monkeypatch this to use tiny arrays
ALL_CORRUPTIONS: tuple[str, ...] = CORRUPTIONS + EXTRA_CORRUPTIONS


def cifar10c_dir(data_dir: str | Path) -> Path:
    """Return the `CIFAR-10-C/` folder inside `data_dir`."""
    return Path(data_dir) / CIFAR10C_DIRNAME


def validate_condition(corruption: str, severity: int) -> None:
    """Raise ValueError unless `corruption` is a CIFAR-10-C name and `severity` is in 1..5."""
    if corruption not in ALL_CORRUPTIONS:
        raise ValueError(
            f"unknown corruption {corruption!r}; expected one of {', '.join(ALL_CORRUPTIONS)}"
        )
    if isinstance(severity, bool) or not isinstance(severity, (int, np.integer)):
        raise ValueError(f"severity must be an int in 1..5, got {severity!r}")
    if int(severity) not in SEVERITIES:
        raise ValueError(f"severity must be in 1..5, got {severity}")


def _severity_rows(n_rows: int, severity: int, path: Path) -> slice:
    """Return the row slice of `severity`, checking the file has 5 * N_PER_SEVERITY rows."""
    expected = len(SEVERITIES) * N_PER_SEVERITY
    if n_rows != expected:
        raise ValueError(f"{path}: expected {expected} rows, found {n_rows}")
    start = (int(severity) - 1) * N_PER_SEVERITY
    return slice(start, start + N_PER_SEVERITY)


def load_cifar10c(data_dir: str, corruption: str, severity: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (images, labels) for one condition: uint8 (N, 32, 32, 3) and int64 (N,).

    The image file is memory-mapped and only the rows of `severity` are sliced, so the
    50,000-image file is never read into memory as a whole. N = N_PER_SEVERITY (10,000).
    """
    validate_condition(corruption, severity)
    folder = cifar10c_dir(data_dir)
    img_path = folder / f"{corruption}.npy"
    lbl_path = folder / "labels.npy"
    for p in (img_path, lbl_path):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found; run python -m scripts.download_cifar10c")
    images = np.load(img_path, mmap_mode="r")
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError(f"{img_path}: expected uint8 (N, 32, 32, 3), got {images.dtype} {images.shape}")
    labels = np.load(lbl_path, mmap_mode="r")
    if labels.shape != (images.shape[0],):
        raise ValueError(f"{lbl_path}: shape {labels.shape} does not match {img_path.name}")
    rows = _severity_rows(images.shape[0], severity, img_path)
    return np.asarray(images[rows]), np.asarray(labels[rows], dtype=np.int64)


class CIFAR10C(Dataset):
    """One CIFAR-10-C (corruption, severity) condition as a Dataset of (tensor, label).

    `transform` defaults to `build_eval_transform()` (ToTensor + Normalize with the training
    stats). `data_dir` defaults to `DATA_DIR` (CONTRACTS C2).
    """

    def __init__(
        self,
        corruption: str,
        severity: int,
        transform: Callable | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        if data_dir is None:
            data_dir = env_path("DATA_DIR")
        if transform is None:
            from src.data.transforms import build_eval_transform

            transform = build_eval_transform()
        self.corruption = corruption
        self.severity = int(severity)
        self.transform = transform
        self.images, self.labels = load_cifar10c(str(data_dir), corruption, severity)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.fromarray(np.array(self.images[idx]))
        return self.transform(img), int(self.labels[idx])
