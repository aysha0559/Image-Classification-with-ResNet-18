"""Our own versions of the 4 seen corruptions, used only for S3 training (DECISIONS D5, D6).

Everything here works on the image it is given. This module reads no dataset files.

Reference followed (read from source, not from memory):
    Hendrycks & Dietterich (2019), "Benchmarking Neural Network Robustness to Common
    Corruptions and Perturbations".
    github.com/hendrycks/robustness, file `ImageNet-C/create_c/make_cifar_c.py`,
    commit ca360085d57889288c4143fddd8a707bbcae01bd.

Parameters per severity 1-5, as in that file:
    gaussian_noise    std on pixels in [0, 1]:         0.04, 0.06, 0.08, 0.09, 0.10
    defocus_blur      (disk radius, alias blur sigma): (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
                                                       (1, 0.2), (1.5, 0.1)
    contrast          factor around the channel mean:  0.75, 0.5, 0.4, 0.3, 0.15
    jpeg_compression  JPEG quality:                    80, 65, 58, 50, 40

Like the reference, float results are clipped and then truncated to uint8, not rounded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image

from src.constants import SEEN_CORRUPTIONS, SEVERITIES

# OpenCV's own thread pool can hang inside forked DataLoader workers.
cv2.setNumThreads(0)

GAUSSIAN_NOISE_STD = (0.04, 0.06, 0.08, 0.09, 0.10)
DEFOCUS_DISK = ((0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1.0, 0.2), (1.5, 0.1))
CONTRAST_FACTOR = (0.75, 0.5, 0.4, 0.3, 0.15)
JPEG_QUALITY = (80, 65, 58, 50, 40)

DISK_HALF_WIDTH = 8  # kernels span -8..8, as in the reference for every radius used here


def _to_uint8(x: np.ndarray) -> np.ndarray:
    """Map floats in [0, 1] (clipping first) to uint8 by truncation, as the reference does."""
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _disk_kernel(radius: float, alias_blur: float) -> np.ndarray:
    """Return a normalised disk of `radius`, softened by a 3x3 Gaussian of sigma `alias_blur`."""
    span = np.arange(-DISK_HALF_WIDTH, DISK_HALF_WIDTH + 1)
    xx, yy = np.meshgrid(span, span)
    disk = np.array(xx**2 + yy**2 <= radius**2, dtype=np.float32)
    disk /= disk.sum()
    return cv2.GaussianBlur(disk, ksize=(3, 3), sigmaX=alias_blur)


DEFOCUS_KERNELS = tuple(_disk_kernel(radius, sigma) for radius, sigma in DEFOCUS_DISK)


def _gaussian_noise(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise, drawn independently per pixel and channel."""
    x = img / 255.0
    return _to_uint8(x + rng.normal(scale=GAUSSIAN_NOISE_STD[severity - 1], size=x.shape))


def _defocus_blur(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Convolve each channel with the severity's disk kernel. Uses no randomness."""
    return _to_uint8(cv2.filter2D(img / 255.0, -1, DEFOCUS_KERNELS[severity - 1]))


def _contrast(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Scale each channel towards its own mean. Uses no randomness."""
    x = img / 255.0
    means = x.mean(axis=(0, 1), keepdims=True)
    return _to_uint8((x - means) * CONTRAST_FACTOR[severity - 1] + means)


def _jpeg_compression(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Encode to JPEG in memory with Pillow's defaults and decode. Uses no randomness."""
    buffer = BytesIO()
    Image.fromarray(img).save(buffer, "JPEG", quality=JPEG_QUALITY[severity - 1])
    return np.array(Image.open(buffer))


_CORRUPTIONS: dict[str, Callable[[np.ndarray, int, np.random.Generator], np.ndarray]] = {
    "gaussian_noise": _gaussian_noise,
    "defocus_blur": _defocus_blur,
    "contrast": _contrast,
    "jpeg_compression": _jpeg_compression,
}


def _check_name(name: str) -> None:
    """Raise ValueError unless `name` is one of the seen corruptions."""
    if name not in _CORRUPTIONS:
        raise ValueError(
            f"unsupported corruption {name!r}; allowed: {', '.join(SEEN_CORRUPTIONS)}"
        )


def _check_severity(severity: int) -> None:
    """Raise ValueError unless `severity` is an int in 1..5."""
    if isinstance(severity, bool) or severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {list(SEVERITIES)}, got {severity!r}")


def apply_corruption(
    img: np.ndarray, name: str, severity: int, rng: np.random.Generator
) -> np.ndarray:
    """Return `img` (uint8 HWC) corrupted by seen corruption `name` at `severity` (1-5).

    Deterministic given the state of `rng`. Raises ValueError for any other name or severity.
    """
    _check_name(name)
    _check_severity(severity)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected a uint8 HWC RGB image, got {img.dtype} {img.shape}")
    return _CORRUPTIONS[name](img, severity, rng)


class RandomSeenCorruption:
    """With probability `p`, apply one corruption at one severity, each chosen uniformly.

    Takes and returns PIL images. The generator is seeded from `torch.initial_seed()` and is
    rebuilt whenever that seed changes, so every DataLoader worker gets its own stream, even
    if this object was already used in the parent process before the workers started.
    """

    def __init__(
        self,
        p: float = 0.5,
        corruptions: Sequence[str] = SEEN_CORRUPTIONS,
        severities: Sequence[int] = SEVERITIES,
    ) -> None:
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p!r}")
        if not corruptions or not severities:
            raise ValueError("corruptions and severities must not be empty")
        for name in corruptions:
            _check_name(name)
        for severity in severities:
            _check_severity(severity)
        self.p = p
        self.corruptions = tuple(corruptions)
        self.severities = tuple(severities)
        self._rng: np.random.Generator | None = None
        self._rng_seed: int | None = None

    def _generator(self) -> np.random.Generator:
        """Return the generator for the current process, seeding it on first use there."""
        seed = torch.initial_seed()
        if self._rng is None or seed != self._rng_seed:
            self._rng = np.random.default_rng(seed)
            self._rng_seed = seed
        return self._rng

    def __call__(self, img: Image.Image) -> Image.Image:
        """Return `img` itself, or a corrupted copy of it, as a PIL image."""
        rng = self._generator()
        if rng.random() >= self.p:
            return img
        name = self.corruptions[rng.integers(len(self.corruptions))]
        severity = self.severities[rng.integers(len(self.severities))]
        return Image.fromarray(apply_corruption(np.asarray(img), name, severity, rng))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(p={self.p}, corruptions={list(self.corruptions)}, "
            f"severities={list(self.severities)})"
        )
