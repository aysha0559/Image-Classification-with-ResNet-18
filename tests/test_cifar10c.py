"""CIFAR-10-C loader on a fake, tiny CIFAR-10-C folder (no download)."""

import numpy as np
import pytest
import torch

from src.constants import CORRUPTIONS
from src.data import cifar10c
from src.data.cifar10c import CIFAR10C, load_cifar10c

N = 5  # images per severity in the fake folder


@pytest.fixture
def fake_dir(tmp_path, monkeypatch):
    """Fake DATA_DIR with CIFAR-10-C/{fog,gaussian_noise}.npy and labels.npy, N per severity.

    Pixel value of every image = its severity * 10 + its position in the severity block, so
    each image says which rows it came from. Labels are (row % 10).
    """
    monkeypatch.setattr(cifar10c, "N_PER_SEVERITY", N)
    folder = tmp_path / "CIFAR-10-C"
    folder.mkdir()
    rows = np.arange(5 * N)
    values = (rows // N + 1) * 10 + rows % N
    images = np.broadcast_to(values[:, None, None, None], (5 * N, 32, 32, 3)).astype(np.uint8)
    for name in ("fog", "gaussian_noise"):
        np.save(folder / f"{name}.npy", images)
    np.save(folder / "labels.npy", (rows % 10).astype(np.uint8))
    return tmp_path


@pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
def test_slices_rows_of_severity(fake_dir, severity):
    images, labels = load_cifar10c(str(fake_dir), "fog", severity)
    assert images.shape == (N, 32, 32, 3)
    assert images.dtype == np.uint8
    expected = severity * 10 + np.arange(N)
    assert np.array_equal(images[:, 0, 0, 0], expected)
    assert np.array_equal(labels, np.arange((severity - 1) * N, severity * N) % 10)


def test_label_dtype_is_int64(fake_dir):
    _, labels = load_cifar10c(str(fake_dir), "gaussian_noise", 2)
    assert labels.dtype == np.int64
    assert labels.shape == (N,)


@pytest.mark.parametrize("name", ["fogg", "clean", "Fog", "", "labels"])
def test_bad_corruption_name_raises(fake_dir, name):
    with pytest.raises(ValueError, match="unknown corruption"):
        load_cifar10c(str(fake_dir), name, 1)


@pytest.mark.parametrize("severity", [0, 6, -1, 2.0, "3", True])
def test_bad_severity_raises(fake_dir, severity):
    with pytest.raises(ValueError, match="severity"):
        load_cifar10c(str(fake_dir), "fog", severity)


def test_wrong_row_count_raises(fake_dir, monkeypatch):
    monkeypatch.setattr(cifar10c, "N_PER_SEVERITY", N + 1)
    with pytest.raises(ValueError, match="rows"):
        load_cifar10c(str(fake_dir), "fog", 1)


def test_missing_file_raises(fake_dir):
    with pytest.raises(FileNotFoundError):
        load_cifar10c(str(fake_dir), "snow", 1)


def test_all_standard_corruptions_accepted_by_validation():
    for name in CORRUPTIONS:
        cifar10c.validate_condition(name, 3)


def test_dataset_default_transform(fake_dir):
    ds = CIFAR10C("fog", 4, data_dir=fake_dir)
    assert len(ds) == N
    x, y = ds[2]
    assert isinstance(x, torch.Tensor) and x.shape == (3, 32, 32) and x.dtype == torch.float32
    assert y == ((4 - 1) * N + 2) % 10
    # Default transform normalises: pixel 42 -> (42/255 - mean) / std, not 42/255.
    from src.data.cifar10 import get_normalization_stats

    mean, std = get_normalization_stats()
    want = (42 / 255 - mean[0]) / std[0]
    assert torch.allclose(x[0, 0, 0], torch.tensor(want, dtype=torch.float32), atol=1e-5)


def test_dataset_custom_transform(fake_dir):
    ds = CIFAR10C("fog", 1, transform=lambda img: np.asarray(img), data_dir=fake_dir)
    x, _ = ds[0]
    assert x.shape == (32, 32, 3) and x[0, 0, 0] == 10
