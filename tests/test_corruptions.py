"""Our seen corruptions and RandomSeenCorruption, on synthetic images only."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.constants import SEEN_CORRUPTIONS, SEVERITIES, UNSEEN_CORRUPTIONS
from src.data import corruptions
from src.data.corruptions import RandomSeenCorruption, apply_corruption
from src.utils import make_generator, seed_worker


def fake_image(seed: int = 0) -> np.ndarray:
    """A 32x32 RGB image with smooth colour gradients, hard edges and mild texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:32, 0:32]
    img = np.stack([xx * 6.0 + 30, yy * 6.0 + 30, (xx + yy) * 3.0 + 30], axis=-1)
    img[8:24, 8:24] = (220, 40, 40)
    img[12:20, 2:30, 1] = 200
    return np.clip(img + rng.normal(scale=4, size=img.shape), 0, 255).astype(np.uint8)


def mean_change(img: np.ndarray, name: str, severity: int) -> float:
    out = apply_corruption(img, name, severity, np.random.default_rng(0))
    return float(np.abs(out.astype(float) - img.astype(float)).mean())


@pytest.mark.parametrize("name", SEEN_CORRUPTIONS)
@pytest.mark.parametrize("severity", SEVERITIES)
def test_output_contract(name, severity):
    img = fake_image()
    before = img.copy()
    out = apply_corruption(img, name, severity, np.random.default_rng(0))
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    assert np.array_equal(img, before), "the input must not be modified"


@pytest.mark.parametrize("name", SEEN_CORRUPTIONS)
def test_same_seed_same_output(name):
    img = fake_image()
    a = apply_corruption(img, name, 3, np.random.default_rng(7))
    b = apply_corruption(img, name, 3, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_noise_differs_between_seeds():
    img = fake_image()
    a = apply_corruption(img, "gaussian_noise", 3, np.random.default_rng(1))
    b = apply_corruption(img, "gaussian_noise", 3, np.random.default_rng(2))
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("name", SEEN_CORRUPTIONS)
def test_severity_5_changes_more_than_severity_1(name):
    img = fake_image()
    assert mean_change(img, name, 5) > mean_change(img, name, 1) > 0


def test_contrast_keeps_channel_means_close():
    img = fake_image()
    out = apply_corruption(img, "contrast", 5, np.random.default_rng(0))
    assert np.abs(out.mean(axis=(0, 1)) - img.mean(axis=(0, 1))).max() < 1.5
    assert out.std() < img.std()


def test_defocus_kernels_sum_to_one():
    for kernel in corruptions.DEFOCUS_KERNELS:
        assert kernel.shape == (17, 17)
        assert kernel.sum() == pytest.approx(1.0, abs=1e-5)


def test_supported_set_is_exactly_the_seen_list():
    assert set(corruptions._CORRUPTIONS) == set(SEEN_CORRUPTIONS)


@pytest.mark.parametrize("name", UNSEEN_CORRUPTIONS)
def test_unseen_name_raises_and_lists_allowed(name):
    with pytest.raises(ValueError, match="gaussian_noise, defocus_blur, contrast, jpeg"):
        apply_corruption(fake_image(), name, 1, np.random.default_rng(0))


@pytest.mark.parametrize("severity", [0, 6, -1, 2.5, "3", True])
def test_bad_severity_raises(severity):
    with pytest.raises(ValueError, match="severity"):
        apply_corruption(fake_image(), "contrast", severity, np.random.default_rng(0))


def test_bad_image_raises():
    with pytest.raises(ValueError, match="uint8 HWC"):
        apply_corruption(np.zeros((32, 32, 3)), "contrast", 1, np.random.default_rng(0))


def test_random_corruption_rejects_unseen_at_build_time():
    with pytest.raises(ValueError, match="unsupported corruption 'fog'"):
        RandomSeenCorruption(corruptions=["fog"])
    with pytest.raises(ValueError, match="p must be"):
        RandomSeenCorruption(p=1.5)


def count_changed(p: float, n: int = 200) -> int:
    transform = RandomSeenCorruption(p=p)
    img = Image.fromarray(fake_image())
    return sum(not np.array_equal(transform(img), np.asarray(img)) for _ in range(n))


def test_probability_extremes():
    assert count_changed(0.0) == 0
    assert count_changed(1.0) == 200


def test_probability_half():
    torch.manual_seed(0)
    assert 400 <= count_changed(0.5, n=1000) <= 600


def test_all_corruptions_and_severities_get_picked(monkeypatch):
    seen = []

    def spy(img, name, severity, rng):
        seen.append((name, severity))
        return img

    monkeypatch.setattr(corruptions, "apply_corruption", spy)
    transform = RandomSeenCorruption(p=1.0)
    img = Image.fromarray(fake_image())
    for _ in range(400):
        transform(img)
    assert set(seen) == {(c, s) for c in SEEN_CORRUPTIONS for s in SEVERITIES}


def test_returns_pil_image():
    out = RandomSeenCorruption(p=1.0)(Image.fromarray(fake_image()))
    assert isinstance(out, Image.Image) and out.size == (32, 32) and out.mode == "RGB"


def test_reproducible_from_the_torch_seed():
    img = Image.fromarray(fake_image())

    def run() -> list[np.ndarray]:
        torch.manual_seed(3)
        transform = RandomSeenCorruption(p=1.0)
        return [np.asarray(transform(img)) for _ in range(20)]

    assert all(np.array_equal(a, b) for a, b in zip(run(), run(), strict=True))


class SameImage(Dataset):
    """The same image every time, through `transform`, so differences come from the transform."""

    def __init__(self, transform: RandomSeenCorruption) -> None:
        self.img = Image.fromarray(fake_image())
        self.transform = transform

    def __len__(self) -> int:
        return 40

    def __getitem__(self, i: int) -> np.ndarray:
        return np.array(self.transform(self.img))


def load_all(transform: RandomSeenCorruption, seed: int) -> np.ndarray:
    loader = DataLoader(
        SameImage(transform),
        batch_size=1,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
    return torch.cat(list(loader)).numpy()


@pytest.mark.parametrize("used_in_parent_first", [False, True])
def test_workers_get_different_streams(used_in_parent_first):
    transform = RandomSeenCorruption(p=1.0, corruptions=["gaussian_noise"], severities=[5])
    if used_in_parent_first:
        transform(Image.fromarray(fake_image()))
    out = load_all(transform, seed=0)
    worker_0, worker_1 = out[0::2], out[1::2]  # the loader alternates between workers
    assert not np.array_equal(worker_0, worker_1)
    assert len({a.tobytes() for a in out}) == len(out)


def test_worker_streams_are_reproducible():
    build = lambda: RandomSeenCorruption(p=1.0)  # noqa: E731
    assert np.array_equal(load_all(build(), seed=5), load_all(build(), seed=5))
    assert not np.array_equal(load_all(build(), seed=5), load_all(build(), seed=6))
