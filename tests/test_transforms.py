"""Training and eval pipelines per strategy (DECISIONS D5), on synthetic images only."""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from src.config import load_config
from src.constants import SEEN_CORRUPTIONS, SEVERITIES, STRATEGIES
from src.data import cifar10
from src.data.corruptions import RandomSeenCorruption
from src.data.transforms import build_eval_transform, build_train_transform
from src.utils import REPO_ROOT

PARAMS = load_config(REPO_ROOT / "configs" / "cnn_standard.yaml", seed=0)["strategy_params"]


def fake_image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))


def ops(transform: T.Compose) -> list[type]:
    return [type(t) for t in transform.transforms]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_each_strategy_gives_a_float_tensor(strategy):
    out = build_train_transform(strategy, PARAMS)(fake_image())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.float32


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("none", [T.ToTensor, T.Normalize]),
        ("standard", [T.RandomCrop, T.RandomHorizontalFlip, T.ToTensor, T.Normalize]),
        ("augmix", [T.RandomCrop, T.RandomHorizontalFlip, T.AugMix, T.ToTensor, T.Normalize]),
        (
            "corruption_aware",
            [T.RandomCrop, T.RandomHorizontalFlip, RandomSeenCorruption, T.ToTensor, T.Normalize],
        ),
    ],
)
def test_pipeline_order(strategy, expected):
    assert ops(build_train_transform(strategy, PARAMS)) == expected


def test_crop_and_flip_arguments():
    crop, flip = build_train_transform("standard", PARAMS).transforms[:2]
    assert crop.size == (32, 32) and crop.padding == 4
    assert flip.p == 0.5


def test_augmix_uses_exactly_the_d5_parameters():
    augmix = build_train_transform("augmix", PARAMS).transforms[2]
    assert augmix.severity == 3
    assert augmix.mixture_width == 3
    assert augmix.chain_depth == -1
    assert augmix.alpha == 1.0
    assert augmix.all_ops is False


def test_corruption_aware_uses_the_d5_parameters():
    corruption = build_train_transform("corruption_aware", PARAMS).transforms[2]
    assert corruption.p == 0.5
    assert corruption.corruptions == SEEN_CORRUPTIONS
    assert corruption.severities == SEVERITIES


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_normalisation_comes_from_the_stats_file(strategy):
    mean, std = cifar10.get_normalization_stats()
    normalize = build_train_transform(strategy, PARAMS).transforms[-1]
    assert tuple(normalize.mean) == mean and tuple(normalize.std) == std


def test_none_and_eval_are_deterministic():
    img = fake_image()
    for transform in (build_train_transform("none", PARAMS), build_eval_transform()):
        torch.manual_seed(0)
        first = transform(img)
        torch.manual_seed(1)
        assert torch.equal(first, transform(img))


@pytest.mark.parametrize("strategy", ["standard", "augmix", "corruption_aware"])
def test_random_strategies_change_with_the_seed(strategy):
    img = fake_image()
    transform = build_train_transform(strategy, PARAMS)

    def run(seed: int) -> list[torch.Tensor]:
        torch.manual_seed(seed)
        return [transform(img) for _ in range(8)]

    a, b, c = run(0), run(1), run(0)
    assert any(not torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert all(torch.equal(x, y) for x, y in zip(a, c, strict=True))


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown strategy 'cutmix'"):
        build_train_transform("cutmix", PARAMS)


class FakeCifar:
    def __getitem__(self, i: int) -> tuple[Image.Image, int]:
        return fake_image(), 0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_datasets_use_the_strategy_for_train_and_eval_for_val(strategy, monkeypatch):
    monkeypatch.setattr(cifar10, "get_split_indices", lambda *a: (list(range(8)), [8, 9]))
    monkeypatch.setattr(cifar10, "_load_cifar10", lambda *a, **k: FakeCifar())
    cfg = load_config(REPO_ROOT / "configs" / f"cnn_{strategy}.yaml", seed=0)
    train, val = cifar10.get_train_val_datasets(cfg, data_dir="unused")
    assert ops(train.transform) == ops(build_train_transform(strategy, PARAMS))
    assert ops(val.transform) == ops(build_eval_transform())
    assert train[0][0].shape == val[0][0].shape == (3, 32, 32)
