"""Train and eval transforms per strategy (DECISIONS D2, D5)."""

from __future__ import annotations

from collections.abc import Callable

from torchvision import transforms as T

from src.constants import STRATEGIES
from src.data.corruptions import RandomSeenCorruption


def _to_normalized_tensor() -> list[Callable]:
    """ToTensor + Normalize with the training-set stats (DECISIONS D2)."""
    from src.data.cifar10 import get_normalization_stats

    mean, std = get_normalization_stats()
    return [T.ToTensor(), T.Normalize(mean, std)]


def _crop_and_flip() -> list[Callable]:
    """The S1 augmentation, shared by S1, S2 and S3."""
    return [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]


def build_train_transform(strategy: str, params: dict) -> Callable:
    """Return the training pipeline of `strategy` (DECISIONS D5).

    `params` is the config's `strategy_params` block; `augmix` and `corruption_aware` read
    their own section of it. Raises ValueError for an unknown strategy.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; allowed: {', '.join(STRATEGIES)}")
    if strategy == "none":
        steps = []
    elif strategy == "standard":
        steps = _crop_and_flip()
    elif strategy == "augmix":
        steps = [*_crop_and_flip(), T.AugMix(**params["augmix"])]
    else:
        steps = [*_crop_and_flip(), RandomSeenCorruption(**params["corruption_aware"])]
    return T.Compose([*steps, *_to_normalized_tensor()])


def build_eval_transform() -> Callable:
    """ToTensor + Normalize with the training-set stats (DECISIONS D2). Deterministic."""
    return T.Compose(_to_normalized_tensor())
