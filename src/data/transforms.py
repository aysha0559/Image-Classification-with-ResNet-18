"""Train and eval transforms per strategy.

Phase 01 adds `build_eval_transform` (ToTensor + Normalize, no randomness).
Phase 03 adds `build_train_transform(strategy, params)` for S0-S3 (DECISIONS D5).
"""

from __future__ import annotations

from collections.abc import Callable

from torchvision import transforms as T


def build_eval_transform() -> Callable:
    """ToTensor + Normalize with the training-set stats (DECISIONS D2). Deterministic."""
    from src.data.cifar10 import get_normalization_stats

    mean, std = get_normalization_stats()
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])
