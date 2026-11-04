"""Simple CNN baseline (PROJECT_OVERVIEW 7.1, DECISIONS D11)."""

from __future__ import annotations

import torch
from torch import nn

CONV_PADDING = 1
BLOCK_DROPOUT = 0.25
CLASSIFIER_DROPOUT = 0.5
HIDDEN_FEATURES = 256
LAST_CONV_CHANNELS = 128
FEATURE_MAP_SIZE = 4  # 32x32 input after three 2x2 max-pools


def _conv_bn_relu(in_channels: int, out_channels: int) -> list[nn.Module]:
    """Return Conv3x3 -> BatchNorm -> ReLU. The conv has no bias because BatchNorm follows it."""
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=CONV_PADDING, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


class SimpleCNN(nn.Module):
    """Baseline CNN for 3x32x32 images: three conv blocks, then two Linear layers.

    Layers, in order:
        [Conv3x3 32, BN, ReLU, Conv3x3 32, BN, ReLU, MaxPool2, Dropout]
        [Conv3x3 64, BN, ReLU, Conv3x3 64, BN, ReLU, MaxPool2, Dropout]
        [Conv3x3 128, BN, ReLU, MaxPool2, Dropout]
        Flatten, Linear(->256), ReLU, Dropout, Linear(->num_classes)

    From DECISIONS D11: padding 1 on every 3x3 conv, dropout 0.25 after each conv block,
    dropout 0.5 before the last Linear.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *_conv_bn_relu(3, 32),
            *_conv_bn_relu(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout(BLOCK_DROPOUT),
            *_conv_bn_relu(32, 64),
            *_conv_bn_relu(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout(BLOCK_DROPOUT),
            *_conv_bn_relu(64, LAST_CONV_CHANNELS),
            nn.MaxPool2d(2),
            nn.Dropout(BLOCK_DROPOUT),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(LAST_CONV_CHANNELS * FEATURE_MAP_SIZE**2, HIDDEN_FEATURES),
            nn.ReLU(inplace=True),
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(HIDDEN_FEATURES, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (N, num_classes) for images of shape (N, 3, 32, 32)."""
        return self.classifier(self.features(x))
