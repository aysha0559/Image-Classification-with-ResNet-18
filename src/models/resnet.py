"""ResNet-18 adapted for 32x32 images (PROJECT_OVERVIEW 7.2), written from He et al. 2016."""

from __future__ import annotations

import torch
from torch import nn

STEM_WIDTH = 64
STAGE_WIDTHS = (64, 128, 256, 512)
BLOCKS_PER_STAGE = (2, 2, 2, 2)


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """Return a 3x3 conv with padding 1 and no bias (BatchNorm follows it)."""
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """Two 3x3 convs plus a shortcut, which is a 1x1 projection when the shape changes."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the residual output, same spatial size as `x` divided by the stride."""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


def _make_stage(in_channels: int, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
    """Return `n_blocks` BasicBlocks; only the first one may downsample."""
    blocks = [BasicBlock(in_channels, out_channels, stride)]
    blocks += [BasicBlock(out_channels, out_channels) for _ in range(n_blocks - 1)]
    return nn.Sequential(*blocks)


class ResNet18(nn.Module):
    """ResNet-18 for 3x32x32 images.

    BasicBlocks [2, 2, 2, 2] with widths 64/128/256/512. The stem is Conv3x3 stride 1 padding 1
    with no max-pool (the ImageNet stem shrinks small images too much). Stages 2-4 downsample by
    stride 2. Global average pool, then Linear(512 -> num_classes).

    Init: Kaiming normal on convs, BatchNorm weight 1 and bias 0.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = _conv3x3(3, STEM_WIDTH)
        self.bn1 = nn.BatchNorm2d(STEM_WIDTH)
        self.relu = nn.ReLU(inplace=True)
        widths = (STEM_WIDTH, *STAGE_WIDTHS)
        self.layer1 = _make_stage(widths[0], widths[1], BLOCKS_PER_STAGE[0], stride=1)
        self.layer2 = _make_stage(widths[1], widths[2], BLOCKS_PER_STAGE[1], stride=2)
        self.layer3 = _make_stage(widths[2], widths[3], BLOCKS_PER_STAGE[2], stride=2)
        self.layer4 = _make_stage(widths[3], widths[4], BLOCKS_PER_STAGE[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(STAGE_WIDTHS[-1], num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming normal to convs and set BatchNorm weight 1, bias 0."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (N, num_classes) for images of shape (N, 3, 32, 32)."""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return self.fc(torch.flatten(self.avgpool(out), 1))
