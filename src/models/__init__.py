"""Model registry: `build_model(name)` for the names in CONTRACTS C3."""

from __future__ import annotations

from torch import nn

from src.models.resnet import ResNet18
from src.models.simple_cnn import SimpleCNN

_MODELS: dict[str, type[nn.Module]] = {"simple_cnn": SimpleCNN, "resnet18": ResNet18}


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """Return a new, untrained model; raise ValueError if `name` is unknown."""
    if name not in _MODELS:
        raise ValueError(f"unknown model {name!r}; allowed: {', '.join(_MODELS)}")
    return _MODELS[name](num_classes=num_classes)
